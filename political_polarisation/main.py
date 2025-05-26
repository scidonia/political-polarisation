import pandas as pd
import pyarrow as pa
import torch
import os
import ctypes
from political_polarisation.phrasal import get_phrases, chunk_phrases, PhrasalChunk
from political_polarisation.context import *
from political_polarisation.utils import eprintln
import datafusion as df


def torch_type_to_ctype(torch_type):
    """Convert torch type to ctypes type."""
    if torch_type == torch.float32:
        return ctypes.c_float
    else:
        raise Exception(f"Unrecognized torch type: {torch_type}")


def dataframe_to_tensor(dataframe, tensor):
    """
    Copy data from a DataFusion dataframe to a PyTorch tensor.

    Args:
        dataframe: DataFusion dataframe with embedding column
        tensor: PyTorch tensor to copy data into
    """
    # Get the embedding column as a numpy array
    embeddings = dataframe.to_pandas()["embedding"].to_numpy()

    # Flatten the tensor for easier copying
    flat_tensor = tensor.view(-1)

    # Get the tensor's data pointer
    tensor_ptr = flat_tensor.data_ptr()

    # Create a ctypes array from the tensor's memory
    c_type = torch_type_to_ctype(tensor.dtype)
    c_array = (c_type * flat_tensor.numel()).from_address(tensor_ptr)

    # Copy data from embeddings to the tensor
    for i, embedding in enumerate(embeddings):
        for j, value in enumerate(embedding):
            idx = i * len(embedding) + j
            c_array[idx] = value


def ingest_csv(csv_path="uk_manifesto_truncated.csv"):
    eprintln("ingesting csv to parquet...")
    ctx = build_session_context()
    dataframe = ctx.read_csv(csv_path, schema=MANIFESTO_SCHEMA)

    # Ensure output directory exists
    os.makedirs("output/records/", exist_ok=True)
    dataframe.write_parquet("output/records/")
    eprintln(f"Ingested {csv_path} to output/records/")


def load_vectors(ctx: df.SessionContext) -> torch.Tensor:
    embeddings = (
        ctx.table("index_vectors").sort(df.col("vector_id")).select(df.col("embedding"))
    )
    count = embeddings.count()

    vectors = torch.empty((count, EMBEDDING_SIZE), dtype=torch.float32, device="cuda")
    dataframe_to_tensor(embeddings, vectors)

    return vectors


def create_phrasal_chunks(chunk_size=1000):
    eprintln("Creating phrasal chunks from records...")
    ctx = build_session_context()

    # Load records
    records = ctx.read_parquet("output/records/")
    records_pd = records.to_pandas()

    # Process each record to create phrasal chunks
    all_chunks = []
    all_manifesto = []
    all_theme = []
    all_themedetail = []

    for idx, row in records_pd.iterrows():
        text = row["text"]
        if not isinstance(text, str) or not text.strip():
            continue

        # Get phrases and chunk them
        phrases = get_phrases(text)
        chunks = chunk_phrases(phrases, chunk_size)

        for chunk in chunks:
            all_chunks.append(chunk.text)
            all_manifesto.append(row["manifesto"])
            all_theme.append(row["theme"])
            all_themedetail.append(row["themedetail"])

    # Create a new DataFrame with the chunks
    chunks_df = pd.DataFrame(
        {
            "manifesto": all_manifesto,
            "theme": all_theme,
            "themedetail": all_themedetail,
            "text": all_chunks,
            "wordcount": [len(text.split()) for text in all_chunks],
        }
    )

    # Convert pandas DataFrame to PyArrow Table
    chunks_arrow = pa.Table.from_pandas(chunks_df)

    # Save to parquet
    os.makedirs("output/chunks/", exist_ok=True)
    chunks_table = ctx.create_dataframe([chunks_arrow.to_batches()])
    chunks_table.write_parquet("output/chunks/")

    eprintln(f"Created {len(chunks_df)} phrasal chunks and saved to output/chunks/")
    return chunks_table


def vectorize_records(input_dir="output/chunks/"):
    from sentence_transformers import SentenceTransformer

    ctx = build_session_context()
    eprintln("Loading chunks...")
    records = ctx.read_parquet(input_dir)

    eprintln("Initializing model...")
    model = SentenceTransformer(
        "Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True
    )

    eprintln("Processing records and generating embeddings...")
    # Create a unique hash for each record
    records = records.with_column("hash", df.functions.md5(df.col("text")))

    # Get unique texts to avoid duplicate embeddings
    unique_texts = records.select(df.col("hash"), df.col("text")).distinct()

    # Convert to pandas for easier processing
    texts_pd = unique_texts.to_pandas()

    # Generate embeddings in batches
    batch_size = 32
    all_embeddings = []
    all_hashes = []

    for i in range(0, len(texts_pd), batch_size):
        batch = texts_pd.iloc[i : i + batch_size]
        batch_texts = batch["text"].tolist()
        batch_hashes = batch["hash"].tolist()

        # Generate embeddings
        embeddings = model.encode(batch_texts, convert_to_tensor=True)

        # Convert to numpy and store
        embeddings_np = embeddings.cpu().numpy()
        all_embeddings.extend(embeddings_np)
        all_hashes.extend(batch_hashes)

        eprintln(f"Processed {i+len(batch)}/{len(texts_pd)} texts")

    # Create DataFrame with embeddings
    embeddings_df = pd.DataFrame({"hash": all_hashes, "embedding": all_embeddings})

    # Convert to DataFusion table and write to parquet
    embeddings_arrow = pa.Table.from_pandas(embeddings_df)
    embeddings_table = ctx.create_dataframe([embeddings_arrow.to_batches()])
    embeddings_table.write_parquet("output/vectors/")

    eprintln("Vectorization complete!")


def compare_manifesto_categories():
    """
    Compare each manifesto's category with categories in every other manifesto
    using cosine distance implemented with torch.
    
    Returns a DataFrame with pairwise distances between manifesto categories.
    """
    import torch.nn.functional as F
    import pandas as pd
    import numpy as np
    
    ctx = build_session_context()
    eprintln("Loading vectors and chunks...")
    
    # Load vectors
    vectors = ctx.read_parquet("output/vectors/")
    
    # Load chunks with manifesto and theme information
    chunks = ctx.read_parquet("output/chunks/")

    # Join vectors with chunks to get manifesto and theme for each vector
    chunks = chunks.with_column("hash", df.functions.md5(df.col("text")))
    vectors = vectors.with_column_renamed("hash", "vector_hash")
    joined = chunks.join(vectors, left_on=["hash"], right_on=["vector_hash"])
    
    # Convert to pandas for easier processing
    joined_pd = joined.select(
        df.col("manifesto"), 
        df.col("theme"), 
        df.col("themedetail"),
        df.col("embedding")
    ).to_pandas()
    
    # Group by manifesto and theme to get average embeddings
    eprintln("Computing average embeddings for each manifesto-theme combination...")
    grouped = joined_pd.groupby(["manifesto", "theme"])
    
    # Calculate average embeddings for each manifesto-theme combination
    avg_embeddings = {}
    for (manifesto, theme), group in grouped:
        embeddings = np.stack(group["embedding"].values)
        avg_embedding = torch.tensor(embeddings.mean(axis=0), dtype=torch.float32)
        # Normalize the embedding
        avg_embedding = F.normalize(avg_embedding, p=2, dim=0)
        avg_embeddings[(manifesto, theme)] = avg_embedding

    # 1: calculate distance between manifestos by theme
    # Calculate pairwise cosine distances
    eprintln("Calculating pairwise cosine distances by theme...")
    results = []
    keys = list(avg_embeddings.keys())

    for i, (manifesto1, theme1) in enumerate(keys):
        for j, (manifesto2, theme2) in enumerate(keys):
            if theme1 != theme2:
                continue

            emb1 = avg_embeddings[(manifesto1, theme1)]
            emb2 = avg_embeddings[(manifesto2, theme2)]

            # Calculate cosine distance (1 - cosine similarity)
            # Since embeddings are normalized, we can use dot product for cosine similarity
            cosine_sim = torch.dot(emb1, emb2).item()
            cosine_dist = 1.0 - cosine_sim

            results.append({
                "manifesto1": manifesto1,
                "theme1": theme1,
                "manifesto2": manifesto2,
                "theme2": theme2,
                "cosine_distance": cosine_dist
            })

    # TODO
    # 2: Calculate distance of every manifesto chunk to every other
    # 2.a: Create a heat map (using seaborn) and save it to a png
    # TODO
    # 3: Calculate overall average distances between one manifsto and another

    # Create DataFrame with results
    results_df = pd.DataFrame(results)

    # Save results to CSV
    os.makedirs("output/comparisons/", exist_ok=True)
    results_path = "output/comparisons/manifesto_theme_distances.csv"
    results_df.to_csv(results_path, index=False)

    eprintln(f"Comparison complete! Results saved to {results_path}")
    return results_df

def process_csv_pipeline(csv_path="uk_manifesto_truncated.csv", chunk_size=1000):
    """
    Process a CSV file through the entire pipeline:
    1. Ingest CSV to parquet
    2. Create phrasal chunks
    3. Vectorize the chunks
    """
    eprintln(f"Starting pipeline for {csv_path}")

    # Step 1: Ingest CSV
    ingest_csv(csv_path)

    # Step 2: Create phrasal chunks
    create_phrasal_chunks(chunk_size)

    # Step 3: Vectorize chunks
    vectorize_records(input_dir="output/chunks/")

    eprintln("Pipeline completed successfully!")
