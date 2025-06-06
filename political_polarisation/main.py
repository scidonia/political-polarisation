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
        df.col("manifesto"), df.col("theme"), df.col("themedetail"), df.col("embedding")
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
            if (
                theme1 != theme2 or i >= j
            ):  # Skip different themes and only compute upper triangle
                continue

            emb1 = avg_embeddings[(manifesto1, theme1)]
            emb2 = avg_embeddings[(manifesto2, theme2)]

            # Calculate cosine distance (1 - cosine similarity)
            # Since embeddings are normalized, we can use dot product for cosine similarity
            cosine_sim = torch.dot(emb1, emb2).item()
            cosine_dist = 1.0 - cosine_sim

            results.append(
                {
                    "manifesto1": manifesto1,
                    "theme1": theme1,
                    "manifesto2": manifesto2,
                    "theme2": theme2,
                    "cosine_distance": cosine_dist,
                }
            )

    # 2: Calculate distance of every manifesto chunk to every other
    eprintln("Calculating pairwise distances between all manifesto chunks...")

    # Get unique manifestos
    unique_manifestos = joined_pd["manifesto"].unique()

    # Create a dictionary to store all embeddings by manifesto
    manifesto_embeddings = {}
    for manifesto in unique_manifestos:
        manifesto_data = joined_pd[joined_pd["manifesto"] == manifesto]
        embeddings = np.stack(manifesto_data["embedding"].values)
        manifesto_embeddings[manifesto] = torch.tensor(embeddings, dtype=torch.float32)

    # Calculate average distance between all chunks of each manifesto pair
    chunk_distances = []
    for i, manifesto1 in enumerate(unique_manifestos):
        for j, manifesto2 in enumerate(unique_manifestos):
            if i >= j:  # Only compute upper triangle
                continue

            embs1 = manifesto_embeddings[manifesto1]
            embs2 = manifesto_embeddings[manifesto2]

            # Normalize all embeddings
            embs1 = F.normalize(embs1, p=2, dim=1)
            embs2 = F.normalize(embs2, p=2, dim=1)

            # Calculate cosine similarity matrix between all chunks
            # (n_chunks1, embedding_dim) @ (embedding_dim, n_chunks2) -> (n_chunks1, n_chunks2)
            similarity_matrix = torch.mm(embs1, embs2.t())

            # Convert to distances
            distance_matrix = 1.0 - similarity_matrix

            # Calculate average distance
            avg_distance = distance_matrix.mean().item()

            chunk_distances.append(
                {
                    "manifesto1": manifesto1,
                    "manifesto2": manifesto2,
                    "avg_chunk_distance": avg_distance,
                    "min_chunk_distance": distance_matrix.min().item(),
                    "max_chunk_distance": distance_matrix.max().item(),
                }
            )

    # Create DataFrame with chunk distance results
    chunk_distances_df = pd.DataFrame(chunk_distances)

    # Save results to CSV
    chunk_distances_path = "output/comparisons/manifesto_chunk_distances.csv"
    chunk_distances_df.to_csv(chunk_distances_path, index=False)
    eprintln(f"Chunk distances saved to {chunk_distances_path}")

    # 2.a: Create a heat map using seaborn and save it to a png
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Create a pivot table for the heatmap
        heatmap_data = pd.pivot_table(
            chunk_distances_df,
            values="avg_chunk_distance",
            index="manifesto1",
            columns="manifesto2",
        )

        # Create the heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", linewidths=0.5, fmt=".3f")
        plt.title("Average Cosine Distance Between Manifesto Chunks")

        # Save the heatmap
        os.makedirs("output/visualizations/", exist_ok=True)
        heatmap_path = "output/visualizations/manifesto_distance_heatmap.png"
        plt.savefig(heatmap_path, bbox_inches="tight", dpi=300)
        plt.close()
        eprintln(f"Heatmap saved to {heatmap_path}")
    except ImportError:
        eprintln(
            "Warning: seaborn or matplotlib not installed. Skipping heatmap generation."
        )

    # Create DataFrame with results from theme comparisons
    results_df = pd.DataFrame(results)

    # Save results to CSV
    os.makedirs("output/comparisons/", exist_ok=True)
    results_path = "output/comparisons/manifesto_theme_distances.csv"
    results_df.to_csv(results_path, index=False)

    # Create a heatmap for theme distances
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Create a multi-index for the heatmap
        # This will create a single heatmap with manifesto pairs on one axis and themes on the other

        # Create a new column with manifesto pair names for better readability
        results_df["manifesto_pair"] = results_df.apply(
            lambda row: f"{row['manifesto1']} vs {row['manifesto2']}", axis=1
        )

        # Add the reverse pairs to make the heatmap complete
        # This is just for visualization - we only computed half the pairs to avoid redundancy
        reverse_results = []
        for _, row in results_df.iterrows():
            if row["manifesto1"] != row["manifesto2"]:  # Skip self-comparisons
                reverse_results.append(
                    {
                        "manifesto1": row["manifesto2"],
                        "theme1": row["theme2"],
                        "manifesto2": row["manifesto1"],
                        "theme2": row["theme1"],
                        "cosine_distance": row["cosine_distance"],
                        "manifesto_pair": f"{row['manifesto2']} vs {row['manifesto1']}",
                    }
                )

        # Add reverse pairs to the dataframe for complete visualization
        vis_df = pd.concat(
            [results_df, pd.DataFrame(reverse_results)], ignore_index=True
        )

        # Create a pivot table with themes as columns and manifesto pairs as rows
        theme_heatmap_data = pd.pivot_table(
            vis_df, values="cosine_distance", index="manifesto_pair", columns="theme1"
        )

        # Create the heatmap
        plt.figure(figsize=(20, 16))
        sns.heatmap(
            theme_heatmap_data, annot=True, cmap="YlGnBu", linewidths=0.5, fmt=".3f"
        )
        plt.title("Cosine Distance Between Manifestos by Theme")
        plt.xlabel("Theme")
        plt.ylabel("Manifesto Pair")

        # Adjust layout to make room for labels
        plt.tight_layout()

        # Save the heatmap
        os.makedirs("output/visualizations/", exist_ok=True)
        theme_heatmap_path = "output/visualizations/manifesto_theme_heatmap.png"
        plt.savefig(theme_heatmap_path, bbox_inches="tight", dpi=300)
        plt.close()
        eprintln(f"Theme heatmap saved to {theme_heatmap_path}")
    except Exception as e:
        eprintln(f"Warning: Error creating theme heatmaps: {e}")

    # 3: Calculate overall average distances between one manifesto and another
    eprintln("Calculating overall average distances between manifestos...")

    # Group by manifesto pair to get average distance across all themes
    manifesto_distances = []

    # Get unique manifesto pairs from the theme distances
    manifesto_pairs = set()
    for _, row in results_df.iterrows():
        pair = (row["manifesto1"], row["manifesto2"])
        manifesto_pairs.add(pair)

    # Calculate average distance for each manifesto pair
    for manifesto1, manifesto2 in manifesto_pairs:
        # Get all theme distances for this manifesto pair
        pair_data = results_df[
            (results_df["manifesto1"] == manifesto1)
            & (results_df["manifesto2"] == manifesto2)
        ]

        # Calculate average distance across all themes
        avg_distance = pair_data["cosine_distance"].mean()

        manifesto_distances.append(
            {
                "manifesto1": manifesto1,
                "manifesto2": manifesto2,
                "avg_theme_distance": avg_distance,
                "min_theme_distance": pair_data["cosine_distance"].min(),
                "max_theme_distance": pair_data["cosine_distance"].max(),
                "theme_count": len(pair_data),
            }
        )

    # Create DataFrame with overall manifesto distance results
    manifesto_distances_df = pd.DataFrame(manifesto_distances)

    # Save results to CSV
    manifesto_distances_path = "output/comparisons/overall_manifesto_distances.csv"
    manifesto_distances_df.to_csv(manifesto_distances_path, index=False)
    eprintln(f"Overall manifesto distances saved to {manifesto_distances_path}")

    # Create a heatmap for overall manifesto distances
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Create a pivot table for the heatmap
        # Add the reverse pairs for visualization
        reverse_manifesto_distances = []
        for _, row in manifesto_distances_df.iterrows():
            reverse_manifesto_distances.append(
                {
                    "manifesto1": row["manifesto2"],
                    "manifesto2": row["manifesto1"],
                    "avg_theme_distance": row["avg_theme_distance"],
                    "min_theme_distance": row["min_theme_distance"],
                    "max_theme_distance": row["max_theme_distance"],
                    "theme_count": row["theme_count"],
                }
            )

        # Combine original and reverse pairs for visualization
        vis_manifesto_df = pd.concat(
            [manifesto_distances_df, pd.DataFrame(reverse_manifesto_distances)],
            ignore_index=True,
        )

        # Create the pivot table with the complete data
        overall_heatmap_data = pd.pivot_table(
            vis_manifesto_df,
            values="avg_theme_distance",
            index="manifesto1",
            columns="manifesto2",
        )

        # Create the heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            overall_heatmap_data, annot=True, cmap="YlGnBu", linewidths=0.5, fmt=".3f"
        )
        plt.title("Average Cosine Distance Between Manifestos (Across All Themes)")

        # Save the heatmap
        overall_heatmap_path = (
            "output/visualizations/overall_manifesto_distance_heatmap.png"
        )
        plt.savefig(overall_heatmap_path, bbox_inches="tight", dpi=300)
        plt.close()
        eprintln(f"Overall heatmap saved to {overall_heatmap_path}")
    except ImportError:
        eprintln(
            "Warning: seaborn or matplotlib not installed. Skipping overall heatmap generation."
        )

    eprintln(f"Comparison complete! Results saved to {results_path}")
    return results_df


def calculate_string_distance(string1, string2, as_query=False, model_name=None):
    """
    Calculate the cosine distance between two strings using the specified model.

    Args:
        string1: First string to compare
        string2: Second string to compare
        as_query: If True, treats string1 as a query
        model_name: Name of the model to use. If None, uses the default model from context.py

    Returns:
        float: Cosine distance between the two strings
    """
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer

    # Use the specified model or default to the one in context.py
    model_to_use = model_name if model_name else MODEL
    eprintln(f"Initializing model: {model_to_use}")

    # Initialize the model
    model = SentenceTransformer(model_to_use, trust_remote_code=True)

    eprintln("Encoding strings...")

    # Handle query formatting based on the model
    if as_query:
        eprintln("Treating first string as a query...")
        if "Mistral" in model_to_use:
            # Mistral model requires a prompt for queries
            task = (
                "Given a question, retrieve relevant passages that answer the question"
            )
            prompt = f"Instruct: {task}\nQuery: "
            embedding1 = model.encode(string1, prompt=prompt, convert_to_tensor=True)
        elif "Qwen" in model_to_use:
            # Qwen model uses prompt_name parameter
            embedding1 = model.encode(
                string1, convert_to_tensor=True, prompt_name="query"
            )
        else:
            # Default behavior for other models
            embedding1 = model.encode(string1, convert_to_tensor=True)
    else:
        embedding1 = model.encode(string1, convert_to_tensor=True)

    # Encode the second string normally
    embedding2 = model.encode(string2, convert_to_tensor=True)

    # Normalize embeddings
    embedding1 = F.normalize(embedding1, p=2, dim=0)
    embedding2 = F.normalize(embedding2, p=2, dim=0)

    # Calculate cosine similarity
    cosine_sim = torch.dot(embedding1, embedding2).item()

    # Convert to distance
    cosine_dist = 1.0 - cosine_sim

    return cosine_dist


def analyze_story_characters(story_path, characters_csv, model_name=None, debug=False):
    """
    Analyze a story text file to identify character references and match them with character descriptions.

    Args:
        story_path: Path to the story text file
        characters_csv: Path to the CSV file containing character descriptions
        model_name: Name of the model to use for embeddings
        debug: If True, prints detailed debug information for each reference
    """
    import re
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer

    # Use the specified model or default to Mistral for character analysis
    model_to_use = model_name if model_name else MODELS.get("mistral")
    eprintln(f"Initializing model: {model_to_use}")

    # Initialize the model
    model = SentenceTransformer(model_to_use, trust_remote_code=True)

    # Load the story text
    eprintln(f"Loading story from {story_path}")
    with open(story_path, "r") as f:
        story_text = f.read()

    # Load character descriptions
    eprintln(f"Loading character descriptions from {characters_csv}")
    characters_df = pd.read_csv(characters_csv)

    # Create phrasal chunks from the story
    eprintln("Creating phrasal chunks from story text")
    phrases = get_phrases(story_text)
    chunks = chunk_phrases(phrases, 500)  # Adjust chunk size as needed

    # Extract character references from brackets in the text
    eprintln("Extracting character references")
    all_references = []
    for chunk in chunks:
        # Find all bracketed references in the chunk
        references = re.findall(r"\[(.*?)\]", chunk.text)
        references = list(set(references))
        if references:
            all_references.extend([(ref, chunk) for ref in references])

    eprintln(f"Found {len(all_references)} character references in the story")

    # Create embeddings for each character description
    eprintln("Creating embeddings for character descriptions")
    character_embeddings = {}
    for _, row in characters_df.iterrows():
        name = row["name"]
        description = row["description"]

        # Create embedding for character description
        task = "Given a character with a detailed description, determine if they match a reference in a story passage."
        prompt = f"Instruct: {task}\nThe individual to identify {name}."
        text = f"Name: {name}\nDescription: {description}"
        embedding = model.encode(description, prompt=prompt, convert_to_tensor=True)
        embedding = F.normalize(embedding, p=2, dim=0)
        character_embeddings[name] = embedding

    # Process each reference and chunk pair
    eprintln("Analyzing character references in chunks")
    results = []

    # Track unique reference-chunk pairs to avoid redundant processing
    processed_pairs = {}

    for reference, chunk in all_references:
        # Create a key for this reference-chunk pair
        pair_key = (reference, chunk.text)

        # Skip if we've already processed this exact reference-chunk pair
        if pair_key in processed_pairs:
            # Reuse the previous result
            result = processed_pairs[pair_key].copy()
        else:
            # Create embedding for the chunk with context about the reference
            task = "Identify which character this reference refers to in the following passage"
            prompt = f"Instruct: {task}\nReference '[{reference}]' appears in this passage. Determine which character this reference is pointing to based on context, actions, and attributes described."
            chunk_embedding = model.encode(
                chunk.text, prompt=prompt, convert_to_tensor=True
            )
            chunk_embedding = F.normalize(chunk_embedding, p=2, dim=0)

            # Compare with each character embedding
            similarities = {}
            for char_name, char_embedding in character_embeddings.items():
                similarity = torch.dot(chunk_embedding, char_embedding).item()
                similarities[char_name] = similarity

            # Find the best match
            best_match = max(similarities.items(), key=lambda x: x[1])

            result = {
                "chunk": chunk.text,
                "reference": reference,
                "best_match": best_match[0],
                "similarity": best_match[1],
                "all_similarities": similarities,
            }

            # Store this result for future identical pairs
            processed_pairs[pair_key] = result.copy()

        if debug:
            print("\n" + "=" * 80)
            print(f"PASSAGE: {chunk.text}")
            print(f"REFERENCE: [{reference}]")
            print(f"BEST MATCH: {best_match[0]} (similarity: {best_match[1]:.4f})")
            print("ALL SIMILARITIES:")
            for char, sim in sorted(
                similarities.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  - {char}: {sim:.4f}")
            print("=" * 80)

        results.append(result)

    # Create output directory
    os.makedirs("output/story_analysis/", exist_ok=True)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_path = "output/story_analysis/character_references.csv"
    results_df.to_csv(results_path, index=False)

    # Print summary
    eprintln(f"Analysis complete! Results saved to {results_path}")

    # Print a summary of the results
    print("\nCharacter Reference Analysis Summary:")
    print("=====================================")

    # Group by reference and count occurrences
    reference_counts = {}
    for result in results:
        ref = result["reference"]
        if ref not in reference_counts:
            reference_counts[ref] = {"count": 0, "matches": {}}

        reference_counts[ref]["count"] += 1

        best_match = result["best_match"]
        if best_match not in reference_counts[ref]["matches"]:
            reference_counts[ref]["matches"][best_match] = 0
        reference_counts[ref]["matches"][best_match] += 1

    # Print the summary
    for ref, data in reference_counts.items():
        print(f"\nReference: {ref}")
        print(f"Occurrences: {data['count']}")
        print("Matched to:")

        # Sort matches by count
        sorted_matches = sorted(
            data["matches"].items(), key=lambda x: x[1], reverse=True
        )
        for match, count in sorted_matches:
            print(f"  - {match}: {count} times ({count/data['count']*100:.1f}%)")

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
