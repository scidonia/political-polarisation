import pandas as pd
import pyarrow as pa
import .context as context
from vectorlink_py import template as tpl, dedup, embed, records
from vectorlink_gpu.datafusion import dataframe_to_tensor, tensor_to_arrow

MANIFESTOS_CSV_PATH = uk_manifesto_chunks_by_11_themes.csv

def ingest_csv():
    eprintln("ingesting csv to parquet...")
    ctx = context.build_session_context()
    dataframe = ctx.read_csv(
        MANIFESTOS_CSV_PATH, schema=context.MANIFESTO_SCHEMA
    )

    dataframe.write_parquet("output/records/")

def load_vectors(ctx: df.SessionContext) -> torch.Tensor:
    embeddings = (
        ctx.table("index_vectors").sort(df.col("vector_id")).select(df.col("embedding"))
    )
    count = embeddings.count()

    vectors = torch.empty(
        (count, context.EMBEDDING_SIZE), dtype=torch.float32, device="cuda"
    )
    dataframe_to_tensor(embeddings, vectors)

    return vectors

def vectorize_records():
    ctx = context.build_session_context()

    eprintln("vectorizing...")
    configuration = {
        "provider": "OpenAI",
        "max_batch_size": 200 * 2**20,
        "dimension": context.EMBEDDING_SIZE,
        "model": context.MODEL,
    }
    embed.vectorize(
        ctx, "output/dedup/", "output/vectors/", configuration=configuration
    )
