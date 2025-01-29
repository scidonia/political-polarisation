import pandas as pd
import pyarrow as pa
from vectorlink_py import template as tpl, dedup, embed, records
from vectorlink_gpu.datafusion import dataframe_to_tensor, tensor_to_arrow
import datafusion as df
import torch
from .context import *

MANIFESTOS_CSV_PATH = "uk_manifesto_chunks_by_11_themes.csv"


def ingest_csv():
    print("ingesting csv to parquet...")
    ctx = build_session_context()
    dataframe = ctx.read_csv(MANIFESTOS_CSV_PATH, schema=MANIFESTO_SCHEMA)

    dataframe.write_parquet("output/records/")


def build_text_records():
    ctx = build_session_context()

    result = (
        ctx.table("records")
        .with_column_renamed("text", "templated")
        .select(
            (df.functions.row_number() - 1).alias("id"),
            df.col("templated"),
            (df.functions.md5(df.col("templated"))).alias("hash"),
        )
    )

    result.write_parquet("output/text")


def vectorize_records():
    ctx = build_session_context()

    print("vectorizing...")
    configuration = {
        "provider": "OpenAI",
        "max_batch_size": 200 * 2**20,
        "dimension": EMBEDDING_SIZE,
        "model": MODEL,
    }
    embed.vectorize(ctx, "output/text/", "output/vectors/", configuration=configuration)


def load_vectors(ctx: df.SessionContext) -> torch.Tensor:
    embeddings = (
        ctx.table("index_vectors").sort(df.col("vector_id")).select(df.col("embedding"))
    )
    count = embeddings.count()

    vectors = torch.empty((count, EMBEDDING_SIZE), dtype=torch.float32, device="cuda")
    dataframe_to_tensor(embeddings, vectors)

    return vectors


def process_records():
    ingest_csv()
    build_text_records()
    vectorize_records()
