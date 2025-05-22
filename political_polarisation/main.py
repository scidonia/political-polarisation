import pandas as pd
import pyarrow as pa
from context import *


def ingest_csv():
    eprintln("ingesting csv to parquet...")
    ctx = context.build_session_context()
    dataframe = ctx.read_csv(MANIFESTOS_CSV_PATH, schema=context.MANIFESTO_SCHEMA)

    dataframe.write_parquet("output/records/")


def load_vectors(ctx: df.SessionContext) -> torch.Tensor:
    embeddings = (
        ctx.table("index_vectors").sort(df.col("vector_id")).select(df.col("embedding"))
    )
    count = embeddings.count()

    vectors = torch.empty((count, EMBEDDING_SIZE), dtype=torch.float32, device="cuda")
    dataframe_to_tensor(embeddings, vectors)

    return vectors


def vectorize_records():
    ctx = context.build_session_context()
    pass
