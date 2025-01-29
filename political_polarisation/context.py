import pyarrow as pa
import datafusion as df

MANIFESTO_SCHEMA = pa.schema(
    [
        pa.field("manifesto", pa.string(), nullable=False),
        pa.field("theme", pa.string(), nullable=False),
        pa.field("themedetail", pa.string(), nullable=False),
        pa.field("wordcount", pa.int64(), nullable=False),
        pa.field("text", pa.string(), nullable=True),
    ]
)

TEXT_SCHEMA = pa.schema(
    [
        pa.field("hash", pa.string_view(), nullable=False),
        pa.field("id", pa.int64(), nullable=False),
        pa.field("templated", pa.string(), nullable=False),
    ]
)

MODEL = "text-embedding-3-large"
EMBEDDING_SIZE = 3072  # 1536

VECTORS_SCHEMA = pa.schema(
    [
        pa.field("hash", pa.string_view(), nullable=False),
        pa.field("embedding", pa.list_(pa.float32(), EMBEDDING_SIZE), nullable=False),
    ]
)


def build_session_context(location="output/") -> df.SessionContext:
    ctx = df.SessionContext()
    ctx.register_parquet("records", f"{location}records/", schema=MANIFESTO_SCHEMA)

    ctx.register_parquet("text", f"{location}text/", schema=TEXT_SCHEMA)
    return ctx
