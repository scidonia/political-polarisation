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

MODEL = "Alibaba-NLP/gte-Qwen2-7B-instruct"
EMBEDDING_SIZE = 4096

VECTORS_SCHEMA = pa.schema(
    [
        pa.field("hash", pa.string_view(), nullable=False),
        pa.field(
            "embedding",
            pa.list_(pa.float32(), EMBEDDING_SIZE),
            nullable=False,
        ),
    ]
)


def build_session_context(location="output/") -> df.SessionContext:
    ctx = df.SessionContext()
    ctx.register_parquet(
        "manifestos", f"{location}manifestos/", schema=MANIFESTO_SCHEMA
    )
    ctx.register_parquet(
        "vectors",
        f"{location}vectors/",
        schema=VECTORS_SCHEMA,
    )
    return ctx
