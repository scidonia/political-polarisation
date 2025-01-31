import pandas as pd
import pyarrow as pa
from vectorlink_py import template as tpl, dedup, embed, records
from vectorlink_gpu.datafusion import dataframe_to_tensor, tensor_to_arrow
import datafusion as df
import torch
import tiktoken
import hashlib
import os
from .context import *
import re
from typing import List

from openai import OpenAI
import openai as oa

from tiktoken.core import Encoding

MANIFESTOS_CSV_PATH = "uk_manifesto_chunks_by_11_themes.csv"


def is_sublist(sublist, mainlist):
    n = len(sublist)
    m = len(mainlist)
    # Handle empty sublist (considered a sublist by definition)
    if n == 0:
        return True
    if m < n:
        return False
    for i in range(m - n + 1):
        if mainlist[i : i + n] == sublist:
            return True
    return False


def ask_oracle_for_chunk(enc, tokens):
    text = enc.decode(tokens)
    subject = "Historic People who are creators of works"
    # api_key = os.environ["DEEPSEEK_API_KEY"]
    # client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    client = OpenAI()

    prompt = "You are an AI designed to find boundaries in a text which will yield sensible semantics."
    question = f"""Can you give me a reproduction of the text below (following the marker "BEGINS") cutting off only the beginning or end where it breaks a sentence, phrase, quote, title or utterance, including no other words but those in the original text.

BEGINS

{text}
"""
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )
    print(f">> {question}")
    content = completion.choices[0].message.content
    print(f"\n<< {content}")

    new_tokens = enc.encode(content)
    if is_sublist(new_tokens, tokens):
        return new_tokens
    else:
        raise Exception("Could not find a semantically meaningful sub-window")


# Step 1: Define Tokenizer and Token Window Parameters
def chunk_tokens(tokens, window_size, overlap):
    """
    Creates overlapping chunks of tokens with a fixed size.
    """
    chunks = []
    for i in range(0, len(tokens), window_size - overlap):
        chunk = tokens[i : i + window_size]  # Slide the window
        if len(chunk) == window_size:  # Ensure only full chunks are included
            chunks.append(chunk)
    return chunks


def mechanical_clean_chunk(enc: Encoding, chunk: List[int]) -> List[int]:
    """
    Trims a chunk of text to ensure it starts and ends at valid sentence, quote, or title boundaries.
    """
    text = enc.decode(chunk)
    # Define regex patterns
    start_pattern = (
        r"^[^A-Z\"']*[A-Z\"']"  # Match non-sentence-starting junk at the beginning
    )
    end_pattern = r"[.!?\"']\s*[^.!?\"']*$"  # Match non-sentence-ending junk at the end

    # Find the valid start
    start_match = re.search(start_pattern, text)
    if start_match:
        start_index = (
            start_match.end() - 1
        )  # `end()` gives the index just after the match
    else:
        start_index = 0  # Start at the very beginning if no match

    # Find the valid end
    end_match = re.search(end_pattern, text)
    if end_match:
        end_index = end_match.start() + 1  # `start()` gives the index of the match
    else:
        end_index = len(text)  # End at the very last character if no match

    # Return the trimmed text
    trimmed = text[start_index:end_index].strip()
    return enc.encode(trimmed)


# This needs to be a column processor
# Step 2: Processing Function to Tokenize Text and Create Chunks
def oracular_text_chunking(
    records: pa.Array,
    *,
    enc: Encoding = tiktoken.get_encoding("o200k_base"),
    window_size: int = 5000,
    overlap: int = 500,
) -> pa.Array:
    records_with_chunks = []
    for record_text in records:
        tokenized = enc.encode(str(record_text))
        chunks = chunk_tokens(tokenized, window_size, overlap)
        chunk_array_list = []
        # Add each chunk with its record id
        for chunk_id, chunk in enumerate(chunks):
            try:
                chunk = ask_oracle_for_chunk(enc, chunk)
            except Exception as e:
                chunk = mechanical_clean_chunk(enc, chunk)
            print("After the oracle")
            print(f"with chunk:\n\n\n {chunk}")
            text = enc.decode(chunk)
            print(text)
            chunk_array_list.append(text)
        records_with_chunks.append(pa.array(chunk_array_list))
    return pa.array(records_with_chunks)


def record_chunker_fn(records: pa.Array) -> pa.Array:
    return oracular_text_chunking(records)


record_chunker = df.udf(
    record_chunker_fn,
    [pa.string_view()],
    pa.list_(pa.string_view()),
    "stable",
)


def ingest_csv():
    print("ingesting csv to parquet...")
    ctx = build_session_context()
    dataframe = ctx.read_csv(MANIFESTOS_CSV_PATH, schema=MANIFESTO_SCHEMA)

    dataframe.write_parquet("output/records/")


def build_text_records():
    ctx = build_session_context()

    result = (
        ctx.table("records")
        .select(
            (df.functions.row_number() - 1).alias("id"),
            df.col("text"),
            (df.functions.md5(df.col("text"))).alias("hash"),
            record_chunker(df.col("text")).alias("chunk"),
        )
        .unnest_column("chunk")
        .with_column("chunk_id", (df.functions.row_number() - 1)),
    )
    print(result)
    result.write_parquet("output/text")


def vectorize_records():
    ctx = build_session_context()

    print("vectorizing...")
    embed.vectorize(
        ctx,
        "output/text/",
        "output/vectors/",
        dimension=EMBEDDING_SIZE,
        model=model,
        batch_size=1,
    )


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
    average_fields()
    build_index_map()
    index_field()
