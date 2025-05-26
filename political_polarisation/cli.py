import argparse
import sys
from political_polarisation.main import (
    ingest_csv,
    create_phrasal_chunks,
    vectorize_records,
    process_csv_pipeline,
    compare_manifesto_categories,
)


def cli_ingest_csv():
    parser = argparse.ArgumentParser(description="Ingest CSV data to parquet")
    parser.add_argument(
        "--csv-path", default="uk_manifesto_truncated.csv", help="Path to the CSV file"
    )
    args = parser.parse_args()
    ingest_csv(args.csv_path)


def cli_create_chunks():
    parser = argparse.ArgumentParser(description="Create phrasal chunks from records")
    parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Size of each chunk"
    )
    args = parser.parse_args()
    create_phrasal_chunks(args.chunk_size)


def cli_vectorize():
    parser = argparse.ArgumentParser(description="Generate vector embeddings")
    parser.add_argument(
        "--input-dir", default="output/chunks/", help="Directory containing chunks"
    )
    args = parser.parse_args()
    vectorize_records(args.input_dir)


def cli_process_csv():
    parser = argparse.ArgumentParser(description="Run the complete pipeline")
    parser.add_argument(
        "--csv-path", default="uk_manifesto_truncated.csv", help="Path to the CSV file"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Size of each chunk"
    )
    args = parser.parse_args()
    process_csv_pipeline(args.csv_path, args.chunk_size)


def cli_compare_categories():
    parser = argparse.ArgumentParser(
        description="Compare manifesto categories using cosine distance"
    )
    args = parser.parse_args()
    compare_manifesto_categories()
