import argparse
import sys
from political_polarisation.main import (
    ingest_csv,
    create_phrasal_chunks,
    vectorize_records,
    process_csv_pipeline,
    compare_manifesto_categories,
    calculate_string_distance,
    analyze_story_characters,
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


def cli_string_distance():
    parser = argparse.ArgumentParser(
        description="Calculate cosine distance between two strings"
    )
    parser.add_argument(
        "string1", 
        help="First string to compare"
    )
    parser.add_argument(
        "string2", 
        help="Second string to compare"
    )
    parser.add_argument(
        "--as-query", 
        action="store_true",
        help="Treat the first string as a query"
    )
    parser.add_argument(
        "--model",
        choices=["qwen", "mistral"],
        default=None,
        help="Model to use for embedding (qwen or mistral)"
    )
    args = parser.parse_args()
    
    # Get the model name from the MODELS dictionary in context.py
    from political_polarisation.context import MODELS
    model_name = MODELS.get(args.model) if args.model else None
    
    distance = calculate_string_distance(
        args.string1, 
        args.string2, 
        as_query=args.as_query,
        model_name=model_name
    )
    print(f"Cosine distance: {distance:.6f}")
    print(f"Cosine similarity: {1.0 - distance:.6f}")


def cli_analyze_story():
    parser = argparse.ArgumentParser(
        description="Analyze story characters and their references"
    )
    parser.add_argument(
        "--story", 
        default="test_story.txt",
        help="Path to the story text file"
    )
    parser.add_argument(
        "--characters", 
        default="story_characters.csv",
        help="Path to the CSV file with character descriptions"
    )
    parser.add_argument(
        "--model",
        choices=["qwen", "mistral"],
        default="mistral",
        help="Model to use for embedding (qwen or mistral)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed debug information for each reference"
    )
    args = parser.parse_args()
    
    # Get the model name from the MODELS dictionary in context.py
    from political_polarisation.context import MODELS
    model_name = MODELS.get(args.model)
    
    analyze_story_characters(
        args.story,
        args.characters,
        model_name=model_name,
        debug=args.debug
    )
