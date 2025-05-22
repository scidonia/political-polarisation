# Political Polarisation Analysis

This project analyzes political polarisation through LLM text embeddings of political manifestos. It provides tools to process, chunk, and vectorize political text data for comparative analysis.

## Overview

The system processes political manifesto text data through a pipeline:
1. Ingest CSV data containing manifesto text
2. Create phrasal chunks from the text for better analysis
3. Generate vector embeddings using the Alibaba-NLP/gte-Qwen2-7B-instruct model

## Prerequisites

- Python 3.12+
- Required packages (installed via pip or uv)
- Access to the Alibaba-NLP/gte-Qwen2-7B-instruct model

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/political-polarisation.git
cd political-polarisation
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

## Data Format

The input CSV should follow this structure (similar to uk_manifesto_truncated.csv):

| manifesto | theme | themedetail | wordcount | text |
|-----------|-------|-------------|-----------|------|
| c_2019    | communities | on towns, cities, rural, housing, transport | 3509 | We Will Unleash Britain's Potential... |
| c_2024    | communities | on towns, cities, rural, housing, transport | 2530 | Our plan to build more houses... |

## Usage

### Complete Pipeline

To run the entire pipeline, you need to specify a CSV file:

```bash
mkdir -p output/records output/chunks output/vectors
process-csv --csv-path path/to/your/manifesto.csv
```

You can also specify the chunk size:

```bash
process-csv --csv-path path/to/your/manifesto.csv --chunk-size 1500
```

For example, to process the included sample file:

```bash
process-csv --csv-path uk_manifesto_truncated.csv
```

### Individual Steps

You can also run each step of the pipeline individually:

1. Ingest CSV data:
```bash
ingest-csv --csv-path path/to/your/manifesto.csv
```

2. Create phrasal chunks:
```bash
create-chunks [--chunk-size 1000]
```

3. Generate vector embeddings:
```bash
vectorize [--input-dir output/chunks/]
```

## Output

The pipeline generates three main outputs:

1. `output/records/` - Raw parquet files of the ingested CSV data
2. `output/chunks/` - Phrasal chunks created from the text data
3. `output/vectors/` - Vector embeddings of the chunks

## Analysis

After running the pipeline, you can use the generated vector embeddings for various analyses:

- Compare similarity between different manifestos
- Track changes in political language over time
- Identify polarization on specific themes

## License

Apache 2.0
