# CMS Data Embeddings Generator

This script processes CMS document files, extracts text content, generates embeddings, and prepares them for loading into an S3 vector store.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:

```bash
python transform_files_to_embeddings.py
```

This will process files from `raw_data/CMS` and save embeddings to `processed_data/embeddings`.

### Command Line Options

- `--input-dir`: Input directory containing CMS files (default: "raw_data/CMS")
- `--output-dir`: Output directory for embeddings (default: "processed_data/embeddings")
- `--chunk-size`: Chunk size for text splitting (default: 1000)
- `--chunk-overlap`: Chunk overlap for text splitting (default: 200)
- `--model-name`: Embedding model name (default: "BAAI/bge-small-en-v1.5")
- `--upload-to-s3`: Flag to upload results to S3
- `--s3-bucket`: S3 bucket name (required if --upload-to-s3 is used)
- `--s3-prefix`: S3 prefix (default: "cms_embeddings")

### Example with S3 Upload

```bash
python transform_files_to_embeddings.py --upload-to-s3 --s3-bucket my-vector-store-bucket --s3-prefix cms_data
```

## Output Format

The script generates parquet files containing:
- Document text chunks
- Metadata from the original files
- Vector embeddings

Each parquet file contains a batch of document chunks with their embeddings, ready to be loaded into a vector database.

## Data Structure

The script processes the following structure:
- JSON metadata files (containing document information)
- HTML content files (containing the actual document text)
- Comments and other related documents

The script extracts text from HTML, chunks it into smaller segments, and generates embeddings for each chunk while preserving the original metadata.

## CMS Agent

The project includes a CMS Agent that can answer questions about CMS dockets and regulations using the vector store.

### Command Line Agent

Run the command-line agent:

```bash
python cms_agent.py
```

This will start an interactive session where you can ask questions about CMS dockets.

### Web Interface

Run the web interface:

```bash
streamlit run cms_agent_web.py
```

This will start a Streamlit web application where you can interact with the CMS Agent through a user-friendly interface.

### Features

- Query the vector store for relevant CMS documents
- Get AI-generated answers based on the retrieved context
- View source information for transparency
- Chat history for continued conversations
- Adjustable number of sources to retrieve

### Example Questions

- What are common issues with Medicare?
- Explain the latest changes to Medicaid eligibility
- What are the key points in the recent CMS final rule?
- How does CMS regulate telehealth services?
