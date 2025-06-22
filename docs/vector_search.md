# Vector Search Command-Line Tool

## Overview

`vector_search.py` is a command-line utility for searching vectors in a VectorClusterStore. It provides a simple interface for querying embedded vectors and retrieving similar documents based on semantic similarity.

## Features

- Command-line and stdin input support
- JSON and human-readable text output formats
- Configurable similarity thresholds
- Integration with Ollama for text embeddings
- Metadata support for rich search results

## Installation

The script is located in the `examples/` directory. Ensure you have built and installed the VectorClusterStore Python bindings first:

```bash
# Build the project
./build.sh

# Install Python bindings
pip install -e .

# Navigate to examples
cd examples
```

## Usage

### Basic Search

```bash
python vector_search.py "your search query"
```

### Search with Options

```bash
python vector_search.py "search query" --top-k 10 --output text
```

### Search via Standard Input

```bash
echo "search query" | python vector_search.py --output json
```

### Command-Line Options

- `--store PATH`: Path to vector store file (default: `../vector_store.bin`)
- `--metadata PATH`: Path to metadata JSON file (default: `../batch_embed_metadata.json`)
- `--model NAME`: Ollama model for embeddings (default: `nomic-embed-text:latest`)
- `--top-k N`: Number of results to return (default: 10)
- `--threshold FLOAT`: Similarity threshold between 0-1 (default: 0.0)
- `--output FORMAT`: Output format - `json` or `text` (default: `json`)
- `--no-embed`: Use raw query as vector (for testing with comma-separated floats)

## Output Formats

### JSON Format

```json
{
  "results": [
    {
      "index": 26,
      "score": 0.3074,
      "metadata": {
        "file_path": "document.txt",
        "content_preview": "Document content...",
        "timestamp": "2025-06-22T16:39:23"
      }
    }
  ]
}
```

### Text Format

```
1. Index: 26, Score: 0.3074
   File: document.txt
   Content: Document content preview...

2. Index: 2, Score: 0.2951
   File: another_document.txt
   Content: Another document content...
```

## Examples

### Search for Documents About Machine Learning

```bash
python vector_search.py "machine learning algorithms" --output text --top-k 5
```

### Filter Results by Similarity Threshold

```bash
python vector_search.py "neural networks" --threshold 0.5 --output json
```

### Use Custom Vector Store and Metadata

```bash
python vector_search.py "deep learning" \
    --store /path/to/custom_store.bin \
    --metadata /path/to/custom_metadata.json
```

### Test with Raw Vectors

```bash
python vector_search.py "0.1,0.2,0.3,0.4" --no-embed --top-k 3
```

## Integration with Other Tools

The JSON output format makes it easy to integrate with other tools:

```bash
# Extract just the file paths
python vector_search.py "query" | jq -r '.results[].metadata.file_path'

# Get the top result
python vector_search.py "query" | jq '.results[0]'

# Filter high-score results
python vector_search.py "query" | jq '.results[] | select(.score > 0.5)'
```

## Troubleshooting

### No Results Found

- Ensure the vector store file exists and contains vectors
- Check that the metadata file matches the vector store
- Verify Ollama is running and the model is available

### Import Errors

- Make sure you've built the project: `./build.sh`
- Install the Python bindings: `pip install -e .`
- Run from the examples directory

### Logging Output in JSON Mode

The C++ library may output debug logs to stdout. To get clean JSON:

```bash
# Redirect stderr (though some logs may still go to stdout)
python vector_search.py "query" 2>/dev/null

# Or filter the output
python vector_search.py "query" 2>/dev/null | tail -n +2
```

## See Also

- [Batch Embed Files](batch_embed.md) - For creating vector stores
- [Ollama Vector Search](ollama_vector_search.md) - Interactive search interface