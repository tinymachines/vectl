# Batch Embed Files

## Overview

`batch_embed_files.py` is a utility for batch processing documents to create vector embeddings. It reads files from a directory, chunks them appropriately, generates embeddings using Ollama, and stores them in a VectorClusterStore.

## Features

- Batch processing of multiple files
- Support for various file formats (text, markdown, etc.)
- Configurable chunking strategies
- Progress tracking and resumption
- Metadata preservation for each embedding
- Parallel processing support

## Installation

The script is located in the `examples/` directory:

```bash
# Build the project
./build.sh

# Install Python bindings
pip install -e .

# Navigate to examples
cd examples
```

## Usage

### Basic Usage

```bash
python batch_embed_files.py --input-dir /path/to/documents --output ../vector_store.bin
```

### Common Options

```bash
python batch_embed_files.py \
    --input-dir /path/to/documents \
    --output ../vector_store.bin \
    --model nomic-embed-text:latest \
    --chunk-size 1000 \
    --chunk-overlap 200 \
    --batch-size 10
```

### Command-Line Options

- `--input-dir PATH`: Directory containing files to embed (required)
- `--output PATH`: Output vector store file (default: `../vector_store.bin`)
- `--metadata PATH`: Output metadata JSON file (default: `../batch_embed_metadata.json`)
- `--model NAME`: Ollama model for embeddings (default: `nomic-embed-text:latest`)
- `--chunk-size N`: Size of text chunks in characters (default: 1000)
- `--chunk-overlap N`: Overlap between chunks (default: 200)
- `--batch-size N`: Number of files to process in parallel (default: 10)
- `--file-pattern PATTERN`: Glob pattern for files (default: `*`)
- `--resume`: Resume from previous run if interrupted

## File Processing

### Supported File Types

The script automatically detects and processes various file types:
- `.txt` - Plain text files
- `.md` - Markdown files
- `.log` - Log files
- Custom parsers can be added for other formats

### Chunking Strategy

Documents are split into chunks to:
- Fit within embedding model token limits
- Provide more granular search results
- Maintain context with overlapping chunks

### Metadata Storage

For each embedding, the following metadata is stored:
- `file_path`: Original file path
- `content_preview`: Preview of the chunk content
- `content_length`: Length of the chunk
- `chunk_index`: Position of chunk in file
- `timestamp`: When the embedding was created
- `embedding_model`: Model used for embedding
- `file_size`: Size of the original file

## Examples

### Process All Markdown Files

```bash
python batch_embed_files.py \
    --input-dir ./documentation \
    --file-pattern "*.md" \
    --output ../docs_vectors.bin
```

### Process Large Files with Custom Chunking

```bash
python batch_embed_files.py \
    --input-dir ./books \
    --chunk-size 2000 \
    --chunk-overlap 400 \
    --output ../books_vectors.bin
```

### Resume Interrupted Processing

```bash
python batch_embed_files.py \
    --input-dir ./large_dataset \
    --resume \
    --output ../dataset_vectors.bin
```

## Output Files

The script creates two files:

1. **Vector Store** (`vector_store.bin`): Binary file containing the embedded vectors
2. **Metadata** (`batch_embed_metadata.json`): JSON file with embedding metadata

### Metadata Format

```json
{
  "next_id": 160,
  "entries": {
    "0": {
      "file_path": "document.txt",
      "content_preview": "First 200 characters...",
      "content_length": 1000,
      "chunk_index": 0,
      "timestamp": "2025-06-22T16:39:22",
      "embedding_model": "nomic-embed-text:latest",
      "file_size": 5432
    }
  }
}
```

## Performance Considerations

- **Batch Size**: Larger batch sizes improve throughput but use more memory
- **Chunk Size**: Smaller chunks provide more granular search but increase storage
- **Model Selection**: Different models have different speed/quality tradeoffs

## Integration with Search

After creating embeddings, use the `vector_search.py` tool to search:

```bash
python vector_search.py "search query" \
    --store ../vector_store.bin \
    --metadata ../batch_embed_metadata.json
```

## Troubleshooting

### Out of Memory

- Reduce `--batch-size`
- Use smaller `--chunk-size`
- Process files in smaller batches

### Slow Processing

- Ensure Ollama is using GPU if available
- Increase `--batch-size` if memory allows
- Use a faster embedding model

### Failed Embeddings

- Check Ollama is running: `ollama list`
- Verify model is downloaded: `ollama pull nomic-embed-text:latest`
- Check file permissions and paths

## See Also

- [Vector Search](vector_search.md) - Search embedded vectors
- [Ollama Vector Search](ollama_vector_search.md) - Interactive search interface