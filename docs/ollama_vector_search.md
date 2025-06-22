# Ollama Vector Search Interactive Tool

## Overview

`ollama_vector_search.py` is an interactive command-line interface for managing and searching vector embeddings. It provides a menu-driven system for storing text embeddings, searching for similar content, and managing the vector store.

## Features

- Interactive menu-driven interface
- Real-time text embedding with Ollama
- Vector storage and retrieval
- Similarity search with configurable parameters
- Store maintenance and optimization
- Index save/load functionality
- Detailed store information display

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

### Starting the Interactive Interface

```bash
python ollama_vector_search.py ../vector_store.bin
```

For block device storage (requires root):
```bash
sudo python ollama_vector_search.py /dev/sdX
```

### Menu Options

1. **Store vectors from text** - Enter text to embed and store
2. **Retrieve vector by ID** - Fetch a specific vector by its ID
3. **Find closest matches** - Search for similar vectors
4. **Perform maintenance** - Optimize the vector store
5. **Save index to file** - Export the current index
6. **Load index from file** - Import a saved index
7. **Print store info** - Display storage statistics
8. **Exit** - Close the application

## Interactive Operations

### Storing Text Embeddings

```
Menu: 1
> Enter text to embed: Machine learning is a subset of artificial intelligence
Embedding generated in 0.15s
Stored vector with ID: 42
```

### Searching for Similar Content

```
Menu: 3
> Enter search text: AI and deep learning
> Number of results (default 5): 10

Search Results:
1. ID: 42, Score: 0.95 - "Machine learning is a subset of artificial intelligence"
2. ID: 23, Score: 0.87 - "Deep neural networks are powerful AI models"
...
```

### Retrieving Specific Vectors

```
Menu: 2
> Enter vector ID: 42
Original text: "Machine learning is a subset of artificial intelligence"
Stored on: 2025-06-22 15:30:45
```

## Configuration

The script uses several configurable parameters:

```python
EMBEDDING_MODEL = "nomic-embed-text:latest"  # Ollama model
VECTOR_DIM = 768                             # Embedding dimension
DEVICE_PATH = "./vector_store.bin"           # Storage location
LOG_FILE = "vector_store.log"                # Log file path
METADATA_FILE = "ollama_metadata.json"       # Metadata storage
```

## Metadata Management

The tool maintains metadata for all stored embeddings:

```json
{
  "next_id": 100,
  "entries": {
    "42": {
      "text": "Original text content",
      "timestamp": "2025-06-22T15:30:45"
    }
  },
  "vector_dim": 768
}
```

## Performance Features

### Clustering

The tool uses K-means clustering for efficient similarity search:
- Automatically clusters vectors for faster retrieval
- Searches only relevant clusters during queries
- Maintains cluster quality through periodic maintenance

### Memory-Mapped I/O

- Efficient handling of large vector stores
- Minimal memory footprint
- Fast random access to vectors

## Advanced Usage

### Batch Import from Previous Sessions

```python
# Load existing metadata
metadata = load_metadata()

# Continue from previous session
next_id = metadata['next_id']
```

### Custom Search Parameters

When searching, you can adjust:
- Number of results (k)
- Search specific clusters
- Apply similarity thresholds

## Troubleshooting

### Connection to Ollama Failed

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve
```

### Vector Dimension Mismatch

If you see dimension warnings:
1. Create a new vector store file
2. Or ensure consistent model usage

### Permission Denied (Block Devices)

For block device access:
```bash
sudo python ollama_vector_search.py /dev/sdX
```

## Integration with Other Tools

### Export for Batch Search

After building a vector store interactively, use it with the command-line search:

```bash
# Build interactively
python ollama_vector_search.py ../vector_store.bin

# Search non-interactively
python vector_search.py "query" --store ../vector_store.bin
```

### Combine with Batch Embedding

You can use both tools together:
1. Use `batch_embed_files.py` for bulk import
2. Use `ollama_vector_search.py` for interactive additions
3. Use `vector_search.py` for programmatic access

## See Also

- [Vector Search](vector_search.md) - Command-line search tool
- [Batch Embed Files](batch_embed.md) - Bulk embedding utility