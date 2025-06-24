# VectorClusterStore Examples

This directory contains example scripts demonstrating how to use VectorClusterStore with various embedding models and use cases.

## Prerequisites

1. Build and install the VectorClusterStore Python bindings:
   ```bash
   cd ..
   ./build.sh
   pip install -e .
   ```

2. Install required Python packages:
   ```bash
   pip install requests numpy
   # For table formatting in ollama_tool.py
   pip install tabulate
   ```

3. For Ollama-based examples, ensure Ollama is running:
   ```bash
   ollama serve
   ```

## Available Examples

### 1. ollama_tool.py - Unified Embedding and Search Tool (Recommended)

A comprehensive command-line tool that combines embedding and search functionality with flexible options.

#### Features:
- Multiple operation modes: embed, search, maintenance, info
- Flexible input/output options (stdin, command-line, JSON/text output)
- Custom index locations with automatic metadata management
- Verbose mode for debugging

#### Usage Examples:

```bash
# Embed a single text
python ollama_tool.py embed --text "Hello world"

# Embed multiple texts from stdin
cat documents.txt | python ollama_tool.py embed

# Search for similar texts (table output)
python ollama_tool.py search --query "machine learning" --top-k 10

# Search with JSON output for integration
python ollama_tool.py search --query "vectors" --format json

# Use custom storage location
python ollama_tool.py embed --index ~/myproject/vectors.bin --text "Project data"

# Run maintenance to optimize clusters
python ollama_tool.py maintenance --index ~/myproject/vectors.bin

# Show detailed store information
python ollama_tool.py info --verbose
```

### 2. ollama_vector_search.py - Interactive Search Interface (Legacy)

An interactive shell for managing vector embeddings with Ollama.

```bash
# Default usage
python ollama_vector_search.py

# Custom storage path
python ollama_vector_search.py /path/to/vector_store.bin
```

Features an interactive menu for:
- Storing text embeddings
- Retrieving vectors by ID
- Finding similar texts
- Performing maintenance
- Managing indexes

### 3. batch_embed_files.py - Bulk File Embedding

Embed multiple text files in batch mode.

```bash
# Find and embed all text files
find ~/documents -name "*.txt" | python batch_embed_files.py

# Custom storage location
find ~/documents -name "*.md" | python batch_embed_files.py ~/vectors/docs.bin

# With custom metadata file
python batch_embed_files.py --metadata-file custom_metadata.json
```

### 4. vector_search.py - Command-Line Search

Simple command-line interface for searching vectors.

```bash
# Basic search
python vector_search.py "search query"

# Search with options
python vector_search.py "query" --top-k 20 --output text

# Custom paths
python vector_search.py "query" --store ../custom.bin --metadata ../custom_metadata.json
```

### 5. test_*.py - Testing Scripts

Various test scripts to verify installation and functionality:

```bash
# Test Python bindings
python test_binding.py

# Test import functionality
python test_import.py

# Test search operations
python test_search.py
```

## Storage Organization

All examples follow a consistent pattern for organizing files:

```
/path/to/your/index/
├── vector_store.bin          # Binary vector storage
├── vector_store.log          # Operation logs
└── vector_store_metadata.json # Text and metadata
```

When you specify a custom index path like `--index ~/myproject/vectors.bin`, the tool automatically creates:
- `~/myproject/vectors.bin` - Vector storage
- `~/myproject/vectors.log` - Log file
- `~/myproject/vectors_metadata.json` - Metadata

## Performance Tips

1. **Batch Operations**: Use stdin for embedding multiple texts at once
2. **Maintenance**: Run maintenance periodically to optimize cluster distribution
3. **Custom Dimensions**: Match `--dimension` to your embedding model's output
4. **Cluster Count**: Adjust `--clusters` based on your dataset size (10-100 typical)

## Troubleshooting

### Import Errors
```bash
# Ensure the module is built and installed
cd ..
./build.sh
pip install -e .
```

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Use custom URL if needed
python ollama_tool.py embed --ollama-url http://remote:11434 --text "test"
```

### Permission Errors
```bash
# Ensure write permissions for index location
mkdir -p ~/myproject
python ollama_tool.py embed --index ~/myproject/vectors.bin --text "test"
```

## Integration Examples

### Shell Script Integration
```bash
#!/bin/bash
# embed_and_search.sh

# Embed new documents
find ~/new_docs -name "*.txt" -mtime -1 | python ollama_tool.py embed --index ~/vectors.bin

# Search and process results
python ollama_tool.py search --query "$1" --format json --index ~/vectors.bin | jq '.results[] | {id, score, text}'
```

### Python Integration
```python
import subprocess
import json

# Embed text
result = subprocess.run(
    ["python", "ollama_tool.py", "embed", "--text", "Hello world", "--format", "json"],
    capture_output=True, text=True
)

# Search
result = subprocess.run(
    ["python", "ollama_tool.py", "search", "--query", "Hello", "--format", "json"],
    capture_output=True, text=True
)
search_results = json.loads(result.stdout)
```

## Next Steps

- Experiment with different embedding models by changing `--model`
- Try different cluster counts for your use case
- Build automation scripts using the JSON output format
- Integrate with your existing data pipelines