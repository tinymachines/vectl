# VectorClusterStore Documentation

Welcome to the VectorClusterStore documentation. This directory contains detailed guides for using the various tools and components of the project.

## Python Tools Documentation

### [Vector Search](vector_search.md)
Command-line tool for searching vectors in a VectorClusterStore. Provides simple interface for querying embedded vectors with JSON/text output formats.

### [Batch Embed Files](batch_embed.md)
Utility for batch processing documents to create vector embeddings. Supports various file formats, configurable chunking, and parallel processing.

### [Ollama Vector Search](ollama_vector_search.md)
Interactive command-line interface for managing and searching vector embeddings. Features menu-driven system for real-time embedding and search operations.

## Getting Started

1. **Build the Project**: Follow the installation instructions in the main [README](../README.md)
2. **Create Embeddings**: Use [batch_embed_files.py](batch_embed.md) to process your documents
3. **Search Vectors**: Use [vector_search.py](vector_search.md) for command-line searches or [ollama_vector_search.py](ollama_vector_search.md) for interactive exploration

## Example Workflow

```bash
# 1. Build and install
cd /path/to/vectl
./build.sh
pip install -e .

# 2. Create embeddings from documents
cd examples
python batch_embed_files.py --input-dir /path/to/docs --output ../vectors.bin

# 3. Search the embeddings
python vector_search.py "machine learning" --output text --top-k 5

# 4. Or use interactive search
python ollama_vector_search.py ../vectors.bin
```

## Architecture Documentation

- [CLAUDE.md](../CLAUDE.md) - AI assistant guidance and project overview
- [Test Suite](TESTSUITE.md) - Testing procedures and examples

## Additional Resources

- [Main README](../README.md) - Installation and basic usage
- [Examples Directory](../examples/) - Working code examples
- [Scripts Directory](../scripts/) - Utility scripts for device management