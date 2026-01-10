# Vector Cluster Store

A high-performance vector embedding storage system with clustering support, optimized for raw block devices. Perfect for semantic search, RAG systems, and other vector database applications.

## Features

- Direct block device access for optimized performance
- K-means clustering for efficient vector similarity search
- Python bindings for seamless integration
- Memory-mapped I/O for high-throughput operations
- Support for both file and block device storage
- **fastcomp** CLI tool for quick text similarity comparisons

## Requirements

- Linux system (tested on Ubuntu 20.04+)
- C++17 compatible compiler (GCC 9+ or Clang 10+)
- Python 3.8+
- pybind11
- numpy

## Installation

### Quick Install (Recommended)

```bash
pip install -e .
```

This compiles the C++ extension and installs the Python package in one step.

### Install with Development Dependencies

```bash
pip install -e ".[dev]"
```

Includes pytest for running tests.

### Verify Installation

```bash
python -c "from vector_store import create_store; print('Installation OK')"
```

## Usage

### Python API

```python
from vector_store import create_store

# Create and initialize a vector store
store = create_store("./vector_store.bin", vector_dim=768, num_clusters=10)

# Store vectors
store.store_vector(0, [0.1] * 768, "document metadata")
store.store_vector(1, [0.2] * 768, "another document")

# Find similar vectors
results = store.find_similar_vectors([0.15] * 768, k=5)
for vector_id, similarity in results:
    print(f"ID: {vector_id}, Similarity: {similarity:.4f}")

# Retrieve a vector
vector = store.retrieve_vector(0)
metadata = store.get_vector_metadata(0)
```

### Low-Level API

```python
import vector_cluster_store_py

# Create a logger
logger = vector_cluster_store_py.Logger("vector_store.log")

# Create and initialize vector store
store = vector_cluster_store_py.VectorClusterStore(logger)
store.initialize("./vector_store.bin", "kmeans", 768, 10)

# Store/retrieve vectors
store.store_vector(0, [0.1] * 768, "metadata")
retrieved = store.retrieve_vector(0)
```

### fastcomp CLI

Compare text similarity using Ollama embeddings:

```bash
# Install with Ollama support
pip install -e ".[ollama]"

# Compare texts (first line is basis, rest are compared against it)
echo -e 'Michigan\nDetroit\nChicago\nCalifornia' | fastcomp

# Use Euclidean distance instead of cosine
printf 'cat\ndog\ncar\n' | fastcomp -m euclidean

# Use a different model
echo -e 'hello\nworld' | fastcomp --model mxbai-embed-large
```

Output shows distance values (lower = more similar):
```
0.123456    # Detroit vs Michigan
0.234567    # Chicago vs Michigan
0.345678    # California vs Michigan
```

Requires Ollama running locally (`ollama serve`).

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ -v --cov=vector_store
```

## Development

### Building with CMake (for C++ development)

```bash
./build.sh          # Build everything
./build.sh clean    # Clean build artifacts
./build.sh rebuild  # Full clean and rebuild
```

### C++ Test Executables

After CMake build:

```bash
./build/test_cluster_store <device>     # Basic storage tests
./build/vector_store_test               # Performance tests
./build/raw_device_test                 # Block device tests
```

## Raw Block Device Storage (Advanced)

For production use with high-performance requirements:

```bash
# Prepare a dedicated block device (WARNING: erases all data)
sudo ./prepare_device.sh /dev/sdX

# Run with block device
sudo python ollama_vector_search.py /dev/sdX
```

## Architecture

```
┌───────────────────┐    ┌────────────────────┐    ┌──────────────────┐
│ LLM Application   │    │ Vector Cluster     │    │                  │
│ (Query Interface) │───►│ Storage Library    │───►│ Storage Device   │
└───────────────────┘    └────────────────────┘    └──────────────────┘
                                    ▲                        │
                                    │                        │
                         ┌──────────┴──────────┐             │
                         │ Clustering Index    │◄────────────┘
                         └─────────────────────┘
```

### Block Device Layout

```
┌──────────────────────────────────────────────────────────────┐
│                       Block Device                           │
├────────────┬───────────────┬──────────────┬──────────────────┤
│ Header     │ Cluster Map   │ Vector Map   │ Vector Data      │
│ (512B)     │ Region        │ Region       │ Region           │
└────────────┴───────────────┴──────────────┴──────────────────┘
```

## Performance

Comparison on 128GB USB device with Raspberry Pi 4B:

| Operation | Filesystem | Raw Block Device | Improvement |
|-----------|------------|------------------|-------------|
| Sequential Write | 30-40 MB/s | 35-45 MB/s | 10-15% |
| Random Read | 5-10 MB/s | 15-25 MB/s | 150-200% |
| Vector Search (1M) | 500-1000ms | 100-300ms | 70-80% |
| Memory Usage | 200-300MB | 50-100MB | 60-70% |

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
