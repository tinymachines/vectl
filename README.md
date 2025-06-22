# VectorClusterStore

A high-performance vector embedding storage system with clustering support, optimized for raw block devices. Perfect for semantic search, RAG systems, and other vector database applications.

## Features

- 🚀 Direct block device access for optimized performance
- 🎯 K-means clustering for efficient vector similarity search
- 🐍 Python bindings for seamless integration
- 💾 Memory-mapped I/O for high-throughput operations
- 📁 Support for both file and block device storage
- 🛠️ Command-line interface for management and testing

## Requirements

- Linux system (tested on Ubuntu 20.04+)
- C++17 compatible compiler (GCC 9+ or Clang 10+)
- CMake 3.10+
- Python 3.6+ (for Python bindings)
- Git (for pybind11 submodule)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/vector-cluster-store.git
cd vector-cluster-store

# Initialize pybind11 submodule
git submodule update --init --recursive
```

### Step 2: Install Python Dependencies

**Important**: You must install pybind11 via pip before building:

```bash
# Install pybind11 first
pip install pybind11

# Install other dependencies
pip install numpy
# Or install all dependencies at once
pip install -r requirements.txt
```

### Step 3: Build the Project

The project uses CMake for building. Note that the standalone `make` command in the root directory may not work properly - use the build script or CMake directly.

#### Option A: Using the Build Script (Recommended)

```bash
./build.sh        # Clean and build everything
```

#### Option B: Manual CMake Build

```bash
# Clean any previous build
rm -rf build/
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build with make (from within the build directory)
make -j$(nproc)

cd ..
```

**Note**: The `Makefile` in the root directory is legacy and may not include all dependencies. Always use CMake for building.

### Step 4: Install Python Bindings

After building the C++ components, install the Python module:

```bash
# Install in development mode (recommended for development)
pip install -e .
```

This step is crucial as it properly sets up the Python module paths and makes the `vector_cluster_store_py` module importable.

### Verify Installation

Test that everything is installed correctly:

```bash
# Test Python bindings
python test_binding.py

# If successful, you should see:
# Successfully imported vector_cluster_store_py
# Created logger
# Created vector store
# Test completed successfully!
```

## Common Installation Issues

### "No module named 'pybind11'"
**Solution**: Install pybind11 first with `pip install pybind11`

### "make: *** No targets specified and no makefile found"
**Solution**: Don't use `make` in the root directory. Use `./build.sh` or build from the `build/` directory with CMake.

### "ImportError: No module named vector_cluster_store_py"
**Solution**: Run `pip install -e .` after building with CMake to install the Python bindings.

### CMake succeeds but can't find the Python module
**Solution**: The CMake build creates the `.so` file in the `build/` directory, but Python needs it installed. Always run `pip install -e .` after building.

## Usage

### Python API Example

```python
import vector_cluster_store_py

# Create a logger
logger = vector_cluster_store_py.Logger("vector_store.log")

# Create and initialize vector store
store = vector_cluster_store_py.VectorClusterStore(logger)
store.initialize("./vector_store.bin", "kmeans", 768, 10)

# Store a vector
vector_id = 0
vector = [0.1, 0.2, 0.3, ...]  # Your 768-dimensional embedding
metadata = "Example metadata"
store.store_vector(vector_id, vector, metadata)

# Find similar vectors
query_vector = [0.1, 0.2, 0.3, ...]  # Query embedding
results = store.find_similar_vectors(query_vector, 5)  # Get top 5 matches
```

### Command Line Usage

For file-based storage (recommended for testing):
```bash
./ollama_vector_search.py ./vector_store.bin
```

For raw block device storage (requires root):
```bash
# Prepare device (WARNING: This will erase all data!)
sudo ./scripts/prepare_device.sh /dev/sdX

# Run with block device
sudo ./ollama_vector_search.py /dev/sdX
```

## Architecture

The system uses a clustered approach for efficient vector storage and retrieval:

```
┌───────────────────┐    ┌────────────────────┐    ┌──────────────────┐
│ Python/C++ App    │───▶│ VectorClusterStore │───▶│ Storage Device   │
│                   │    │ (C++ Core)         │    │ (File/Block)     │
└───────────────────┘    └────────────────────┘    └──────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │ K-means Clustering  │
                         │ (Similarity Search) │
                         └─────────────────────┘
```

### Storage Layout

```
┌──────────────────────────────────────────────────────────────┐
│                       Storage Device                         │
├────────────┬───────────────┬──────────────┬─────────────────┤
│ Header     │ Cluster Map   │ Vector Map   │ Vector Data     │
│ (512B)     │ (Index)       │ (Metadata)   │ (Embeddings)    │
└────────────┴───────────────┴──────────────┴─────────────────┘
```

## Performance

Performance comparison on a 128GB USB device with Raspberry Pi 4B:

| Operation | Traditional Filesystem | Raw Block Device | Improvement |
|-----------|------------------------|------------------|-------------|
| Sequential Write | 30-40 MB/s | 35-45 MB/s | 10-15% |
| Random Read | 5-10 MB/s | 15-25 MB/s | 150-200% |
| Vector Search (1M vectors) | 500-1000ms | 100-300ms | 70-80% |
| Memory Usage | 200-300MB | 50-100MB | 60-70% |

## Development

### Running Tests

```bash
# C++ unit tests
./build/test_cluster_store
./build/vector_store_test

# Python tests
python test_binding.py
python test_import.py
python test_search.py
```

### Building for Development

For development, always use the build script:
```bash
./build.sh clean   # Clean only
./build.sh rebuild # Clean and rebuild
```

## Troubleshooting

### Check Installation
```bash
# Verify CMake
cmake --version

# Verify pybind11
pip show pybind11

# Check built files
ls -la build/*.so
```

### Debug Import Issues
```python
import sys
print(sys.path)  # Check Python path
import vector_cluster_store_py  # Should work after pip install -e .
```

## Contributing

Contributions are welcome! Please ensure:
1. Code follows C++17 standards
2. Python code follows PEP 8
3. All tests pass
4. New features include tests

## License

[MIT License](LICENSE)

## Acknowledgments

- Built with [pybind11](https://github.com/pybind/pybind11) for Python bindings
- Optimized for vector similarity search use cases