# VectorClusterStore Test Suite

This document describes the testing procedures and test files for VectorClusterStore.

## C++ Tests

### Building Tests

```bash
# Using the build script (recommended)
./build.sh

# Or manually with CMake
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ..
```

### Unit Tests

#### test_cluster_store
Tests the core clustering functionality:
```bash
./build/test_cluster_store
```

#### vector_store_test
Comprehensive test of vector storage operations:
```bash
./build/vector_store_test
```

#### raw_device_test
Tests raw device I/O operations:
```bash
./build/raw_device_test
```

### Manual C++ Testing

To compile and run tests manually:

```bash
# Compile test
g++ -std=c++17 -o vector_store_test \
    src/vector_store_test.cpp \
    src/vector_cluster_store.cpp \
    src/kmeans_clustering.cpp \
    -I./src -pthread

# Run test
./vector_store_test
```

## Python Tests

### Basic Import Test

Tests that the Python module can be imported:
```bash
python test_import.py
```

Expected output:
```
Successfully imported vector_cluster_store_py
Module path: /path/to/build/vector_cluster_store_py.cpython-312-x86_64-linux-gnu.so
Available classes/functions: ['Logger', 'VectorClusterStore', ...]
```

### Binding Test

Tests basic Python bindings functionality:
```bash
python test_binding.py
```

Expected output:
```
Successfully imported vector_cluster_store_py
Created logger
Created vector store
Test completed successfully!
```

### Search Test

Tests vector storage and retrieval with random vectors:
```bash
python test_search.py
```

This test:
- Creates a vector store
- Generates random 768-dimensional vectors
- Stores vectors with metadata
- Performs similarity search
- Verifies results

## Integration Tests

### End-to-End Embedding and Search

```bash
# Create test documents
mkdir -p test_docs
echo "Machine learning is awesome" > test_docs/doc1.txt
echo "Deep learning with neural networks" > test_docs/doc2.txt
echo "Natural language processing" > test_docs/doc3.txt

# Embed documents
cd examples
python batch_embed_files.py --input-dir ../test_docs --output ../test_vectors.bin

# Search embeddings
python vector_search.py "AI and machine learning" --store ../test_vectors.bin --output text

# Clean up
rm -rf ../test_docs ../test_vectors.bin ../batch_embed_metadata.json
```

### Performance Testing

Test with larger datasets:
```bash
# Generate test data
python -c "
import os
os.makedirs('perf_test_docs', exist_ok=True)
for i in range(1000):
    with open(f'perf_test_docs/doc_{i}.txt', 'w') as f:
        f.write(f'Document {i}: ' + ' '.join([f'word{j}' for j in range(100)]))
"

# Time the embedding process
time python examples/batch_embed_files.py \
    --input-dir perf_test_docs \
    --output perf_test.bin \
    --batch-size 50

# Test search performance
time python examples/vector_search.py "word42" \
    --store perf_test.bin \
    --top-k 100
```

## Test Coverage Areas

### Storage Operations
- [x] Vector storage and retrieval
- [x] Metadata handling
- [x] File-based storage
- [x] Block device storage
- [x] Memory-mapped I/O

### Clustering
- [x] K-means clustering
- [x] Cluster assignment
- [x] Similarity search within clusters
- [x] Cluster maintenance

### Python Bindings
- [x] Module import
- [x] Logger creation
- [x] Store initialization
- [x] Vector operations
- [x] Search functionality

### Error Handling
- [x] Invalid dimensions
- [x] Missing files
- [x] Permission errors
- [x] Corrupted data

## Continuous Integration

For CI/CD pipelines, run:

```bash
#!/bin/bash
set -e

# Build
./build.sh

# Install Python module
pip install -e .

# Run C++ tests
./build/test_cluster_store
./build/vector_store_test

# Run Python tests
python test_import.py
python test_binding.py
python test_search.py

# Run integration test
cd examples
python -c "print('Test content')" > test.txt
python batch_embed_files.py --input-dir . --file-pattern "test.txt" --output test.bin
python vector_search.py "test" --store test.bin --output json
rm -f test.txt test.bin batch_embed_metadata.json
```

## Debugging Tests

### Enable Debug Logging

```python
# In Python scripts
logger = vector_cluster_store_py.Logger("debug.log")
```

### Check Build Artifacts

```bash
# Verify .so file exists
ls -la build/*.so

# Check symbols
nm -D build/vector_cluster_store_py*.so | grep -i vector
```

### Common Test Issues

1. **ImportError**: Run `pip install -e .` after building
2. **Segmentation Fault**: Check vector dimensions match
3. **Permission Denied**: Use appropriate permissions for block devices
4. **File Not Found**: Ensure working directory is correct