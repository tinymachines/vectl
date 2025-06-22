# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VectorClusterStore is a high-performance vector embedding storage system with clustering support, optimized for raw block devices. It's designed for semantic search, RAG systems, and other vector database applications. The project provides both C++ libraries and Python bindings via pybind11.

## Build Commands

### Clean and Build
```bash
./build.sh          # Clean and build all components
./build.sh clean    # Clean only
./build.sh rebuild  # Clean and rebuild
```

### Manual Build (if build script fails)
```bash
rm -rf build/
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ..
```

### Python Module Installation
```bash
pip install -e .    # Install Python bindings in development mode
```

### Running Tests
```bash
# C++ tests
./build/test_cluster_store
./build/raw_device_test
./build/vector_store_test

# Python binding tests
python test_binding.py
python test_import.py
python test_search.py
```

## Architecture

### Core Components

1. **VectorClusterStore** (`src/vector_cluster_store.h/cpp`)
   - Main storage engine handling vector persistence and retrieval
   - Supports both file-based and raw block device storage
   - Memory-mapped I/O for performance

2. **KMeansClustering** (`src/kmeans_clustering.h/cpp`)
   - Implements k-means clustering for efficient similarity search
   - Integrates with VectorClusterStore via ClusteringInterface

3. **Logger** (`src/logger.h`)
   - Thread-safe logging system used throughout the codebase

4. **Python Bindings** (`src/python_bindings.cpp`)
   - Exposes C++ functionality to Python using pybind11
   - Module name: `vector_cluster_store_py`

### Storage Layout
- Header (512B): Contains metadata and configuration
- Cluster Map Region: Stores clustering information
- Vector Map Region: Maps vector IDs to storage locations
- Vector Data Region: Actual vector embeddings and metadata

## Development Guidelines

### Adding New Features
1. C++ code follows C++17 standard
2. Python bindings should expose new functionality in `src/python_bindings.cpp`
3. Test new features in both C++ (`src/test_*.cpp`) and Python (`test_*.py`)

### Code Style
- C++ uses standard library conventions
- Python code should follow PEP 8
- Use existing logging infrastructure via Logger class

### Testing Approach
- Unit tests in `src/test_*.cpp` for C++ components
- Python tests in root directory (`test_*.py`)
- Performance tests in `vector_store_test.cpp`

## Important Notes

### pybind11 Submodule
The `./pybind11` directory is a git submodule and should be:
- Excluded from check-ins to this repository
- Properly initialized when cloning: `git submodule update --init --recursive`
- Not modified directly in this repository

### Dependencies
- Core: CMake 3.10+, C++17 compiler, pybind11
- Python: numpy, pybind11 (see requirements.txt)
- Optional: Ollama for text embeddings

### File vs Block Device Storage
- Use file-based storage for development/testing: `./vector_store.bin`
- Block device storage requires root privileges and device preparation
- Scripts in `scripts/` handle device preparation and reset