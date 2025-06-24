# VectorClusterStore Test Suite

This directory contains the pytest-based test suite for VectorClusterStore.

## Running Tests

### Prerequisites

1. Build the project:
   ```bash
   cd ..
   ./build.sh
   pip install -e .
   ```

2. Install pytest:
   ```bash
   pip install pytest
   ```

3. For integration tests, ensure Ollama is running:
   ```bash
   ollama serve
   ```

### Run All Tests

```bash
# From the project root
pytest test/

# Or from within the test directory
cd test
pytest
```

### Run Specific Test Categories

```bash
# Only run unit tests (skip integration tests)
pytest test/test_vector_store.py

# Only run CLI tests
pytest test/test_ollama_tool.py

# Only run integration tests
pytest test/test_integration.py

# Skip tests that require Ollama
pytest -m "not requires_ollama"
```

### Run with Coverage

```bash
pip install pytest-cov
pytest test/ --cov=vector_cluster_store_py --cov-report=html
```

## Test Structure

### test_vector_store.py
- Unit tests for core VectorClusterStore functionality
- Tests basic operations: create, store, retrieve, search
- Tests persistence and batch operations
- No external dependencies required

### test_ollama_tool.py
- Tests for the ollama_tool.py CLI interface
- Tests all command modes: embed, search, maintenance, info
- Tests file creation, batch processing, and output formats
- Some tests require Ollama to be running

### test_integration.py
- End-to-end integration tests
- Tests complete workflows across multiple components
- Tests interaction between CLI and Python API
- Tests large-scale operations

### conftest.py
- Pytest configuration and shared fixtures
- Provides temporary directories, sample data, and module imports
- Checks for Ollama availability

## Fixtures

- `temp_dir`: Temporary directory for test files
- `vector_store_path`: Path for test vector store files
- `sample_texts`: Sample text data for testing
- `ollama_available`: Check if Ollama is running
- `vector_store_py`: Import of the C++ Python bindings
- `ollama_tool_path`: Path to ollama_tool.py script

## Writing New Tests

1. Create test functions starting with `test_`
2. Use fixtures for common setup
3. Clean up resources (fixtures handle this automatically)
4. Mark slow or integration tests appropriately:
   ```python
   @pytest.mark.slow
   def test_large_dataset():
       ...
   
   @pytest.mark.requires_ollama
   def test_embeddings():
       ...
   ```

## Debugging Failed Tests

```bash
# Run with more verbose output
pytest -vv test/

# Run with full traceback
pytest --tb=long test/

# Run specific test
pytest test/test_vector_store.py::TestVectorStore::test_store_and_retrieve_vector

# Run with print statements visible
pytest -s test/
```

## Continuous Integration

These tests are designed to work in CI environments. For CI:

1. Build the project first
2. Skip Ollama-dependent tests if Ollama isn't available
3. Use `pytest --tb=short -q` for concise output