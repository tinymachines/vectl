"""
Pytest configuration and fixtures for VectorClusterStore tests
"""
import pytest
import tempfile
import shutil
import os
import sys
import subprocess

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build"))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp(prefix="vectl_test_")
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def vector_store_path(temp_dir):
    """Return a path for test vector store"""
    return os.path.join(temp_dir, "test_vectors.bin")


@pytest.fixture
def sample_texts():
    """Sample texts for testing"""
    return [
        "Machine learning is a subset of artificial intelligence",
        "Vector databases are optimized for similarity search",
        "Embeddings capture semantic meaning of text",
        "Clustering improves search performance in large datasets",
        "Python is a popular programming language for data science"
    ]


@pytest.fixture
def ollama_available():
    """Check if Ollama is available and running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


@pytest.fixture
def vector_store_py():
    """Import and return the vector_cluster_store_py module"""
    try:
        import vector_cluster_store_py
        return vector_cluster_store_py
    except ImportError:
        pytest.skip("vector_cluster_store_py module not available. Run build.sh first.")


@pytest.fixture
def ollama_tool_path():
    """Path to ollama_tool.py"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                       "examples", "ollama_tool.py")