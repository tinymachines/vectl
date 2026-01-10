"""
pytest configuration and shared fixtures for vector-store tests.
"""
import pytest
import tempfile
import os
import numpy as np


@pytest.fixture
def temp_store_path():
    """Create a temporary file path for a vector store."""
    fd, path = tempfile.mkstemp(suffix='.bin', prefix='test_vector_store_')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def temp_log_path():
    """Create a temporary log file path."""
    fd, path = tempfile.mkstemp(suffix='.log', prefix='test_')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_vector():
    """Return a sample normalized 768-dimensional vector."""
    vec = np.random.normal(0, 1, 768)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


@pytest.fixture
def initialized_store(temp_store_path, temp_log_path):
    """
    Create and initialize a vector store for testing.

    Returns a tuple of (store, logger) for tests that need both.
    """
    import vector_cluster_store_py

    logger = vector_cluster_store_py.Logger(temp_log_path)
    store = vector_cluster_store_py.VectorClusterStore(logger)
    success = store.initialize(temp_store_path, "kmeans", 768, 10)
    assert success, "Failed to initialize vector store"

    yield store, logger
