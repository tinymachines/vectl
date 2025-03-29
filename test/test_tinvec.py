# tests/test_vector_store.py
import pytest
import numpy as np
from vector_store import VectorStore

def test_vector_store_basic():
    # Create a temporary file for testing
    import tempfile
    with tempfile.NamedTemporaryFile() as tmp:
        store = VectorStore(tmp.name, 10)
        
        # Create a test vector
        vec = np.random.randn(10).astype(np.float32)
        
        # Test store and retrieve
        assert store.store_vector(1, vec)
        retrieved = store.get_vector(1)
        
        # Check if vectors match
        np.testing.assert_allclose(vec, retrieved, rtol=1e-5)
