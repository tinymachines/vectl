"""
Tests for the core VectorClusterStore functionality
"""
import pytest
import numpy as np
import json
import os


class TestVectorStore:
    """Test basic vector store operations"""
    
    def test_import_module(self, vector_store_py):
        """Test that the module can be imported"""
        assert vector_store_py is not None
        assert hasattr(vector_store_py, 'VectorClusterStore')
        assert hasattr(vector_store_py, 'Logger')
    
    def test_create_logger(self, vector_store_py, temp_dir):
        """Test logger creation"""
        log_path = os.path.join(temp_dir, "test.log")
        logger = vector_store_py.Logger(log_path)
        assert logger is not None
    
    def test_create_vector_store(self, vector_store_py, temp_dir):
        """Test vector store creation"""
        log_path = os.path.join(temp_dir, "test.log")
        logger = vector_store_py.Logger(log_path)
        store = vector_store_py.VectorClusterStore(logger)
        assert store is not None
    
    def test_initialize_store(self, vector_store_py, vector_store_path):
        """Test store initialization"""
        # Touch the file first
        open(vector_store_path, 'a').close()
        
        logger = vector_store_py.Logger("")  # No log file
        store = vector_store_py.VectorClusterStore(logger)
        
        # Initialize with default parameters
        success = store.initialize(vector_store_path, "kmeans", 128, 10)
        assert success
    
    def test_store_and_retrieve_vector(self, vector_store_py, vector_store_path):
        """Test storing and retrieving a vector"""
        # Touch the file first
        open(vector_store_path, 'a').close()
        
        logger = vector_store_py.Logger("")
        store = vector_store_py.VectorClusterStore(logger)
        
        # Initialize store
        vector_dim = 128
        store.initialize(vector_store_path, "kmeans", vector_dim, 10)
        
        # Create test vector
        test_vector = np.random.rand(vector_dim).tolist()
        vector_id = 0
        metadata = json.dumps({"text": "Test vector", "id": vector_id})
        
        # Store vector
        success = store.store_vector(vector_id, test_vector, metadata)
        assert success
        
        # Retrieve vector
        retrieved = store.retrieve_vector(vector_id)
        assert len(retrieved) == vector_dim
        
        # Check values are close (accounting for float precision)
        np.testing.assert_allclose(test_vector, retrieved, rtol=1e-5)
    
    def test_find_similar_vectors(self, vector_store_py, vector_store_path):
        """Test similarity search"""
        # Touch the file first
        open(vector_store_path, 'a').close()
        
        logger = vector_store_py.Logger("")
        store = vector_store_py.VectorClusterStore(logger)
        
        # Initialize store
        vector_dim = 128
        store.initialize(vector_store_path, "kmeans", vector_dim, 10)
        
        # Store multiple vectors
        vectors = []
        for i in range(5):
            vec = np.random.rand(vector_dim).tolist()
            vectors.append(vec)
            metadata = json.dumps({"text": f"Vector {i}", "id": i})
            store.store_vector(i, vec, metadata)
        
        # Search for similar vectors
        query_vector = vectors[0]  # Use first vector as query
        results = store.find_similar_vectors(query_vector, 3)
        
        assert len(results) > 0
        assert len(results) <= 3
        
        # First result should be the query vector itself with high similarity
        assert results[0][0] == 0  # ID of first vector
        assert results[0][1] > 0.99  # Very high similarity
    
    def test_maintenance(self, vector_store_py, vector_store_path):
        """Test maintenance operation"""
        # Touch the file first
        open(vector_store_path, 'a').close()
        
        logger = vector_store_py.Logger("")
        store = vector_store_py.VectorClusterStore(logger)
        
        # Initialize and add vectors
        vector_dim = 128
        store.initialize(vector_store_path, "kmeans", vector_dim, 10)
        
        # Add some vectors
        for i in range(10):
            vec = np.random.rand(vector_dim).tolist()
            store.store_vector(i, vec, f"Vector {i}")
        
        # Perform maintenance
        success = store.perform_maintenance()
        assert success
    
    def test_persistence(self, vector_store_py, vector_store_path):
        """Test that data persists across store instances"""
        logger = vector_store_py.Logger("")
        vector_dim = 128
        
        # First instance - store data
        # Touch the file first
        open(vector_store_path, 'a').close()
        
        store1 = vector_store_py.VectorClusterStore(logger)
        store1.initialize(vector_store_path, "kmeans", vector_dim, 10)
        
        test_vector = np.random.rand(vector_dim).tolist()
        vector_id = 42
        metadata = json.dumps({"text": "Persistence test"})
        
        success = store1.store_vector(vector_id, test_vector, metadata)
        assert success
        
        # Delete first instance
        del store1
        
        # Second instance - retrieve data
        store2 = vector_store_py.VectorClusterStore(logger)
        store2.initialize(vector_store_path, "kmeans", vector_dim, 10)
        
        retrieved = store2.retrieve_vector(vector_id)
        assert len(retrieved) == vector_dim
        np.testing.assert_allclose(test_vector, retrieved, rtol=1e-5)
    
    def test_large_batch(self, vector_store_py, vector_store_path):
        """Test storing a large batch of vectors"""
        # Touch the file first
        open(vector_store_path, 'a').close()
        
        logger = vector_store_py.Logger("")
        store = vector_store_py.VectorClusterStore(logger)
        
        # Initialize store
        vector_dim = 128
        store.initialize(vector_store_path, "kmeans", vector_dim, 10)
        
        # Store 100 vectors
        batch_size = 100
        for i in range(batch_size):
            vec = np.random.rand(vector_dim).tolist()
            success = store.store_vector(i, vec, f"Batch vector {i}")
            assert success
        
        # Verify we can retrieve them
        for i in [0, 50, 99]:  # Check first, middle, last
            retrieved = store.retrieve_vector(i)
            assert len(retrieved) == vector_dim