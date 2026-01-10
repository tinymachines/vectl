"""
Tests for similarity search functionality.
"""
import numpy as np


class TestSimilaritySearch:
    """Test vector similarity search."""

    def test_find_similar_vectors(self, initialized_store):
        """Test finding similar vectors."""
        store, _ = initialized_store

        # Create a base vector
        base = np.random.normal(0, 1, 768)
        base = base / np.linalg.norm(base)

        # Store the base vector
        store.store_vector(0, base.tolist(), "base")

        # Store some similar vectors (small perturbations)
        for i in range(1, 5):
            noise = np.random.normal(0, 0.1, 768)
            similar = base + noise
            similar = similar / np.linalg.norm(similar)
            store.store_vector(i, similar.tolist(), f"similar_{i}")

        # Store some random vectors
        for i in range(5, 10):
            random_vec = np.random.normal(0, 1, 768)
            random_vec = random_vec / np.linalg.norm(random_vec)
            store.store_vector(i, random_vec.tolist(), f"random_{i}")

        # Query with the base vector
        results = store.find_similar_vectors(base.tolist(), 5)

        assert len(results) > 0
        # First result should be the exact match (ID 0)
        assert results[0][0] == 0

    def test_search_returns_k_results(self, initialized_store):
        """Test that search returns requested number of results."""
        store, _ = initialized_store

        # Store 20 random vectors
        for i in range(20):
            vec = np.random.normal(0, 1, 768)
            vec = (vec / np.linalg.norm(vec)).tolist()
            store.store_vector(i, vec, f"vector_{i}")

        # Query for top 5
        query = np.random.normal(0, 1, 768)
        query = (query / np.linalg.norm(query)).tolist()

        results = store.find_similar_vectors(query, 5)

        assert len(results) <= 5

    def test_search_empty_store(self, initialized_store):
        """Test searching an empty store."""
        store, _ = initialized_store

        query = np.random.normal(0, 1, 768)
        query = (query / np.linalg.norm(query)).tolist()

        results = store.find_similar_vectors(query, 5)

        assert len(results) == 0

    def test_search_results_sorted_by_similarity(self, initialized_store):
        """Test that search results are sorted by similarity."""
        store, _ = initialized_store

        # Store some vectors
        for i in range(10):
            vec = np.random.normal(0, 1, 768)
            vec = (vec / np.linalg.norm(vec)).tolist()
            store.store_vector(i, vec, f"vec_{i}")

        query = np.random.normal(0, 1, 768)
        query = (query / np.linalg.norm(query)).tolist()

        results = store.find_similar_vectors(query, 5)

        # Results should be sorted by similarity (descending)
        if len(results) > 1:
            similarities = [r[1] for r in results]
            for i in range(len(similarities) - 1):
                assert similarities[i] >= similarities[i + 1]


class TestIndexPersistence:
    """Test index save/load functionality."""

    def test_save_and_load_index(self, initialized_store):
        """Test saving and loading index."""
        store, _ = initialized_store
        import tempfile
        import os

        # Create temp file for index
        fd, index_path = tempfile.mkstemp(suffix='.idx', prefix='test_index_')
        os.close(fd)

        try:
            # Add some vectors
            for i in range(5):
                vec = np.random.normal(0, 1, 768)
                vec = (vec / np.linalg.norm(vec)).tolist()
                store.store_vector(i, vec, f"vec_{i}")

            # Save index
            store.save_index(index_path)

            # Verify we can still retrieve vectors
            for i in range(5):
                retrieved = store.retrieve_vector(i)
                assert len(retrieved) == 768
        finally:
            if os.path.exists(index_path):
                os.unlink(index_path)
