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


class TestClusterPersistence:
    """Test that cluster state persists across store reopens."""

    def test_similarity_search_after_reopen(self, temp_log_path):
        """
        Test that similarity search works after reopening the store.

        This is a regression test for the cluster map serialization bug
        where centroids were not properly restored on reopen.
        """
        import tempfile
        import os
        import vector_cluster_store_py

        # Create temp file for store
        fd, store_path = tempfile.mkstemp(suffix='.bin', prefix='test_cluster_persist_')
        os.close(fd)

        try:
            # Create a known base vector
            base_vector = np.random.normal(0, 1, 768)
            base_vector = (base_vector / np.linalg.norm(base_vector)).tolist()

            # Phase 1: Create store and add vectors
            logger1 = vector_cluster_store_py.Logger(temp_log_path)
            store1 = vector_cluster_store_py.VectorClusterStore(logger1)
            assert store1.initialize(store_path, "kmeans", 768, 10)

            # Store the base vector and some similar vectors
            store1.store_vector(0, base_vector, "base")
            for i in range(1, 10):
                # Create vectors similar to base
                noise = np.random.normal(0, 0.1, 768)
                similar = np.array(base_vector) + noise
                similar = (similar / np.linalg.norm(similar)).tolist()
                store1.store_vector(i, similar, f"similar_{i}")

            # Verify search works before closing
            results_before = store1.find_similar_vectors(base_vector, 5)
            assert len(results_before) > 0, "Search should find vectors before close"

            # Close the store (Python garbage collection)
            del store1
            del logger1

            # Phase 2: Reopen store and verify search still works
            logger2 = vector_cluster_store_py.Logger(temp_log_path + ".2")
            store2 = vector_cluster_store_py.VectorClusterStore(logger2)
            assert store2.initialize(store_path, "kmeans", 768, 10)

            # Search with the same base vector
            results_after = store2.find_similar_vectors(base_vector, 5)

            # This was the bug: after reopen, search returned 0 results
            assert len(results_after) > 0, \
                "Similarity search should return results after reopen (cluster persistence bug)"

            # The base vector (ID 0) should be in the results
            result_ids = [r[0] for r in results_after]
            assert 0 in result_ids, "Base vector should be found in search results"

            del store2
            del logger2

        finally:
            # Cleanup
            if os.path.exists(store_path):
                os.unlink(store_path)
            if os.path.exists(temp_log_path + ".2"):
                os.unlink(temp_log_path + ".2")

    def test_vector_retrieval_after_reopen(self, temp_log_path):
        """Test that vectors can be retrieved after reopening the store."""
        import tempfile
        import os
        import vector_cluster_store_py

        fd, store_path = tempfile.mkstemp(suffix='.bin', prefix='test_retrieve_persist_')
        os.close(fd)

        try:
            # Create test vectors
            vectors = {}
            for i in range(5):
                vec = np.random.normal(0, 1, 768)
                vec = (vec / np.linalg.norm(vec)).tolist()
                vectors[i] = vec

            # Phase 1: Create and populate store
            logger1 = vector_cluster_store_py.Logger(temp_log_path)
            store1 = vector_cluster_store_py.VectorClusterStore(logger1)
            assert store1.initialize(store_path, "kmeans", 768, 10)

            for vid, vec in vectors.items():
                store1.store_vector(vid, vec, f"vec_{vid}")

            del store1
            del logger1

            # Phase 2: Reopen and verify retrieval
            logger2 = vector_cluster_store_py.Logger(temp_log_path + ".2")
            store2 = vector_cluster_store_py.VectorClusterStore(logger2)
            assert store2.initialize(store_path, "kmeans", 768, 10)

            for vid, original_vec in vectors.items():
                retrieved = store2.retrieve_vector(vid)
                assert len(retrieved) == 768, f"Vector {vid} should be retrievable"
                # Check values match
                for j, (orig, ret) in enumerate(zip(original_vec, retrieved)):
                    assert abs(orig - ret) < 1e-5, f"Vector {vid} value mismatch at index {j}"

            del store2
            del logger2

        finally:
            if os.path.exists(store_path):
                os.unlink(store_path)
            if os.path.exists(temp_log_path + ".2"):
                os.unlink(temp_log_path + ".2")

    def test_multiple_clusters_persistence(self, temp_log_path):
        """
        Test that multiple clusters are correctly serialized and deserialized.

        This is a regression test for the ClusterInfo serialization bug where
        the first ClusterInfo consumed all remaining bytes during deserialization,
        causing "cannot create std::vector larger than max_size()" errors.
        """
        import tempfile
        import os
        import vector_cluster_store_py

        fd, store_path = tempfile.mkstemp(suffix='.bin', prefix='test_multi_cluster_')
        os.close(fd)

        try:
            # Create vectors spread across different regions of the vector space
            # to encourage assignment to different clusters
            num_vectors = 50
            vectors = {}
            for i in range(num_vectors):
                # Create vectors with different "centers" to spread across clusters
                center = (i % 10) * 0.5  # 10 different centers
                vec = np.random.normal(center, 0.1, 768)
                vec = (vec / np.linalg.norm(vec)).tolist()
                vectors[i] = vec

            # Phase 1: Create store with multiple clusters and add vectors
            logger1 = vector_cluster_store_py.Logger(temp_log_path)
            store1 = vector_cluster_store_py.VectorClusterStore(logger1)
            assert store1.initialize(store_path, "kmeans", 768, 10)

            for vid, vec in vectors.items():
                store1.store_vector(vid, vec, f"vec_{vid}")

            # Query before close to establish baseline
            query_vec = vectors[0]
            results_before = store1.find_similar_vectors(query_vec, 10)
            assert len(results_before) > 0, "Should find results before close"

            del store1
            del logger1

            # Phase 2: Reopen and verify all clusters were restored
            logger2 = vector_cluster_store_py.Logger(temp_log_path + ".2")
            store2 = vector_cluster_store_py.VectorClusterStore(logger2)

            # This is where the bug manifested - initialize would fail with
            # "cannot create std::vector larger than max_size()"
            assert store2.initialize(store_path, "kmeans", 768, 10), \
                "Store should initialize successfully after reopen"

            # Verify search works and returns similar results
            results_after = store2.find_similar_vectors(query_vec, 10)
            assert len(results_after) > 0, \
                "Should find results after reopen (multiple cluster persistence)"

            # The query vector (ID 0) should be findable after reopen
            ids_after = set(r[0] for r in results_after)
            assert 0 in ids_after, "Query vector should be found after reopen"

            # Verify we're getting reasonable results (vectors exist and have similarity scores)
            # Note: cosine similarity ranges from -1 to 1, with small floating-point tolerance
            eps = 1e-5
            for vid, score in results_after:
                assert 0 <= vid < num_vectors, f"Result ID {vid} should be valid"
                assert -1 - eps <= score <= 1 + eps, f"Similarity score {score} should be in [-1, 1]"

            del store2
            del logger2

        finally:
            if os.path.exists(store_path):
                os.unlink(store_path)
            if os.path.exists(temp_log_path + ".2"):
                os.unlink(temp_log_path + ".2")
