"""
Tests for vector storage operations.
"""
import numpy as np


class TestVectorStorage:
    """Test storing and retrieving vectors."""

    def test_store_single_vector(self, initialized_store, sample_vector):
        """Test storing a single vector."""
        store, _ = initialized_store

        success = store.store_vector(1, sample_vector, "test metadata")
        assert success is True

    def test_retrieve_stored_vector(self, initialized_store, sample_vector):
        """Test retrieving a previously stored vector."""
        store, _ = initialized_store

        store.store_vector(1, sample_vector, "test metadata")

        retrieved = store.retrieve_vector(1)
        assert retrieved is not None
        assert len(retrieved) == 768

    def test_vector_values_preserved(self, initialized_store, sample_vector):
        """Test that vector values are preserved after store/retrieve."""
        store, _ = initialized_store

        store.store_vector(1, sample_vector, "test")
        retrieved = store.retrieve_vector(1)

        # Check values are close (floating point comparison)
        assert len(retrieved) == len(sample_vector)
        for orig, ret in zip(sample_vector, retrieved):
            assert abs(orig - ret) < 1e-5

    def test_retrieve_metadata(self, initialized_store, sample_vector):
        """Test retrieving vector metadata."""
        store, _ = initialized_store
        metadata = "test metadata for retrieval"

        store.store_vector(1, sample_vector, metadata)
        retrieved_metadata = store.get_vector_metadata(1)

        assert retrieved_metadata == metadata

    def test_store_multiple_vectors(self, initialized_store):
        """Test storing multiple vectors."""
        store, _ = initialized_store

        for i in range(10):
            vec = np.random.normal(0, 1, 768)
            vec = (vec / np.linalg.norm(vec)).tolist()
            success = store.store_vector(i, vec, f"vector_{i}")
            assert success is True, f"Failed to store vector {i}"

    def test_delete_vector(self, initialized_store, sample_vector):
        """Test deleting a vector."""
        store, _ = initialized_store

        store.store_vector(1, sample_vector, "to delete")
        success = store.delete_vector(1)

        assert success is True


class TestVectorRetrieval:
    """Test vector retrieval edge cases."""

    def test_retrieve_nonexistent_vector(self, initialized_store):
        """Test retrieving a vector that doesn't exist."""
        store, _ = initialized_store

        retrieved = store.retrieve_vector(9999)
        # Should return empty vector for non-existent ID
        assert len(retrieved) == 0

    def test_retrieve_after_delete(self, initialized_store, sample_vector):
        """Test that deleted vectors cannot be retrieved."""
        store, _ = initialized_store

        store.store_vector(1, sample_vector, "test")
        store.delete_vector(1)
        retrieved = store.retrieve_vector(1)

        assert len(retrieved) == 0


class TestMaintenance:
    """Test store maintenance operations."""

    def test_perform_maintenance(self, initialized_store):
        """Test that maintenance can be performed."""
        store, _ = initialized_store

        # Add some vectors first
        for i in range(5):
            vec = np.random.normal(0, 1, 768)
            vec = (vec / np.linalg.norm(vec)).tolist()
            store.store_vector(i, vec, f"vec_{i}")

        # Perform maintenance (should not raise)
        store.perform_maintenance()
