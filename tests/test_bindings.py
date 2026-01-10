"""
Tests for basic pybind11 bindings functionality.
"""
import pytest


class TestImports:
    """Test that all expected imports work."""

    def test_import_module(self):
        """Test that the native module can be imported."""
        import vector_cluster_store_py
        assert vector_cluster_store_py is not None

    def test_import_wrapper(self):
        """Test that the Python wrapper package can be imported."""
        import vector_store
        assert vector_store is not None
        assert hasattr(vector_store, 'VectorClusterStore')
        assert hasattr(vector_store, 'Logger')
        assert hasattr(vector_store, 'create_store')

    def test_version(self):
        """Test that version is defined."""
        import vector_store
        assert hasattr(vector_store, '__version__')
        assert vector_store.__version__ == '0.2.0'


class TestBasicObjects:
    """Test creation of basic objects."""

    def test_create_logger(self, temp_log_path):
        """Test Logger object creation."""
        import vector_cluster_store_py
        logger = vector_cluster_store_py.Logger(temp_log_path)
        assert logger is not None

    def test_create_store(self, temp_log_path):
        """Test VectorClusterStore object creation."""
        import vector_cluster_store_py
        logger = vector_cluster_store_py.Logger(temp_log_path)
        store = vector_cluster_store_py.VectorClusterStore(logger)
        assert store is not None

    def test_initialize_store(self, temp_store_path, temp_log_path):
        """Test store initialization."""
        import vector_cluster_store_py
        logger = vector_cluster_store_py.Logger(temp_log_path)
        store = vector_cluster_store_py.VectorClusterStore(logger)
        success = store.initialize(temp_store_path, "kmeans", 768, 10)
        assert success is True


class TestConvenienceFunction:
    """Test the create_store convenience function."""

    def test_create_store_function(self, temp_store_path):
        """Test that create_store returns an initialized store."""
        from vector_store import create_store
        store = create_store(temp_store_path, vector_dim=768, num_clusters=10)
        assert store is not None

    def test_create_store_invalid_path(self):
        """Test that create_store raises on invalid path."""
        from vector_store import create_store
        with pytest.raises(RuntimeError):
            create_store("/nonexistent/path/to/store.bin")
