"""
Tests for fastcomp module.
"""
import pytest
import numpy as np

from vector_store.fastcomp import cosine_distance, euclidean_distance


class TestDistanceFunctions:
    """Test distance calculation functions."""

    def test_cosine_distance_identical(self):
        """Identical vectors should have distance 0."""
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-6)

    def test_cosine_distance_opposite(self):
        """Opposite vectors should have distance 2."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        assert cosine_distance(v1, v2) == pytest.approx(2.0, abs=1e-6)

    def test_cosine_distance_orthogonal(self):
        """Orthogonal vectors should have distance 1."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        assert cosine_distance(v1, v2) == pytest.approx(1.0, abs=1e-6)

    def test_cosine_distance_similar(self):
        """Similar vectors should have small distance."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([1.1, 2.1, 3.1])
        dist = cosine_distance(v1, v2)
        assert 0 < dist < 0.01

    def test_cosine_distance_zero_vector(self):
        """Zero vector should return max distance."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([0.0, 0.0, 0.0])
        assert cosine_distance(v1, v2) == 1.0

    def test_euclidean_distance_identical(self):
        """Identical vectors should have distance 0."""
        v = np.array([1.0, 2.0, 3.0])
        assert euclidean_distance(v, v) == pytest.approx(0.0, abs=1e-6)

    def test_euclidean_distance_known(self):
        """Test with known distance."""
        v1 = np.array([0.0, 0.0])
        v2 = np.array([3.0, 4.0])
        assert euclidean_distance(v1, v2) == pytest.approx(5.0, abs=1e-6)

    def test_euclidean_distance_unit_vectors(self):
        """Unit vectors along axes."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        assert euclidean_distance(v1, v2) == pytest.approx(np.sqrt(2), abs=1e-6)


class TestModuleImports:
    """Test module imports."""

    def test_import_fastcomp(self):
        """Test that fastcomp module can be imported."""
        from vector_store import fastcomp
        assert fastcomp is not None

    def test_import_functions(self):
        """Test that main functions are importable."""
        from vector_store.fastcomp import (
            get_embedding,
            cosine_distance,
            euclidean_distance,
            compare_texts,
            main
        )
        assert callable(get_embedding)
        assert callable(cosine_distance)
        assert callable(euclidean_distance)
        assert callable(compare_texts)
        assert callable(main)


class TestCompareTexts:
    """Test compare_texts function edge cases."""

    def test_compare_texts_too_few(self):
        """Should return None with fewer than 2 texts."""
        from vector_store.fastcomp import compare_texts
        result = compare_texts(["single text"])
        assert result is None

    def test_compare_texts_empty(self):
        """Should return None with empty list."""
        from vector_store.fastcomp import compare_texts
        result = compare_texts([])
        assert result is None
