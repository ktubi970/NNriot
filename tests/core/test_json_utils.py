import pytest

pytestmark = pytest.mark.unit


from tests.fixtures_data import get_extended_test_data


def _non_zero_values(vector):
    from scipy import sparse

    return vector.data if sparse.issparse(vector) else vector[vector != 0]


def test_enhanced_vectorization_preserves_batch_shape_and_finite_values():
    import numpy as np
    from core import json_utils

    matches = get_extended_test_data()
    vectors = json_utils.enhanced_json_to_vector(matches, dim=512)

    assert vectors.shape == (len(matches), 512)
    assert np.count_nonzero(_non_zero_values(vectors)) > 0
    assert np.isfinite(_non_zero_values(vectors)).all()

    # enhanced_json_to_vector normalizes each non-empty row to unit L2 norm.
    row_norms = np.sqrt(np.asarray(vectors.multiply(vectors).sum(axis=1)).ravel())
    np.testing.assert_allclose(row_norms, 1.0, rtol=1e-6, atol=1e-7)


def test_sparse_vectorization_returns_non_empty_csr_matrix():
    from scipy import sparse
    from core import json_utils

    matches = get_extended_test_data()
    vectors = json_utils.create_sparse_vector(matches, dim=512, sparsity=0.02)

    assert sparse.isspmatrix_csr(vectors)
    assert vectors.shape == (len(matches), 512)
    assert 0 < vectors.nnz < vectors.shape[0] * vectors.shape[1]
