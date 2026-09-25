"""Shared training preparation stays lightweight and preserves row alignment."""
import subprocess
import sys

import numpy as np
from scipy.sparse import csr_matrix


def test_training_utilities_import_without_tensorflow():
    result = subprocess.run([sys.executable, "-c",
        "import sys; import core.training_utils; assert 'tensorflow' not in sys.modules"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_microbatches_preserve_feature_target_pairing_and_final_partial_batch():
    from core.training_utils import sparse_microbatch_generator

    x = csr_matrix(np.arange(10, dtype=np.float32).reshape(5, 2))
    targets = {"winner": np.arange(5), "total_kills": np.arange(5) + 30}
    batches = list(sparse_microbatch_generator(x, targets, batch_size=2))
    assert [len(batch[0]) for batch in batches] == [2, 2, 1]
    seen = []
    for values, labels in batches:
        np.testing.assert_array_equal(values[:, 0] / 2, labels["winner"])
        np.testing.assert_array_equal(labels["winner"] + 30, labels["total_kills"])
        seen.extend(labels["winner"].tolist())
    assert sorted(seen) == [0, 1, 2, 3, 4]
