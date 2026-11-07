from pathlib import Path
import numpy as np
import pytest


def test_benchmark_refuses_existing_output_before_reading_sources(tmp_path):
    from scripts.benchmark_ordered_matchups import run
    from argparse import Namespace
    with pytest.raises(FileExistsError):
        run(Namespace(output=str(tmp_path), splits_cache='missing', manifest='missing', database='missing'))


def test_benchmark_matrix_drift_fails_before_scores_are_interpreted():
    from scripts.benchmark_ordered_matchups import verify_reference_features
    reference = {'variants': {'ordered': {'feature_sha256': {'train': 'abc'}}}}
    verify_reference_features(reference, 'ordered', {'train': 'abc'})
    with pytest.raises(ValueError, match='matrix'):
        verify_reference_features(reference, 'ordered', {'train': 'changed'})


def test_reference_prediction_comparison_requires_equal_labels_and_probabilities():
    from scripts.benchmark_ordered_matchups import compare_reference_predictions
    p = np.array([.2, .7])
    assert compare_reference_predictions(p, p, strict=True)['max_absolute_difference'] == 0.
    with pytest.raises(AssertionError):
        compare_reference_predictions(p, np.array([.2, .4]), strict=True)
