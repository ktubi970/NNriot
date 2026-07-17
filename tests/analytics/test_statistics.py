import numpy as np
import pytest

from analytics.statistics import (
    binary_effect,
    benjamini_hochberg,
    continuous_effect,
)


def test_continuous_effect_recovers_known_paired_difference():
    differences = np.array([2.0, 3.0, 4.0, 5.0] * 150)

    result = continuous_effect(differences, seed=7, resamples=2000)

    assert result.value == pytest.approx(3.5)
    assert result.confidence_interval_95[0] > 0
    assert result.p_value <= 0.01
    assert result.effect_size > 0.2
    assert result.events_by_group == {"matches": 600}


def test_continuous_effect_is_seed_reproducible():
    values = np.linspace(-1.0, 2.0, 500)
    assert continuous_effect(values, seed=9, resamples=500) == continuous_effect(
        values, seed=9, resamples=500
    )


def test_continuous_null_effect_stays_descriptive():
    result = continuous_effect(np.zeros(600), seed=9, resamples=500)

    assert result.value == 0.0
    assert result.confidence_interval_95 == (0.0, 0.0)
    assert result.p_value == 1.0
    assert result.effect_size is None


def test_binary_effect_uses_one_match_once():
    exposure = np.array([1] * 300 + [0] * 300)
    outcome = np.array([1] * 210 + [0] * 90 + [1] * 150 + [0] * 150)

    result = binary_effect(exposure, outcome)

    assert result.value == pytest.approx(0.20)
    assert result.baseline == pytest.approx(0.50)
    assert result.events_by_group == {"exposed": 300, "comparison": 300}


def test_binary_effect_rejects_sparse_group():
    with pytest.raises(ValueError, match="SPARSE_COMPARISON_GROUP"):
        binary_effect(
            np.array([1] * 99 + [0] * 501),
            np.zeros(600),
            min_group=100,
        )


def test_binary_effect_rejects_non_binary_values_before_casting():
    with pytest.raises(ValueError, match="SOURCE_DATA_INVALID"):
        binary_effect(np.array([0.5] * 300 + [0] * 300), np.zeros(600))


def test_binary_effect_serializes_perfect_separation_without_infinity():
    result = binary_effect(
        np.array([1] * 300 + [0] * 300),
        np.array([1] * 300 + [0] * 300),
    )

    assert result.value == 1.0
    assert result.odds_ratio is None


def test_bh_is_monotone_in_original_order():
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx(
        [0.03, 0.04, 0.04]
    )


def test_bh_rejects_invalid_probability():
    with pytest.raises(ValueError, match="SOURCE_DATA_INVALID"):
        benjamini_hochberg([0.01, float("nan")])
