import pytest

pytestmark = pytest.mark.unit


def test_patch_adjusted_region_difference_passes_all_gates():
    import numpy as np
    from analytics.regions import RegionalObservation, compare_regions

    rows = []
    rng = np.random.default_rng(3)
    for region, shift in (("EUW", 0.35), ("KR", 0.0)):
        for patch in ("16.13", "16.14"):
            for index in range(600):
                rows.append(
                    RegionalObservation(
                        region,
                        patch,
                        shift + rng.normal(0, 1),
                        index,
                        None,
                    )
                )

    result = compare_regions(rows, seed=11, resamples=2000)

    assert result.display_allowed is True
    assert result.q_value <= 0.05
    assert abs(result.standardized_contrast) >= 0.20


def test_region_ranking_is_hidden_when_one_region_is_small():
    from analytics.regions import RegionalObservation, compare_regions

    rows = [
        RegionalObservation("EUW", "16.14", 0.4, index, None)
        for index in range(600)
    ]
    rows += [
        RegionalObservation("KR", "16.14", 0.0, index, None)
        for index in range(499)
    ]

    result = compare_regions(rows, seed=11, resamples=500)

    assert result.display_allowed is False
    assert "INSUFFICIENT_MATCHES" in result.limitations


def test_region_ranking_is_hidden_when_recent_direction_reverses():
    import numpy as np
    from analytics.regions import RegionalObservation, compare_regions

    rows = []
    rng = np.random.default_rng(8)
    for region in ("EUW", "KR"):
        for index in range(600):
            if region == "EUW":
                shift = 0.70 if index < 300 else -0.10
            else:
                shift = 0.0
            rows.append(
                RegionalObservation(
                    region,
                    "16.14",
                    shift + rng.normal(0, 0.5),
                    index,
                    None,
                )
            )

    result = compare_regions(rows, seed=11, resamples=1000)

    assert result.display_allowed is False
    assert "TEMPORAL_DIRECTION_UNSTABLE" in result.limitations


def test_binary_region_comparison_requires_powered_outcomes():
    from analytics.regions import RegionalObservation, compare_regions

    rows = []
    for region, exposed_wins, comparison_wins in (
        ("EUW", 240, 150),
        ("KR", 180, 150),
    ):
        group_seen = {0: 0, 1: 0}
        for index in range(600):
            exposure = index % 2
            outcome = int(
                group_seen[exposure]
                < (exposed_wins if exposure else comparison_wins)
            )
            group_seen[exposure] += 1
            rows.append(
                RegionalObservation(
                    region,
                    "16.14",
                    exposure,
                    index,
                    outcome,
                )
            )

    result = compare_regions(rows, seed=4, resamples=500)

    assert result.is_binary is True
    assert result.display_allowed is True
    assert result.raw_contrast > 0.03
