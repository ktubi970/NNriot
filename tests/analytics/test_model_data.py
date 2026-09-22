import pytest


@pytest.mark.integration
def test_matrix_is_one_blue_minus_red_row_per_match(powered_analytics_db):
    import numpy as np
    from analytics.model_data import MODEL_FEATURES, load_model_dataset

    dataset = load_model_dataset(str(powered_analytics_db))

    assert dataset.x.shape == (dataset.match_ids.shape[0], len(MODEL_FEATURES))
    assert dataset.x.shape[0] == 2100
    assert set(np.unique(dataset.y)) <= {0, 1}
    assert "win" not in dataset.feature_names
    assert "winner_team_id" not in dataset.feature_names
    assert all(name.endswith("_delta") for name in dataset.feature_names)
    assert all(len(puuids) == 10 for puuids in dataset.match_puuids)


@pytest.mark.unit
def test_swapping_sides_negates_features_and_target():
    import numpy as np
    from analytics.model_data import MODEL_FEATURES, rows_to_matrix

    blue = {name: float(index + 1) for index, name in enumerate(MODEL_FEATURES)}
    red = {name: 0.0 for name in MODEL_FEATURES}

    forward = rows_to_matrix([("m1", blue, red, 1, "EUW", "16.14", 1)])
    swapped = rows_to_matrix([("m1", red, blue, 0, "EUW", "16.14", 1)])

    assert np.allclose(forward.x, -swapped.x)
    assert forward.y[0] == 1 - swapped.y[0]


@pytest.mark.unit
def test_forbidden_feature_fails_before_fit():
    from analytics.model_data import assert_allowed_features

    with pytest.raises(ValueError, match="TARGET_FORBIDDEN"):
        assert_allowed_features(("gold_delta", "win"))


@pytest.mark.unit
def test_unknown_feature_fails_before_fit():
    from analytics.model_data import assert_allowed_features

    with pytest.raises(ValueError, match="UNREGISTERED_MODEL_FEATURE"):
        assert_allowed_features(("gold_delta", "secret_signal_delta"))
