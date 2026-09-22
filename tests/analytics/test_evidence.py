def test_moderate_requires_all_statistical_and_practical_gates():
    from analytics.evidence import classify_recommendation, stable_hash

    result = classify_recommendation(
        matches=800,
        players=300,
        group_counts=(300, 500),
        ci=(0.02, 0.08),
        q_value=0.01,
        effect_size=0.25,
        win_rate_difference=None,
        temporal_directions=(1, 1, -1),
    )

    assert result == {
        "allowed": True,
        "strength": "moderate",
        "wording_key": "association_moderate",
    }


def test_strong_requires_all_available_temporal_folds():
    from analytics.evidence import classify_recommendation, stable_hash

    result = classify_recommendation(
        matches=2500,
        players=900,
        group_counts=(1200, 1300),
        ci=(0.51, 0.70),
        q_value=0.005,
        effect_size=0.60,
        win_rate_difference=None,
        temporal_directions=(1, 1, -1),
    )

    assert result["strength"] == "moderate"


def test_ci_touching_null_abstains():
    from analytics.evidence import classify_recommendation, stable_hash

    result = classify_recommendation(
        matches=2500,
        players=900,
        group_counts=(1200, 1300),
        ci=(0.0, 0.70),
        q_value=0.001,
        effect_size=0.60,
        win_rate_difference=None,
        temporal_directions=(1, 1, 1),
    )

    assert result["allowed"] is False


def test_strong_supports_a_consistent_negative_direction():
    from analytics.evidence import classify_recommendation, stable_hash

    result = classify_recommendation(
        matches=2500,
        players=900,
        group_counts=(1200, 1300),
        ci=(-0.70, -0.51),
        q_value=0.005,
        effect_size=-0.60,
        win_rate_difference=None,
        temporal_directions=(-1, -1, -1),
    )

    assert result["strength"] == "strong"


def test_temporal_majority_must_match_overall_effect_direction():
    from analytics.evidence import classify_recommendation, stable_hash

    result = classify_recommendation(
        matches=2500,
        players=900,
        group_counts=(1200, 1300),
        ci=(0.51, 0.70),
        q_value=0.005,
        effect_size=0.60,
        win_rate_difference=None,
        temporal_directions=(-1, -1, 1),
    )

    assert result["allowed"] is False


def test_all_inverse_folds_cannot_make_evidence_strong():
    from analytics.evidence import classify_recommendation, stable_hash

    result = classify_recommendation(
        matches=2500,
        players=900,
        group_counts=(1200, 1300),
        ci=(0.51, 0.70),
        q_value=0.005,
        effect_size=0.60,
        win_rate_difference=None,
        temporal_directions=(-1, -1, -1),
    )

    assert result["allowed"] is False


def test_hash_is_canonical_across_dictionary_order():
    from analytics.evidence import classify_recommendation, stable_hash

    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
