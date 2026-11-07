import numpy as np
import pytest
from core.champion_registry import NUM_CHAMPIONS, UNK_CHAMPION_ID, champion_to_id
from core.draft_features import (
    ROLE_NAMES,
    TABULAR_FEATURE_DIM,
    DENSE_FEATURE_DIM,
    TABULAR_FEATURE_NAMES,
    get_tabular_feature_names,
    extract_tabular_draft_features,
    extract_dense_draft_features,
    batch_extract_draft_features,
)


@pytest.fixture
def sample_match_features():
    return {
        "team_a": [
            {"champion": "Aatrox", "role": "TOP"},
            {"champion": "LeeSin", "role": "JUNGLE"},
            {"champion": "Ahri", "role": "MIDDLE"},
            {"champion": "Jinx", "role": "BOTTOM"},
            {"champion": "Thresh", "role": "UTILITY"},
        ],
        "team_b": [
            {"champion": "Darius", "role": "TOP"},
            {"champion": "Elise", "role": "JUNGLE"},
            {"champion": "Zed", "role": "MIDDLE"},
            {"champion": "Ezreal", "role": "BOTTOM"},
            {"champion": "Lulu", "role": "UTILITY"},
        ],
        "game_mode": "CLASSIC",
        "game_version": "14.1.1",
    }


def test_tabular_feature_names_constant():
    names = get_tabular_feature_names()
    assert len(names) == TABULAR_FEATURE_DIM
    assert isinstance(TABULAR_FEATURE_NAMES, tuple)
    assert names == list(TABULAR_FEATURE_NAMES)
    assert names[0] == "team_a_TOP"
    assert names[4] == "team_a_UTILITY"
    assert names[5] == "team_b_TOP"
    assert names[9] == "team_b_UTILITY"
    assert names[10] == "comp_diff_champ_0"
    assert names[-1] == f"comp_diff_champ_{NUM_CHAMPIONS - 1}"


def test_extract_tabular_draft_features_shape(sample_match_features):
    vec, names = extract_tabular_draft_features(sample_match_features)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (TABULAR_FEATURE_DIM,)
    assert len(names) == TABULAR_FEATURE_DIM
    # First 5 are team_a roles, next 5 are team_b roles
    assert names[0] == "team_a_TOP"
    assert names[4] == "team_a_UTILITY"
    assert names[5] == "team_b_TOP"
    assert names[9] == "team_b_UTILITY"
    # Verification of values
    assert vec[0] == champion_to_id("Aatrox")
    assert vec[2] == champion_to_id("Ahri")
    assert vec[5] == champion_to_id("Darius")


def test_composition_diff_properties(sample_match_features):
    vec, _ = extract_tabular_draft_features(sample_match_features)
    comp_diff = vec[10:]
    assert len(comp_diff) == NUM_CHAMPIONS
    # Team B has Darius (+1), Team A has Aatrox (-1)
    darius_id = champion_to_id("Darius")
    aatrox_id = champion_to_id("Aatrox")
    assert comp_diff[darius_id] == 1.0
    assert comp_diff[aatrox_id] == -1.0
    # Sum of comp_diff should be 0 because both teams have 5 champions
    assert np.sum(comp_diff) == 0.0


def test_extract_dense_draft_features_shape(sample_match_features):
    dense = extract_dense_draft_features(sample_match_features)
    assert isinstance(dense, np.ndarray)
    assert dense.shape == (DENSE_FEATURE_DIM,)
    assert dense.dtype == np.float32

    # Verify one-hot activations
    aatrox_id = champion_to_id("Aatrox")
    darius_id = champion_to_id("Darius")
    # Team A TOP is slot 0
    assert dense[aatrox_id] == 1.0
    # Team B TOP is slot 0 of team B (offset 5 * NUM_CHAMPIONS)
    assert dense[5 * NUM_CHAMPIONS + darius_id] == 1.0
    # Comp diff segment at 10 * NUM_CHAMPIONS
    diff_offset = 10 * NUM_CHAMPIONS
    assert dense[diff_offset + darius_id] == 1.0
    assert dense[diff_offset + aatrox_id] == -1.0


def test_batch_extract_draft_features(sample_match_features):
    batch = [sample_match_features, sample_match_features]
    tabular_batch = batch_extract_draft_features(batch, mode="tabular")
    assert tabular_batch.shape == (2, TABULAR_FEATURE_DIM)

    # Numerical equality verification against single extraction
    single_tabular, _ = extract_tabular_draft_features(sample_match_features)
    np.testing.assert_array_equal(tabular_batch[0], single_tabular)
    np.testing.assert_array_equal(tabular_batch[1], single_tabular)

    dense_batch = batch_extract_draft_features(batch, mode="dense")
    assert dense_batch.shape == (2, DENSE_FEATURE_DIM)

    # Numerical equality verification against single extraction
    single_dense = extract_dense_draft_features(sample_match_features)
    np.testing.assert_array_equal(dense_batch[0], single_dense)
    np.testing.assert_array_equal(dense_batch[1], single_dense)


def test_batch_extract_invalid_mode(sample_match_features):
    with pytest.raises(ValueError, match="Unknown mode"):
        batch_extract_draft_features([sample_match_features], mode="invalid")


def test_missing_or_misordered_roles():
    # Test fallback when roles are unassigned or duplicated
    features = {
        "team_a": [
            {"champion": "Aatrox", "role": "UNKNOWN"},
            {"champion": "LeeSin", "role": "UNKNOWN"},
            {"champion": "Ahri", "role": "MIDDLE"},
            {"champion": "Jinx", "role": "BOTTOM"},
            {"champion": "Thresh", "role": "UTILITY"},
        ],
        "team_b": [
            {"champion": "Darius", "role": "TOP"},
            {"champion": "Elise", "role": "JUNGLE"},
            {"champion": "Zed", "role": "MIDDLE"},
            {"champion": "Ezreal", "role": "BOTTOM"},
            {"champion": "Lulu", "role": "UTILITY"},
        ],
    }
    vec, _ = extract_tabular_draft_features(features)
    assert vec.shape == (TABULAR_FEATURE_DIM,)
    # Ahri (MIDDLE), Jinx (BOTTOM), Thresh (UTILITY) are in slots 2, 3, 4
    assert vec[2] == champion_to_id("Ahri")
    assert vec[3] == champion_to_id("Jinx")
    assert vec[4] == champion_to_id("Thresh")
    # Slots 0 (TOP) and 1 (JUNGLE) filled by unassigned players Aatrox and LeeSin
    assert vec[0] == champion_to_id("Aatrox")
    assert vec[1] == champion_to_id("LeeSin")


def test_alternate_keys_and_empty_teams():
    features = {
        "team_a": [
            {"championName": "Aatrox", "teamPosition": "TOP"},
            {"champion_name": "LeeSin", "teamPosition": "JUNGLE"},
        ],
        "team_b": [],
    }
    vec, names = extract_tabular_draft_features(features)
    assert vec.shape == (TABULAR_FEATURE_DIM,)
    assert vec[0] == champion_to_id("Aatrox")
    assert vec[1] == champion_to_id("LeeSin")
    assert vec[2] == UNK_CHAMPION_ID
    # Empty team B has UNK_CHAMPION_ID for all slots
    assert all(vec[5:10] == UNK_CHAMPION_ID)


def test_non_list_participants():
    features = {"team_a": None, "team_b": "invalid_string"}
    vec, names = extract_tabular_draft_features(features)
    assert vec.shape == (TABULAR_FEATURE_DIM,)
    assert np.all(vec[:10] == UNK_CHAMPION_ID)
    assert np.all(vec[10:] == 0.0)
