"""Unit tests for player priors tracking and feature extraction."""
import pytest
import numpy as np

from core.player_priors import (
    PlayerPriorsTracker,
    extract_player_priors_vector,
    TOTAL_PLAYER_PRIOR_DIM,
    PLAYER_PRIOR_FEATURE_COUNT,
)


def test_player_priors_tracker_defaults():
    tracker = PlayerPriorsTracker()
    default_p = tracker.get_default_priors()
    assert len(default_p) == PLAYER_PRIOR_FEATURE_COUNT
    assert default_p[0] == 0.50  # smooth_wr
    assert default_p[1] == 0.50  # smooth_cwr
    assert default_p[2] == 0.0   # champ_exp
    assert default_p[3] == 0.50  # rec_wr

    # Unknown player returns defaults
    assert tracker.get_player_priors("unknown_puuid", "Aatrox", "TOP") == default_p
    assert tracker.get_player_priors(None, "Aatrox", "TOP") == default_p


def test_player_priors_tracker_updates():
    tracker = PlayerPriorsTracker()
    puuid = "player_123"

    # Match 1: Win on Aatrox TOP
    tracker.update(puuid, "Aatrox", "TOP", win=1)
    priors1 = tracker.get_player_priors(puuid, "Aatrox", "TOP")
    # smooth_wr = (1 + 10*0.5) / (1 + 10) = 6 / 11 ~ 0.545
    assert priors1[0] > 0.50
    # smooth_cwr = (1 + 5*0.5) / (1 + 5) = 3.5 / 6 ~ 0.583
    assert priors1[1] > 0.50
    # champ_exp = log1p(1) ~ 0.693
    assert np.isclose(priors1[2], np.log1p(1))
    # role_fam = (1 + 1) / (1 + 5) = 2/6 ~ 0.333
    assert priors1[4] > 0.20

    # Querying a DIFFERENT champion on the same player
    priors_other = tracker.get_player_priors(puuid, "Ahri", "MIDDLE")
    # overall wr is still > 0.50
    assert priors_other[0] > 0.50
    # Ahri cwr is neutral (0 wins, 0 games -> 2.5 / 5 = 0.50)
    assert priors_other[1] == 0.50
    assert priors_other[2] == 0.0


def test_extract_player_priors_vector_dimension_and_leak_free():
    team_a = [
        {"puuid": f"t1_p{i}", "champion": "Shen", "role": "TOP"}
        for i in range(5)
    ]
    team_b = [
        {"puuid": f"t2_p{i}", "champion": "Zed", "role": "TOP"}
        for i in range(5)
    ]

    tracker = PlayerPriorsTracker()
    # Team A player 0 has 10 wins on Shen
    for _ in range(10):
        tracker.update("t1_p0", "Shen", "TOP", win=1)

    vec = extract_player_priors_vector(team_a, team_b, tracker=tracker)
    assert vec.shape == (TOTAL_PLAYER_PRIOR_DIM,)
    assert isinstance(vec, np.ndarray)

    # Team A mean champion winrate should be strictly higher than Team B
    # Team A champ wr is at index 1, Team B champ wr is at index 7, diff is at index 13
    assert vec[1] > vec[7]
    assert vec[13] > 0.0  # Team A - Team B diff > 0


def test_extract_player_priors_handles_empty_or_malformed():
    vec = extract_player_priors_vector([], [], tracker=None)
    assert vec.shape == (TOTAL_PLAYER_PRIOR_DIM,)
    # All diffs should be 0.0 with identical defaults
    assert np.all(vec[12:18] == 0.0)
