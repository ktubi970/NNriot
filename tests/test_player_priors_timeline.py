"""Unit tests for early-game timeline Bayesian priors in PlayerPriorsTracker."""
import numpy as np
import pytest

from core.player_priors import (
    EXTENDED_PLAYER_PRIOR_FEATURE_COUNT,
    PLAYER_PRIOR_FEATURE_COUNT,
    PlayerPriorsTracker,
)


def test_player_priors_timeline_cold_start():
    tracker = PlayerPriorsTracker()

    # Unknown player with extended=True and include_timeline=True
    priors = tracker.get_player_priors("unknown_puuid", "Aatrox", "TOP", extended=True, include_timeline=True)
    assert len(priors) == 14
    # The first 10 are base + extended priors
    assert priors[:10] == tracker.get_default_priors(extended=True, include_timeline=False)
    # The last 4 are [norm_gd15, norm_csd15, norm_xpd15, norm_fbi10]
    assert priors[10] == 0.0   # Neutral GD15
    assert priors[11] == 0.0   # Neutral CSD15
    assert priors[12] == 0.0   # Neutral XPD15
    assert priors[13] == 0.10  # Default 10% base rate for FB involvement

    # None puuid
    none_priors = tracker.get_player_priors(None, "Aatrox", "TOP", extended=True, include_timeline=True)
    assert none_priors == priors


def test_player_priors_timeline_default_priors_variants():
    # Base legacy: 6 features
    p_base = PlayerPriorsTracker.get_default_priors(extended=False, include_timeline=False)
    assert len(p_base) == PLAYER_PRIOR_FEATURE_COUNT
    assert p_base == [0.50, 0.50, 0.0, 0.50, 0.20, 0.0]

    # Extended legacy: 10 features
    p_ext = PlayerPriorsTracker.get_default_priors(extended=True, include_timeline=False)
    assert len(p_ext) == EXTENDED_PLAYER_PRIOR_FEATURE_COUNT
    assert p_ext == [0.50, 0.50, 0.0, 0.50, 0.20, 0.0, 0.50, 0.60, 0.70, 0.35]

    # Base + timeline: 10 features (6 base + 4 timeline)
    p_base_tl = PlayerPriorsTracker.get_default_priors(extended=False, include_timeline=True)
    assert len(p_base_tl) == 10
    assert p_base_tl[:6] == p_base
    assert p_base_tl[6:] == [0.0, 0.0, 0.0, 0.10]

    # Extended + timeline: 14 features (10 base + 4 timeline)
    p_ext_tl = PlayerPriorsTracker.get_default_priors(extended=True, include_timeline=True)
    assert len(p_ext_tl) == 14
    assert p_ext_tl[:10] == p_ext
    assert p_ext_tl[10:] == [0.0, 0.0, 0.0, 0.10]


def test_player_priors_timeline_update_and_smoothing():
    tracker = PlayerPriorsTracker()
    puuid = "test_player_1"

    # Record 5 strong games: avg GD15 = +1000, CSD15 = +20, XPD15 = +600, FBI10 = 1.0
    for _ in range(5):
        tracker.record_timeline_metrics(
            puuid=puuid,
            gd15=1000.0,
            csd15=20.0,
            xpd15=600.0,
            fbi10=1.0,
        )

    # Bayesian Laplace prior with weight W=5:
    # smooth_gd15 = (5 * 1000 + 5 * 0) / (5 + 5) = 500.0
    # norm_gd15 = 500 / 2000 = 0.25
    # smooth_csd15 = (5 * 20 + 5 * 0) / 10 = 10.0 -> norm = 10 / 30 = 0.3333
    # smooth_xpd15 = (5 * 600 + 5 * 0) / 10 = 300.0 -> norm = 300 / 1500 = 0.20
    # smooth_fbi10 = (5 * 1.0 + 5 * 0.1) / 10 = 0.55 -> norm = 0.55
    priors = tracker.get_player_priors(puuid, "Darius", "TOP", extended=True, include_timeline=True)
    assert len(priors) == 14
    assert pytest.approx(priors[10], rel=1e-2) == 0.25
    assert pytest.approx(priors[11], rel=1e-2) == 0.3333
    assert pytest.approx(priors[12], rel=1e-2) == 0.20
    assert pytest.approx(priors[13], rel=1e-2) == 0.55


def test_player_priors_timeline_negative_values():
    tracker = PlayerPriorsTracker()
    puuid = "test_player_losing_lane"

    # Record 5 losing lane games: avg GD15 = -1000, CSD15 = -20, XPD15 = -600, FBI10 = 0.0
    for _ in range(5):
        tracker.record_timeline_metrics(
            puuid=puuid,
            gd15=-1000.0,
            csd15=-20.0,
            xpd15=-600.0,
            fbi10=0.0,
        )

    # smooth_gd15 = (5 * -1000 + 5 * 0) / 10 = -500.0 -> norm = -500 / 2000 = -0.25
    # smooth_csd15 = (5 * -20 + 5 * 0) / 10 = -10.0 -> norm = -10 / 30 = -0.3333
    # smooth_xpd15 = (5 * -600 + 5 * 0) / 10 = -300.0 -> norm = -300 / 1500 = -0.20
    # smooth_fbi10 = (5 * 0.0 + 5 * 0.1) / 10 = 0.05 -> norm = 0.05
    priors = tracker.get_player_priors(puuid, "Aatrox", "TOP", extended=True, include_timeline=True)
    assert len(priors) == 14
    assert pytest.approx(priors[10], rel=1e-2) == -0.25
    assert pytest.approx(priors[11], rel=1e-2) == -0.3333
    assert pytest.approx(priors[12], rel=1e-2) == -0.20
    assert pytest.approx(priors[13], rel=1e-2) == 0.05


def test_player_priors_timeline_clipping_bounds():
    tracker = PlayerPriorsTracker()
    puuid_high = "extreme_high"
    puuid_low = "extreme_low"

    # Record 100 extreme high games
    for _ in range(100):
        tracker.record_timeline_metrics(puuid_high, gd15=10000.0, csd15=100.0, xpd15=5000.0, fbi10=2.0)

    priors_high = tracker.get_player_priors(puuid_high, "Darius", "TOP", extended=True, include_timeline=True)
    assert priors_high[10] == 1.0   # clipped at 1.0
    assert priors_high[11] == 1.0   # clipped at 1.0
    assert priors_high[12] == 1.0   # clipped at 1.0
    assert priors_high[13] == 1.0   # clipped at 1.0

    # Record 100 extreme low games
    for _ in range(100):
        tracker.record_timeline_metrics(puuid_low, gd15=-10000.0, csd15=-100.0, xpd15=-5000.0, fbi10=-1.0)

    priors_low = tracker.get_player_priors(puuid_low, "Darius", "TOP", extended=True, include_timeline=True)
    assert priors_low[10] == -1.0  # clipped at -1.0
    assert priors_low[11] == -1.0  # clipped at -1.0
    assert priors_low[12] == -1.0  # clipped at -1.0
    assert priors_low[13] == 0.0   # clipped at 0.0


def test_player_priors_timeline_backward_compatibility():
    tracker = PlayerPriorsTracker()
    puuid = "compat_player"

    tracker.update(puuid, "Aatrox", "TOP", win=1, kills=5, deaths=2, assists=5, total_cs=150, gold_earned=8000, game_duration=1500)
    tracker.record_timeline_metrics(puuid, gd15=500.0, csd15=10.0, xpd15=200.0, fbi10=1.0)

    # include_timeline defaults to False
    priors_base = tracker.get_player_priors(puuid, "Aatrox", "TOP")
    assert len(priors_base) == 6

    priors_ext = tracker.get_player_priors(puuid, "Aatrox", "TOP", extended=True)
    assert len(priors_ext) == 10

    # Explicit include_timeline=False
    assert tracker.get_player_priors(puuid, "Aatrox", "TOP", extended=True, include_timeline=False) == priors_ext


def test_player_priors_timeline_graceful_empty_puuid():
    tracker = PlayerPriorsTracker()

    # None and empty string puuid in record_timeline_metrics must not crash
    tracker.record_timeline_metrics(None, gd15=500.0, csd15=10.0, xpd15=200.0, fbi10=1.0)
    tracker.record_timeline_metrics("", gd15=500.0, csd15=10.0, xpd15=200.0, fbi10=1.0)

    # Empty tracker must still be empty
    assert len(tracker.player_timeline_totals) == 0
