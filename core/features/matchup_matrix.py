"""Empirical Lane Counter Matchup and Patch Meta Tracker.

Maintains leak-free, point-in-time champion winrates and 1v1 lane matchup deltas:
- Lane Matchup Delta: WR(Champ_A | Champ_B) - 0.5 in [-0.25, +0.25]
- Mathematical anti-symmetry guarantee:
    Delta_WR(B | A) == -Delta_WR(A | B)
- Bayesian m-estimate smoothing to prevent small sample overfitting:
    WR = (wins + m * 0.5) / (total + m)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence
import numpy as np

from core.champion_registry import UNK_CHAMPION_ID, champion_to_id

DEFAULT_MATCHUP_WEIGHT: float = 15.0  # Equivalent games prior weight for 1v1 matchup
DEFAULT_GLOBAL_WEIGHT: float = 50.0   # Equivalent games prior weight for overall champion WR


class LaneMatchupTracker:
    """In-memory streaming tracker for point-in-time champion and lane matchup statistics."""

    def __init__(
        self,
        matchup_weight: float = DEFAULT_MATCHUP_WEIGHT,
        global_weight: float = DEFAULT_GLOBAL_WEIGHT,
    ) -> None:
        self.matchup_weight = float(matchup_weight)
        self.global_weight = float(global_weight)

        # (role, min_cid, max_cid) -> {"games": int, "wins_low_cid": int}
        # Storing min_cid/max_cid guarantees identical shared record and perfect anti-symmetry
        self.matchups: dict[tuple[str, int, int], dict[str, int]] = defaultdict(
            lambda: {"games": 0, "wins_low_cid": 0}
        )
        # cid -> {"games": int, "wins": int}
        self.champ_globals: dict[int, dict[str, int]] = defaultdict(
            lambda: {"games": 0, "wins": 0}
        )

    def get_matchup_delta(self, role: str, cid_a: int, cid_b: int) -> float:
        """Compute Bayesian-smoothed lane winrate advantage for champ A over champ B.

        Returns float in [-0.25, 0.25].
        Guarantees get_matchup_delta(role, B, A) == -get_matchup_delta(role, A, B).
        """
        if cid_a == UNK_CHAMPION_ID or cid_b == UNK_CHAMPION_ID or cid_a == cid_b:
            return 0.0

        role_clean = str(role or "UNKNOWN").upper().strip()
        low_cid, high_cid = (cid_a, cid_b) if cid_a < cid_b else (cid_b, cid_a)
        key = (role_clean, low_cid, high_cid)

        stat = self.matchups.get(key)
        if not stat or stat["games"] == 0:
            # Fall back to champion baseline global difference
            wr_a = self.get_champion_winrate(cid_a)
            wr_b = self.get_champion_winrate(cid_b)
            return float(np.clip(wr_a - wr_b, -0.20, 0.20))

        games = stat["games"]
        wins_low = stat["wins_low_cid"]

        # Bayesian m-estimate of P(low_cid wins)
        p_low = (wins_low + self.matchup_weight * 0.50) / (games + self.matchup_weight)
        delta_low = p_low - 0.50

        return float(np.clip(delta_low if cid_a == low_cid else -delta_low, -0.25, 0.25))

    def get_champion_winrate(self, cid: int) -> float:
        """Return Bayesian smoothed global winrate for a champion [0.35, 0.65]."""
        if cid == UNK_CHAMPION_ID:
            return 0.50

        stat = self.champ_globals.get(cid)
        if not stat or stat["games"] == 0:
            return 0.50

        games = stat["games"]
        wins = stat["wins"]
        smooth_wr = (wins + self.global_weight * 0.50) / (games + self.global_weight)
        return float(np.clip(smooth_wr, 0.35, 0.65))

    def update_match(
        self,
        team_a_participants: Sequence[dict[str, Any]],
        team_b_participants: Sequence[dict[str, Any]],
        winner: int,  # 0 for Team A win, 1 for Team B win
    ) -> None:
        """Update tracker with concluded match result (0=Team A win, 1=Team B win)."""
        is_a_win = 1 if winner == 0 else 0
        is_b_win = 1 - is_a_win

        # 1. Update individual champion global winrates
        for p in team_a_participants:
            cid = champion_to_id(p.get("champion") or p.get("championName"))
            if cid != UNK_CHAMPION_ID:
                stat = self.champ_globals[cid]
                stat["games"] += 1
                stat["wins"] += is_a_win

        for p in team_b_participants:
            cid = champion_to_id(p.get("champion") or p.get("championName"))
            if cid != UNK_CHAMPION_ID:
                stat = self.champ_globals[cid]
                stat["games"] += 1
                stat["wins"] += is_b_win

        # 2. Update direct 1v1 lane matchups
        def _get_role_map(participants):
            res = {}
            for p in participants:
                r = str(p.get("role") or p.get("teamPosition") or "").upper().strip()
                cid = champion_to_id(p.get("champion") or p.get("championName"))
                if r and cid != UNK_CHAMPION_ID:
                    res[r] = cid
            return res

        a_roles = _get_role_map(team_a_participants)
        b_roles = _get_role_map(team_b_participants)

        for role, cid_a in a_roles.items():
            if role in b_roles:
                cid_b = b_roles[role]
                if cid_a != cid_b:
                    low_cid, high_cid = (cid_a, cid_b) if cid_a < cid_b else (cid_b, cid_a)
                    key = (role, low_cid, high_cid)
                    m_stat = self.matchups[key]
                    m_stat["games"] += 1
                    # Did the player with the lower cid win?
                    if (cid_a == low_cid and is_a_win == 1) or (cid_b == low_cid and is_b_win == 1):
                        m_stat["wins_low_cid"] += 1


_GLOBAL_TRACKER: LaneMatchupTracker | None = None


def get_lane_matchup_tracker() -> LaneMatchupTracker:
    """Return shared global singleton matchup tracker."""
    global _GLOBAL_TRACKER
    if _GLOBAL_TRACKER is None:
        _GLOBAL_TRACKER = LaneMatchupTracker()
    return _GLOBAL_TRACKER
