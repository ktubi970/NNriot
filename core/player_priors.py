"""Player prior and point-in-time history tracking for League of Legends.

Provides Bayesian-smoothed player performance priors (overall winrate, champion
mastery, recent form, and role familiarity) strictly before match start time,
guaranteeing zero data leakage.
"""
from __future__ import annotations

from collections import defaultdict, deque
import math
from typing import Any, Sequence
import numpy as np

# Bayesian Laplace prior constants
DEFAULT_CHAMPION_PRIOR_WEIGHT: float = 5.0
DEFAULT_CHAMPION_PRIOR_MEAN: float = 0.50
DEFAULT_OVERALL_PRIOR_WEIGHT: float = 10.0
DEFAULT_OVERALL_PRIOR_MEAN: float = 0.50
DEFAULT_RECENT_WINDOW: int = 20

CANONICAL_ROLES: tuple[str, ...] = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
PLAYER_PRIOR_FEATURE_COUNT: int = 6  # per player: [smooth_wr, smooth_cwr, champ_exp, rec_wr, role_fam, tot_exp]
TEAM_AGGREGATE_COUNT: int = 18       # 6 team_a + 6 team_b + 6 team_diff
LANE_DIFF_COUNT: int = 15            # 5 lanes * 3 diffs (champ_wr, champ_exp, rec_wr)
TOTAL_PLAYER_PRIOR_DIM: int = TEAM_AGGREGATE_COUNT + LANE_DIFF_COUNT  # 33 features


class PlayerPriorsTracker:
    """In-memory chronological tracker maintaining point-in-time player statistics."""

    def __init__(
        self,
        champ_weight: float = DEFAULT_CHAMPION_PRIOR_WEIGHT,
        champ_mean: float = DEFAULT_CHAMPION_PRIOR_MEAN,
        overall_weight: float = DEFAULT_OVERALL_PRIOR_WEIGHT,
        overall_mean: float = DEFAULT_OVERALL_PRIOR_MEAN,
        recent_window: int = DEFAULT_RECENT_WINDOW,
    ) -> None:
        self.champ_weight = float(champ_weight)
        self.champ_mean = float(champ_mean)
        self.overall_weight = float(overall_weight)
        self.overall_mean = float(overall_mean)
        self.recent_window = int(recent_window)

        # puuid -> {"games": int, "wins": int}
        self.player_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"games": 0, "wins": 0})
        # puuid -> champ_name -> {"games": int, "wins": int}
        self.player_champs: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"games": 0, "wins": 0})
        )
        # puuid -> role_name -> int
        self.player_roles: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # puuid -> deque of recent match outcomes (0 or 1)
        self.player_recent: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=self.recent_window))

    def get_player_priors(self, puuid: str | None, champion: str | None, role: str | None) -> list[float]:
        """Compute point-in-time Bayesian smoothed features for a player.

        If puuid is None, empty, or has no prior games, returns neutral priors.
        """
        if not puuid or puuid not in self.player_totals:
            return self.get_default_priors()

        tot = self.player_totals[puuid]
        tot_g = tot["games"]
        tot_w = tot["wins"]

        if tot_g == 0:
            return self.get_default_priors()

        clean_champ = str(champion or "").strip()
        champ_stat = self.player_champs[puuid].get(clean_champ, {"games": 0, "wins": 0})
        c_g = champ_stat["games"]
        c_w = champ_stat["wins"]

        clean_role = str(role or "UNKNOWN").upper().strip()
        r_g = self.player_roles[puuid].get(clean_role, 0)
        rec = self.player_recent[puuid]

        smooth_wr = (tot_w + self.overall_weight * self.overall_mean) / (tot_g + self.overall_weight)
        smooth_cwr = (c_w + self.champ_weight * self.champ_mean) / (c_g + self.champ_weight)
        champ_exp = math.log1p(c_g)
        rec_wr = (sum(rec) + self.champ_weight * 0.5) / (len(rec) + self.champ_weight) if rec else 0.50
        role_fam = (r_g + 1.0) / (tot_g + 5.0)
        tot_exp = math.log1p(tot_g)

        return [float(smooth_wr), float(smooth_cwr), float(champ_exp), float(rec_wr), float(role_fam), float(tot_exp)]

    @staticmethod
    def get_default_priors() -> list[float]:
        """Return neutral Bayesian priors for unknown players or cold starts."""
        # [smooth_wr=0.5, smooth_cwr=0.5, champ_exp=0.0, rec_wr=0.5, role_fam=0.2, tot_exp=0.0]
        return [0.50, 0.50, 0.0, 0.50, 0.20, 0.0]

    def update(self, puuid: str | None, champion: str | None, role: str | None, win: int) -> None:
        """Update tracker state after a match concludes."""
        if not puuid:
            return
        w = 1 if win else 0
        tot = self.player_totals[puuid]
        tot["games"] += 1
        tot["wins"] += w

        clean_champ = str(champion or "").strip()
        if clean_champ:
            c = self.player_champs[puuid][clean_champ]
            c["games"] += 1
            c["wins"] += w

        clean_role = str(role or "UNKNOWN").upper().strip()
        self.player_roles[puuid][clean_role] += 1
        self.player_recent[puuid].append(w)


def extract_player_priors_vector(
    team_a_participants: Sequence[dict[str, Any]],
    team_b_participants: Sequence[dict[str, Any]],
    tracker: PlayerPriorsTracker | None = None,
) -> np.ndarray:
    """Extract a 33-dimensional domain vector representing team and lane player priors.

    Features layout:
    - 0..5: Team A mean player priors (smooth_wr, smooth_cwr, champ_exp, rec_wr, role_fam, tot_exp)
    - 6..11: Team B mean player priors
    - 12..17: Team diff (Team A - Team B)
    - 18..32: Lane matchup diffs for 5 canonical roles:
        - [TOP: cwr_diff, exp_diff, rec_diff]
        - [JUNGLE: cwr_diff, exp_diff, rec_diff]
        - [MIDDLE: cwr_diff, exp_diff, rec_diff]
        - [BOTTOM: cwr_diff, exp_diff, rec_diff]
        - [UTILITY: cwr_diff, exp_diff, rec_diff]
    """
    def _get_player_vector(p: dict[str, Any]) -> list[float]:
        # If precomputed priors dict is already attached to participant, use it
        if "priors" in p and isinstance(p["priors"], (list, tuple)) and len(p["priors"]) == PLAYER_PRIOR_FEATURE_COUNT:
            return [float(x) for x in p["priors"]]
        if tracker is not None:
            puuid = p.get("puuid")
            champ = p.get("champion") or p.get("championName") or p.get("champion_name")
            role = p.get("role") or p.get("teamPosition")
            return tracker.get_player_priors(puuid, champ, role)
        return PlayerPriorsTracker.get_default_priors()

    # Role indexing for lane matchups
    role_to_idx = {r: i for i, r in enumerate(CANONICAL_ROLES)}
    alias_roles = {"MID": "MIDDLE", "BOT": "BOTTOM", "SUPPORT": "UTILITY", "SUP": "UTILITY", "ADC": "BOTTOM"}

    def _role_of(p: dict[str, Any]) -> str:
        raw = str(p.get("role") or p.get("teamPosition") or "").upper().strip()
        return alias_roles.get(raw, raw)

    t1_vectors: list[list[float]] = []
    t2_vectors: list[list[float]] = []

    t1_by_role: dict[str, list[float]] = {}
    t2_by_role: dict[str, list[float]] = {}

    for p in team_a_participants:
        v = _get_player_vector(p)
        t1_vectors.append(v)
        r = _role_of(p)
        if r in role_to_idx and r not in t1_by_role:
            t1_by_role[r] = v

    for p in team_b_participants:
        v = _get_player_vector(p)
        t2_vectors.append(v)
        r = _role_of(p)
        if r in role_to_idx and r not in t2_by_role:
            t2_by_role[r] = v

    # Fallbacks if team has fewer or more than 5 participants
    default_v = PlayerPriorsTracker.get_default_priors()
    while len(t1_vectors) < 5:
        t1_vectors.append(default_v)
    while len(t2_vectors) < 5:
        t2_vectors.append(default_v)

    t1_arr = np.mean(t1_vectors[:5], axis=0)
    t2_arr = np.mean(t2_vectors[:5], axis=0)
    diff_arr = t1_arr - t2_arr

    features: list[float] = list(t1_arr) + list(t2_arr) + list(diff_arr)

    # 5 Lane matchup diffs
    for role in CANONICAL_ROLES:
        v1 = t1_by_role.get(role, default_v)
        v2 = t2_by_role.get(role, default_v)
        # v[1] is champ_wr, v[2] is champ_exp, v[3] is rec_wr
        features.extend([
            v1[1] - v2[1],  # champ winrate diff
            v1[2] - v2[2],  # champ exp diff
            v1[3] - v2[3],  # recent form diff
        ])

    return np.asarray(features, dtype=np.float32)
