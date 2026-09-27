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


EXTENDED_PLAYER_PRIOR_FEATURE_COUNT: int = 10  # 6 legacy + kda, cspm, gpm, aggression

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

        # puuid -> dict of performance and combat aggregates
        self.player_totals: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "games": 0, "wins": 0,
                "kills": 0, "deaths": 0, "assists": 0,
                "cs": 0, "gold": 0, "minutes": 0.0,
            }
        )
        # puuid -> champ_name -> {"games": int, "wins": int}
        self.player_champs: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"games": 0, "wins": 0})
        )
        # puuid -> role_name -> int
        self.player_roles: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # puuid -> deque of recent match outcomes (0 or 1)
        self.player_recent: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=self.recent_window))

    def get_player_priors(
        self,
        puuid: str | None,
        champion: str | None,
        role: str | None,
        extended: bool = False,
    ) -> list[float]:
        """Compute point-in-time Bayesian smoothed features for a player.

        If puuid is None, empty, or has no prior games, returns neutral priors.
        """
        if not puuid or puuid not in self.player_totals:
            return self.get_default_priors(extended=extended)

        tot = self.player_totals[puuid]
        tot_g = tot["games"]
        tot_w = tot["wins"]

        if tot_g == 0:
            return self.get_default_priors(extended=extended)

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

        base_priors = [float(smooth_wr), float(smooth_cwr), float(champ_exp), float(rec_wr), float(role_fam), float(tot_exp)]

        if extended:
            kills = tot.get("kills", 0)
            deaths = tot.get("deaths", 0)
            assists = tot.get("assists", 0)
            cs = tot.get("cs", 0)
            gold = tot.get("gold", 0)
            minutes = max(1.0, tot.get("minutes", 0.0))

            # Bayesian smoothed KDA (prior: 5 assists/kills for 2 deaths -> 2.50 KDA)
            smooth_kda = (kills + assists + 5.0) / (deaths + 2.0)
            norm_kda = min(2.0, smooth_kda / 5.0)

            # CS per min smoothed (prior: 60 cs in 10 min -> 6.0 CS/min)
            smooth_cspm = (cs + 60.0) / (minutes + 10.0)
            norm_cspm = min(1.5, smooth_cspm / 10.0)

            # Gold per min smoothed (prior: 3500 gold in 10 min -> 350 Gold/min)
            smooth_gpm = (gold + 3500.0) / (minutes + 10.0)
            norm_gpm = min(1.5, smooth_gpm / 500.0)

            # Action intensity / aggression (kills + deaths per minute)
            aggression = (kills + deaths + 1.0) / (minutes + 2.0)
            norm_aggression = min(1.5, aggression)

            return base_priors + [float(norm_kda), float(norm_cspm), float(norm_gpm), float(norm_aggression)]

        return base_priors

    @staticmethod
    def get_default_priors(extended: bool = False) -> list[float]:
        """Return neutral Bayesian priors for unknown players or cold starts."""
        base = [0.50, 0.50, 0.0, 0.50, 0.20, 0.0]
        if extended:
            return base + [0.50, 0.60, 0.70, 0.35]
        return base

    def update(
        self,
        puuid: str | None,
        champion: str | None,
        role: str | None,
        win: int,
        kills: int = 0,
        deaths: int = 0,
        assists: int = 0,
        total_cs: int = 0,
        gold_earned: int = 0,
        game_duration: int = 0,
    ) -> None:
        """Update tracker state after a match concludes."""
        if not puuid:
            return
        w = 1 if win else 0
        tot = self.player_totals[puuid]
        tot["games"] += 1
        tot["wins"] += w
        tot["kills"] += int(kills)
        tot["deaths"] += int(deaths)
        tot["assists"] += int(assists)
        tot["cs"] += int(total_cs)
        tot["gold"] += int(gold_earned)
        duration_min = max(5.0, min(60.0, float(game_duration) / 60.0)) if game_duration > 0 else 25.0
        tot["minutes"] += duration_min

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
        # Extended point-in-time priors retain the same six-field prefix.
        # Preserve that snapshot instead of falling back to current/neutral data.
        if "priors" in p and isinstance(p["priors"], (list, tuple)) and len(p["priors"]) >= PLAYER_PRIOR_FEATURE_COUNT:
            return [float(x) for x in p["priors"][:PLAYER_PRIOR_FEATURE_COUNT]]
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
