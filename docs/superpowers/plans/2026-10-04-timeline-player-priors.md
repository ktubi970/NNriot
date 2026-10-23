# Timeline Player Priors Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract early-game differential timeline metrics (GD@15, CSD@15, XPD@15, FBI@10) from historical match timelines, incorporate them as point-in-time Bayesian priors into player profiles, and extend the equivariant neural input tensors from 14 to 18 dimensions per player.

**Architecture:** Conforms to `AGENTS.md` conventions:
- Pure pre-match prediction: timelines of *past* matches inform the *current* player's priors without future leakage.
- Strict anti-symmetry: differential metrics against lane opponents invert sign under team swap ($GD_B = -GD_A$).
- Laplace Bayesian smoothing: unknown players or missing timelines cleanly fallback to neutral priors ($0.0$).

**Tech Stack:** Python 3.12+, SQLite3, NumPy, TensorFlow 2.x, Pytest.

## Global Constraints

- Must preserve team permutation anti-symmetry: $P(\text{winner}=1 \mid B, A) = 1 - P(\text{winner}=1 \mid A, B)$.
- Strict point-in-time ordering: no future match data or timeline data may leak into priors of earlier matches.
- Backward compatibility: existing 14-dimension tensor paths (`ENRICHED_PLAYER_TENSOR_DIM = 14`) and tabular LightGBM pipelines must continue to work without regression.
- All file paths in documentation and links must be relative or properly formatted.

---

### Task 1: Timeline Feature Extraction Module

**Files:**
- Create: `core/features/timeline_metrics.py`
- Test: `tests/test_timeline_metrics.py`

**Interfaces:**
- Consumes: Riot Match-V5 timeline dictionary (`info.frames`, `info.participants`, `events`).
- Produces: `extract_timeline_lane_metrics(timeline_data: dict, participants: list[dict] | None = None) -> dict[str, dict[str, float]]` mapping each player's `puuid` to `{"gd15": float, "csd15": float, "xpd15": float, "fbi10": float}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_timeline_metrics.py`:

```python
import pytest
from core.features.timeline_metrics import extract_timeline_lane_metrics

def test_extract_timeline_lane_metrics_standard():
    # Synthetic timeline with 16 frames and 10 participants
    frames = []
    for m in range(16):
        p_frames = {}
        for pid in range(1, 11):
            # Blue side (1-5) gets 500 gold/min, Red side (6-10) gets 400 gold/min
            gold = 500 * m if pid <= 5 else 400 * m
            cs = 8 * m if pid <= 5 else 7 * m
            xp = 400 * m if pid <= 5 else 350 * m
            p_frames[str(pid)] = {
                "participantId": pid,
                "totalGold": gold,
                "minionsKilled": cs,
                "jungleMinionsKilled": 0,
                "xp": xp,
            }
        events = []
        if m == 5:
            # First blood at min 5: killer pid 1, assist pid 2, victim pid 6
            events.append({
                "type": "CHAMPION_KILL",
                "timestamp": 300000,
                "killerId": 1,
                "assistingParticipantIds": [2],
                "victimId": 6,
            })
        frames.append({"participantFrames": p_frames, "events": events})

    timeline_data = {
        "info": {
            "participants": [{"participantId": i, "puuid": f"puuid_{i}"} for i in range(1, 11)],
            "frames": frames,
        }
    }

    metrics = extract_timeline_lane_metrics(timeline_data)
    assert len(metrics) == 10

    # Participant 1 (puuid_1, Top Blue) vs Participant 6 (puuid_6, Top Red)
    # At min 15: Gold A = 7500, Gold B = 6000 -> GD15 = +1500
    p1 = metrics["puuid_1"]
    assert p1["gd15"] == 1500.0
    assert p1["csd15"] == 15.0  # 120 - 105
    assert p1["xpd15"] == 750.0 # 6000 - 5250
    assert p1["fbi10"] == 1.0   # Killer in FB at min 5

    p6 = metrics["puuid_6"]
    assert p6["gd15"] == -1500.0  # Exact anti-symmetry
    assert p6["csd15"] == -15.0
    assert p6["xpd15"] == -750.0
    assert p6["fbi10"] == 0.0

    p2 = metrics["puuid_2"]
    assert p2["fbi10"] == 1.0  # Assist in FB

def test_extract_timeline_lane_metrics_empty_or_malformed():
    assert extract_timeline_lane_metrics({}) == {}
    assert extract_timeline_lane_metrics({"info": {}}) == {}
    assert extract_timeline_lane_metrics(None) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_timeline_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.features.timeline_metrics'`

- [ ] **Step 3: Write minimal implementation**

Create `core/features/timeline_metrics.py`:

```python
"""Extraction of early-game differential metrics from Riot Match-V5 timelines."""
from __future__ import annotations

from typing import Any

FIRST_BLOOD_CUTOFF_MS: int = 600_000  # 10 minutes in milliseconds
TARGET_FRAME_MINUTE: int = 15


def extract_timeline_lane_metrics(
    timeline_data: dict[str, Any] | None,
    participants: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, float]]:
    """Extract GD@15, CSD@15, XPD@15, and FBI@10 for all 10 participants.

    Returns:
        Mapping of puuid -> {"gd15": float, "csd15": float, "xpd15": float, "fbi10": float}
    """
    if not isinstance(timeline_data, dict):
        return {}

    info = timeline_data.get("info", timeline_data)
    frames = info.get("frames") or []
    if not frames:
        return {}

    part_meta = info.get("participants") or []
    pid_to_puuid: dict[int, str] = {}
    for p in part_meta:
        pid = p.get("participantId")
        puuid = p.get("puuid")
        if pid is not None and puuid:
            pid_to_puuid[int(pid)] = str(puuid)

    # Fallback to participants list if provided
    if participants:
        for idx, p in enumerate(participants, start=1):
            pid = p.get("participantId") or idx
            puuid = p.get("puuid")
            if puuid and int(pid) not in pid_to_puuid:
                pid_to_puuid[int(pid)] = str(puuid)

    if len(pid_to_puuid) < 10:
        # Require all 10 participants for valid lane differential
        return {}

    # Target frame: minute 15, or last available frame if game ended earlier
    frame_idx = min(TARGET_FRAME_MINUTE, len(frames) - 1)
    target_frame = frames[frame_idx]
    p_frames = target_frame.get("participantFrames") or {}

    # Extract per-participant stats
    stats: dict[int, dict[str, float]] = {}
    for pid in range(1, 11):
        pf = p_frames.get(str(pid)) or {}
        gold = float(pf.get("totalGold", 0.0))
        minions = float(pf.get("minionsKilled", 0.0))
        monsters = float(pf.get("jungleMinionsKilled", 0.0))
        xp = float(pf.get("xp", 0.0))
        stats[pid] = {
            "gold": gold,
            "cs": minions + monsters,
            "xp": xp,
        }

    # Detect First Blood involvement before 10 min
    fb_involved_pids: set[int] = set()
    found_first_blood = False

    for frame in frames:
        events = frame.get("events") or []
        for ev in events:
            if ev.get("type") == "CHAMPION_KILL":
                ts = ev.get("timestamp", 0)
                if ts <= FIRST_BLOOD_CUTOFF_MS:
                    killer = ev.get("killerId")
                    assists = ev.get("assistingParticipantIds") or []
                    if killer:
                        fb_involved_pids.add(int(killer))
                    for a in assists:
                        fb_involved_pids.add(int(a))
                found_first_blood = True
                break
        if found_first_blood:
            break

    # Calculate differentials against opposite lane counterpart (1 <-> 6, 2 <-> 7, etc.)
    result: dict[str, dict[str, float]] = {}
    for pid in range(1, 11):
        puuid = pid_to_puuid.get(pid)
        if not puuid:
            continue

        opp_pid = pid + 5 if pid <= 5 else pid - 5
        my_stat = stats.get(pid, {"gold": 0.0, "cs": 0.0, "xp": 0.0})
        opp_stat = stats.get(opp_pid, {"gold": 0.0, "cs": 0.0, "xp": 0.0})

        gd15 = my_stat["gold"] - opp_stat["gold"]
        csd15 = my_stat["cs"] - opp_stat["cs"]
        xpd15 = my_stat["xp"] - opp_stat["xp"]
        fbi10 = 1.0 if pid in fb_involved_pids else 0.0

        result[puuid] = {
            "gd15": float(gd15),
            "csd15": float(csd15),
            "xpd15": float(xpd15),
            "fbi10": float(fbi10),
        }

    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_timeline_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/features/timeline_metrics.py tests/test_timeline_metrics.py
git commit -m "feat(features): add timeline early-game lane metrics extractor"
```

---

### Task 2: Point-in-Time Bayesian Priors in `PlayerPriorsTracker`

**Files:**
- Modify: `core/player_priors.py`
- Test: `tests/test_player_priors_timeline.py`

**Interfaces:**
- Consumes: `timeline_metrics` dictionary from Task 1.
- Produces: `get_player_priors(..., include_timeline: bool = True)` returning a 14-element list (10 base priors + 4 normalized timeline priors).

- [ ] **Step 1: Write the failing test**

Create `tests/test_player_priors_timeline.py`:

```python
import pytest
from core.player_priors import PlayerPriorsTracker

def test_player_priors_timeline_cold_start():
    tracker = PlayerPriorsTracker()
    # Unknown player with include_timeline=True
    priors = tracker.get_player_priors("unknown_puuid", "Aatrox", "TOP", extended=True, include_timeline=True)
    assert len(priors) == 14
    # The last 4 are [norm_gd15, norm_csd15, norm_xpd15, norm_fbi10]
    assert priors[10] == 0.0   # Neutral GD15
    assert priors[11] == 0.0   # Neutral CSD15
    assert priors[12] == 0.0   # Neutral XPD15
    assert priors[13] == 0.10  # Default 10% base rate for FB involvement

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_player_priors_timeline.py -v`
Expected: FAIL with `TypeError: get_player_priors() got an unexpected keyword argument 'include_timeline'`

- [ ] **Step 3: Modify `core/player_priors.py`**

In `core/player_priors.py`:
1. In `__init__`, add:
```python
        # puuid -> {"count": int, "gd15_sum": float, "csd15_sum": float, "xpd15_sum": float, "fbi10_sum": float}
        self.player_timeline_totals: dict[str, dict[str, float]] = defaultdict(
            lambda: {"count": 0, "gd15_sum": 0.0, "csd15_sum": 0.0, "xpd15_sum": 0.0, "fbi10_sum": 0.0}
        )
```
2. Add the method `record_timeline_metrics`:
```python
    def record_timeline_metrics(
        self,
        puuid: str,
        gd15: float,
        csd15: float,
        xpd15: float,
        fbi10: float,
    ) -> None:
        """Accumulate early-game lane differential metrics for a player."""
        if not puuid:
            return
        t = self.player_timeline_totals[puuid]
        t["count"] += 1
        t["gd15_sum"] += float(gd15)
        t["csd15_sum"] += float(csd15)
        t["xpd15_sum"] += float(xpd15)
        t["fbi10_sum"] += float(fbi10)
```
3. Update `get_player_priors` and `get_default_priors` signatures:
```python
    def get_player_priors(
        self,
        puuid: str | None,
        champion: str | None,
        role: str | None,
        extended: bool = False,
        include_timeline: bool = False,
    ) -> list[float]:
```
When `include_timeline=True`:
```python
        if include_timeline:
            t = self.player_timeline_totals.get(puuid or "", {"count": 0, "gd15_sum": 0.0, "csd15_sum": 0.0, "xpd15_sum": 0.0, "fbi10_sum": 0.0})
            cnt = t["count"]
            w = 5.0
            smooth_gd15 = (t["gd15_sum"] + w * 0.0) / (cnt + w)
            smooth_csd15 = (t["csd15_sum"] + w * 0.0) / (cnt + w)
            smooth_xpd15 = (t["xpd15_sum"] + w * 0.0) / (cnt + w)
            smooth_fbi10 = (t["fbi10_sum"] + w * 0.10) / (cnt + w)

            norm_gd15 = float(np.clip(smooth_gd15 / 2000.0, -1.0, 1.0))
            norm_csd15 = float(np.clip(smooth_csd15 / 30.0, -1.0, 1.0))
            norm_xpd15 = float(np.clip(smooth_xpd15 / 1500.0, -1.0, 1.0))
            norm_fbi10 = float(np.clip(smooth_fbi10, 0.0, 1.0))

            base_priors.extend([norm_gd15, norm_csd15, norm_xpd15, norm_fbi10])
```
And in `get_default_priors`:
```python
    @staticmethod
    def get_default_priors(extended: bool = False, include_timeline: bool = False) -> list[float]:
        base = [0.50, 0.50, 0.0, 0.50, 0.20, 0.0]
        if extended:
            base.extend([0.50, 0.60, 0.70, 0.35])
        if include_timeline:
            base.extend([0.0, 0.0, 0.0, 0.10])
        return base
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_player_priors_timeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/player_priors.py tests/test_player_priors_timeline.py
git commit -m "feat(priors): support Bayesian early-game timeline priors in PlayerPriorsTracker"
```

---

### Task 3: Pipeline & Tensor Dimension Extension

**Files:**
- Modify: `core/features/pipeline.py`
- Test: `tests/test_timeline_pipeline.py`

**Interfaces:**
- Consumes: `PlayerPriorsTracker` with timeline priors.
- Produces: `DraftFeaturePipeline.extract_tensors(..., include_timeline: bool = True)` returning `(team_a, team_b)` tensors of shape `(5, 18)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_timeline_pipeline.py`:

```python
import numpy as np
import pytest
from core.features.pipeline import (
    DraftFeaturePipeline,
    ENRICHED_PLAYER_TENSOR_DIM,
    ENRICHED_PLAYER_TIMELINE_TENSOR_DIM,
)
from core.player_priors import PlayerPriorsTracker

def test_pipeline_tensor_dimensions():
    pipeline = DraftFeaturePipeline()
    sample_match = {
        "team_a": [{"puuid": f"a_{i}", "champion": "Aatrox", "role": "TOP"} for i in range(5)],
        "team_b": [{"puuid": f"b_{i}", "champion": "Darius", "role": "TOP"} for i in range(5)],
    }

    # Backward compatibility: default is (5, 14)
    t_a, t_b = pipeline.extract_tensors(sample_match)
    assert t_a.shape == (5, 14)
    assert t_b.shape == (5, 14)

    # Timeline augmented: (5, 18)
    t_a_tl, t_b_tl = pipeline.extract_tensors(sample_match, include_timeline=True)
    assert t_a_tl.shape == (5, 18)
    assert t_b_tl.shape == (5, 18)

def test_pipeline_timeline_anti_symmetry():
    tracker = PlayerPriorsTracker()
    # Team A player has +1000 GD15, Team B player has -1000 GD15
    tracker.record_timeline_metrics("p_a", gd15=1000.0, csd15=15.0, xpd15=500.0, fbi10=1.0)
    tracker.record_timeline_metrics("p_b", gd15=-1000.0, csd15=-15.0, xpd15=-500.0, fbi10=0.0)

    pipeline = DraftFeaturePipeline(player_tracker=tracker)
    match_forward = {
        "team_a": [{"puuid": "p_a", "champion": "Aatrox", "role": "TOP"}] + [{"puuid": f"a_{i}", "champion": "Ashe", "role": "BOTTOM"} for i in range(4)],
        "team_b": [{"puuid": "p_b", "champion": "Darius", "role": "TOP"}] + [{"puuid": f"b_{i}", "champion": "Jinx", "role": "BOTTOM"} for i in range(4)],
    }
    match_swapped = {
        "team_a": match_forward["team_b"],
        "team_b": match_forward["team_a"],
    }

    ta_f, tb_f = pipeline.extract_tensors(match_forward, include_timeline=True)
    ta_s, tb_s = pipeline.extract_tensors(match_swapped, include_timeline=True)

    # Inverting match teams must map ta_f exactly to tb_s, and tb_f to ta_s
    np.testing.assert_allclose(ta_f, tb_s, rtol=1e-5)
    np.testing.assert_allclose(tb_f, ta_s, rtol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_timeline_pipeline.py -v`
Expected: FAIL with `ImportError: cannot import name 'ENRICHED_PLAYER_TIMELINE_TENSOR_DIM'`

- [ ] **Step 3: Modify `core/features/pipeline.py`**

In `core/features/pipeline.py`:
1. Add constant:
```python
TIMELINE_PLAYER_FEATURE_COUNT: int = 4
ENRICHED_PLAYER_TIMELINE_TENSOR_DIM: int = ENRICHED_PLAYER_TENSOR_DIM + TIMELINE_PLAYER_FEATURE_COUNT  # 18
```
2. Update `extract_tensors` signature and logic:
```python
    def extract_tensors(
        self,
        feature_dict: dict[str, Any] | None,
        include_spatial: bool = False,
        include_timeline: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if include_spatial:
            feat_dim = ENRICHED_PLAYER_SPATIAL_TENSOR_DIM
        elif include_timeline:
            feat_dim = ENRICHED_PLAYER_TIMELINE_TENSOR_DIM
        else:
            feat_dim = ENRICHED_PLAYER_TENSOR_DIM

        t_a = np.zeros((5, feat_dim), dtype=np.float32)
        t_b = np.zeros((5, feat_dim), dtype=np.float32)
```
In priors unpacking:
```python
            def _get_priors(p, cid):
                if "priors" in p and isinstance(p["priors"], (list, tuple)):
                    p_list = [float(x) for x in p["priors"]]
                    if include_timeline:
                        if len(p_list) >= 14:
                            return p_list[:14]
                        return p_list[:10] + [0.0, 0.0, 0.0, 0.10]
                    if len(p_list) >= 10:
                        return p_list[:10]
                    return p_list[:6] + default_priors[6:10]
                if self.player_tracker is not None:
                    return self.player_tracker.get_player_priors(
                        p.get("puuid"), cid, role_name, extended=True, include_timeline=include_timeline
                    )
                return self.player_tracker.get_default_priors(extended=True, include_timeline=include_timeline) if self.player_tracker else default_priors
```
Pack into tensors:
```python
            pri_len = 14 if include_timeline else 10
            t_a[slot, 0] = float(cid_a)
            t_a[slot, 1] = float(slot)
            t_a[slot, 2 : 2 + pri_len] = pri_a
            t_a[slot, 2 + pri_len] = float(delta_a)
            t_a[slot, 3 + pri_len] = float(prof_a.base_winrate)

            t_b[slot, 0] = float(cid_b)
            t_b[slot, 1] = float(slot)
            t_b[slot, 2 : 2 + pri_len] = pri_b
            t_b[slot, 2 + pri_len] = float(delta_b)
            t_b[slot, 3 + pri_len] = float(prof_b.base_winrate)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_timeline_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/features/pipeline.py tests/test_timeline_pipeline.py
git commit -m "feat(pipeline): support 18-dimensional timeline-augmented team tensors"
```

---

### Task 4: Chronological Point-in-Time Streaming Integration

**Files:**
- Modify: `core/point_in_time_dataset.py`
- Test: `tests/test_point_in_time_timeline.py`

**Interfaces:**
- Consumes: `match_timelines` table in SQLite.
- Produces: `load_enriched_records(..., include_timeline: bool = True)` producing records where each participant dictionary includes timeline priors.

- [ ] **Step 1: Write the failing test**

Create `tests/test_point_in_time_timeline.py`:

```python
import sqlite3
import json
import pytest
from core.point_in_time_dataset import load_enriched_records

def test_point_in_time_timeline_loading(tmp_path):
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(db_file)
    conn.execute("""
        CREATE TABLE matches (
            match_id TEXT PRIMARY KEY,
            game_mode TEXT,
            game_creation_time INTEGER,
            raw_json TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE training_dataset (
            match_id TEXT PRIMARY KEY,
            feature_json TEXT,
            labels_json TEXT,
            winner_label INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE match_participants (
            rowid INTEGER PRIMARY KEY,
            match_id TEXT,
            puuid TEXT,
            champion_name TEXT,
            team_position TEXT,
            win INTEGER,
            kills INTEGER,
            deaths INTEGER,
            assists INTEGER,
            total_cs INTEGER,
            gold_earned INTEGER,
            game_duration INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE match_timelines (
            match_id TEXT PRIMARY KEY,
            timeline_json TEXT
        )
    """)

    # Insert 2 matches: M1 at t=1000, M2 at t=2000
    for mid, gtime in [("M1", 1000), ("M2", 2000)]:
        conn.execute("INSERT INTO matches VALUES (?, 'CLASSIC', ?, '{}')", (mid, gtime))
        conn.execute("INSERT INTO training_dataset VALUES (?, '{}', '{\"winner\": 1}', 1)", (mid,))
        for idx in range(1, 11):
            conn.execute(
                "INSERT INTO match_participants VALUES (NULL, ?, ?, 'Aatrox', 'TOP', 1, 3, 1, 4, 150, 8000, 1500)",
                (mid, f"puuid_{idx}"),
            )

    # Insert timeline for M1 only
    synthetic_timeline = {
        "info": {
            "participants": [{"participantId": i, "puuid": f"puuid_{i}"} for i in range(1, 11)],
            "frames": [
                {"participantFrames": {str(i): {"totalGold": 5000, "minionsKilled": 100, "jungleMinionsKilled": 0, "xp": 4000} for i in range(1, 11)}, "events": []}
            ],
        }
    }
    conn.execute("INSERT INTO match_timelines VALUES ('M1', ?)", (json.dumps(synthetic_timeline),))
    conn.commit()
    conn.close()

    records = load_enriched_records(str(db_file), limit=2, warmup_matches=0, include_timeline=True)
    assert len(records) == 2

    # M1 was processed first: priors must be neutral cold-start (14 features)
    priors_m1 = records[0]["feature_json"]["team_a"][0]["priors"]
    assert len(priors_m1) == 14
    assert priors_m1[10] == 0.0  # Cold start GD15

    # M2 was processed after M1: tracker has been updated with M1's timeline
    priors_m2 = records[1]["feature_json"]["team_a"][0]["priors"]
    assert len(priors_m2) == 14
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_point_in_time_timeline.py -v`
Expected: FAIL with `TypeError: load_enriched_records() got an unexpected keyword argument 'include_timeline'`

- [ ] **Step 3: Modify `core/point_in_time_dataset.py`**

In `core/point_in_time_dataset.py`:
1. Add `include_timeline: bool = False` to `load_enriched_records`.
2. Check for `match_timelines` table:
```python
        has_timelines = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='match_timelines'"
        ).fetchone() is not None
```
3. If `has_timelines` and `include_timeline`, batch pre-fetch timelines for candidate matches:
```python
        timelines_by_match: dict[str, Any] = {}
        if has_timelines and include_timeline:
            for i in range(0, len(candidate_mids), chunk_size):
                chunk = candidate_mids[i : i + chunk_size]
                ph = ",".join("?" for _ in chunk)
                t_cursor = conn.execute(
                    f"SELECT match_id, timeline_json FROM match_timelines WHERE match_id IN ({ph}) AND timeline_json IS NOT NULL",
                    chunk,
                )
                for trow in t_cursor:
                    try:
                        raw_t = trow["timeline_json"]
                        t_data = json.loads(raw_t) if raw_t.startswith("{") else json.loads(gzip.decompress(base64.b64decode(raw_t)).decode("utf-8"))
                        timelines_by_match[trow["match_id"]] = t_data
                    except Exception:
                        pass
```
4. In the streaming loop:
```python
                priors = tracker.get_player_priors(puuid, champ, pos, extended=True, include_timeline=include_timeline)
```
5. In tracker outcome update:
```python
            if include_timeline and mid in timelines_by_match:
                from core.features.timeline_metrics import extract_timeline_lane_metrics
                t_metrics = extract_timeline_lane_metrics(timelines_by_match[mid])
                for puuid, m in t_metrics.items():
                    tracker.record_timeline_metrics(
                        puuid=puuid,
                        gd15=m["gd15"],
                        csd15=m["csd15"],
                        xpd15=m["xpd15"],
                        fbi10=m["fbi10"],
                    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_point_in_time_timeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/point_in_time_dataset.py tests/test_point_in_time_timeline.py
git commit -m "feat(dataset): stream point-in-time timeline priors in load_enriched_records"
```

---

### Task 5: DeepSets v4 Model Training Script & Benchmark

**Files:**
- Create: `scripts/train_deepsets_v4_timeline.py`
- Test: `tests/test_train_deepsets_v4_timeline.py`

**Interfaces:**
- Consumes: Enriched dataset records with 18-dim tensors.
- Produces: Trained Keras model with `player_feature_dim=18`, metrics file `model.metrics.json`, and calibration file `model.calibration.json`.

- [ ] **Step 1: Write integration test for DeepSets v4 architecture**

Create `tests/test_train_deepsets_v4_timeline.py`:

```python
import numpy as np
import pytest
from core.equivariant_model import build_equivariant_deepsets_model

def test_build_deepsets_v4_timeline_model():
    model = build_equivariant_deepsets_model(
        player_feature_dim=18,
        include_macro_clash=True,
        macro_feature_dim=15,
        use_cross_attention=False,
    )
    assert model is not None

    batch_size = 2
    xa = np.zeros((batch_size, 5, 18), dtype=np.float32)
    xb = np.zeros((batch_size, 5, 18), dtype=np.float32)
    xm = np.zeros((batch_size, 15), dtype=np.float32)

    preds = model.predict({"team_a": xa, "team_b": xb, "macro_clash": xm}, verbose=0)
    assert "winner" in preds
    assert preds["winner"].shape == (batch_size, 2)
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `pytest tests/test_train_deepsets_v4_timeline.py -v`
Expected: PASS (verifying existing equivariant builder accepts `player_feature_dim=18`).

- [ ] **Step 3: Implement `scripts/train_deepsets_v4_timeline.py`**

Create `scripts/train_deepsets_v4_timeline.py`:

```python
"""Train Equivariant DeepSets v4 with 18-dim player tensors including timeline priors."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.database import DB_PATH
from core.features.pipeline import ENRICHED_PLAYER_TIMELINE_TENSOR_DIM, DraftFeaturePipeline
from core.features.draft_composition import MACRO_CLASH_FEATURE_DIM
from core.equivariant_model import build_equivariant_deepsets_model
from core.generate_graph import LOSS_PER_HEAD
from core.point_in_time_dataset import load_enriched_records
from core.pre_match_dataset import split_records
from core.training_utils import _build_targets
from core.training_metrics import evaluate_winner
from core.calibration import fit_temperature, save_calibration_artifact
from core.experiment import build_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_deepsets_v4_timeline")


def train_deepsets_v4_timeline(
    db_path: str = DB_PATH,
    output_dir: str = "runs/candidate_deepsets_v4_timeline",
    samples: int = 15000,
    warmup: int = 3000,
    epochs: int = 5,
    batch_size: int = 64,
    seed: int = 42,
    learning_rate: float = 0.0008,
) -> dict:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Streaming %d records with timeline priors from %s...", samples, db_path)
    records = load_enriched_records(db_path=db_path, limit=samples, warmup_matches=warmup, include_timeline=True)
    if len(records) < 100:
        raise ValueError(f"Insufficient records loaded: {len(records)}")

    splits, split_meta = split_records(records, strategy="chronological")
    manifest = build_manifest(splits, source_db=db_path, split_metadata=split_meta)
    with (out_path / "model.split.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    pipeline = DraftFeaturePipeline()
    tensors = {}
    for split_name, recs in splits.items():
        n = len(recs)
        ta = np.zeros((n, 5, ENRICHED_PLAYER_TIMELINE_TENSOR_DIM), dtype=np.float32)
        tb = np.zeros((n, 5, ENRICHED_PLAYER_TIMELINE_TENSOR_DIM), dtype=np.float32)
        tm = np.zeros((n, MACRO_CLASH_FEATURE_DIM), dtype=np.float32)
        for i, r in enumerate(recs):
            feat = r["feature_json"]
            a, b = pipeline.extract_tensors(feat, include_timeline=True)
            ta[i] = a
            tb[i] = b
            tm[i] = pipeline.extract_macro_clash(feat)
        tensors[split_name] = {"team_a": ta, "team_b": tb, "macro_clash": tm}

    targets = {name: _build_targets(recs) for name, recs in splits.items()}

    logger.info("Building DeepSets model with %d-dim player features...", ENRICHED_PLAYER_TIMELINE_TENSOR_DIM)
    model = build_equivariant_deepsets_model(
        player_feature_dim=ENRICHED_PLAYER_TIMELINE_TENSOR_DIM,
        include_macro_clash=True,
        macro_feature_dim=MACRO_CLASH_FEATURE_DIM,
        use_cross_attention=False,
        learning_rate=learning_rate,
    )

    # Train model
    history = model.fit(
        x=tensors["train"],
        y=targets["train"],
        validation_data=(tensors["validation"], targets["validation"]),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # Save model checkpoint
    model_path = out_path / "model.keras"
    model.save(model_path)

    # Evaluate test split
    test_preds = model.predict(tensors["test"], verbose=0)
    p_b_test = test_preds["winner"][:, 1]
    y_test = (splits["test"][i]["winner_label"] == 1).astype(int) if hasattr(splits["test"][0]["winner_label"], "__int__") else np.array([1 if r["winner_label"] == "100" or r["winner_label"] == 1 else 0 for r in splits["test"]])
    val_preds = model.predict(tensors["validation"], verbose=0)
    p_b_val = val_preds["winner"][:, 1]
    y_val = np.array([1 if r["winner_label"] == "100" or r["winner_label"] == 1 else 0 for r in splits["validation"]])

    # Temperature calibration
    temp = fit_temperature(p_b_val, y_val)
    save_calibration_artifact(out_path / "model.calibration.json", temperature=temp, method="platt")

    eval_report = evaluate_winner(p_b_test, y_test)
    metrics_summary = {
        "model_type": "deepsets_v4_timeline",
        "player_feature_dim": ENRICHED_PLAYER_TIMELINE_TENSOR_DIM,
        "temperature": temp,
        "test_eval": eval_report,
    }
    with (out_path / "model.metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info("Training complete. Test accuracy: %.4f, Log loss: %.4f", eval_report["accuracy"], eval_report["log_loss"])
    return metrics_summary


if __name__ == "__main__":
    train_deepsets_v4_timeline()
```

- [ ] **Step 4: Run test to verify passing**

Run: `pytest tests/test_train_deepsets_v4_timeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/train_deepsets_v4_timeline.py tests/test_train_deepsets_v4_timeline.py
git commit -m "feat(model): add train_deepsets_v4_timeline training and evaluation pipeline"
```
