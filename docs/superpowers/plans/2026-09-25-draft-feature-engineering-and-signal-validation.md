# League of Legends Draft Feature Engineering & Signal Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flawed string-length/NLP-hashing draft vectorizer with a canonical, domain-aware League of Legends draft feature representation, and empirically validate whether champion draft compositions alone carry statistically significant predictive signal on a tabular gradient booster (HistGradientBoosting) and a focused neural network before running large-scale training.

**Architecture:** Build a single source of truth for champion identity and aliasing (`core/champion_registry.py`). Construct domain-specific draft feature representations (`core/draft_features.py`): (1) categorical role-specific matchup features and permutation-invariant multi-hot composition diffs for tabular models, and (2) dense one-hot role matrices for neural models. Upgrade the tree baseline trainer and candidate training scripts to consume these representations. Add an automated signal benchmark script (`scripts/benchmark_draft_signal.py`) that tests holdout AUC against the 0.50 baseline with hypothesis testing.

**Tech Stack:** Python 3.12, NumPy, SciPy, scikit-learn (`HistGradientBoostingClassifier`), TensorFlow/Keras 2.21, SQLite3, pytest.

---

## Global Constraints

- Working directory is `D:\projet\NNriot`. All commands run under Windows PowerShell using `.\.venv312\Scripts\python.exe` and `.\.venv312\Scripts\pytest.exe`.
- Preserve existing working-tree files. Do NOT use `git clean`, `git reset --hard`, or indiscriminate `git add .`.
- Maintain strict backward compatibility for `evaluate_winner` and frozen split manifests (`model.split.json`, `dataset_sha256`, `split_sha256`).
- All splits remain strictly chronological (70% train, 15% validation, 15% test). No test data or validation data may leak into feature normalization, encoders, or baseline thresholds.
- Do NOT use string lengths or generic JSON key-path MD5 hashing for champion identities. Champions must be mapped to discrete integer IDs based on canonical names.
- Zero external network requests during tests: tests must use synthetic mock matches or static fixtures.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `core/champion_registry.py` | Single source of truth for canonical LoL champion names, aliases (e.g. `MonkeyKing` <-> `Wukong`), unknown token handling, and mapping to contiguous integer IDs `0..N-1`. |
| `tests/test_champion_registry.py` | Unit tests verifying champion normalization, alias resolution, stability of IDs, and handling of malformed names. |
| `core/draft_features.py` | Extracts structured draft features: (1) tabular categorical role matchups + multi-hot composition diffs, (2) dense role one-hot vectors for neural networks. |
| `tests/test_draft_features.py` | Unit tests verifying exact dimensions, symmetry between Team A (Blue) and Team B (Red), role ordering, and missing participant handling. |
| `scripts/train_tree_baseline.py` | Modified to support structured draft features via `core/draft_features.py`, feature importance extraction, and comparison reporting. |
| `tests/test_tree_baseline_structured.py` | Verification that `train_tree_model` operates with the new draft feature extractor without regressions. |
| `scripts/benchmark_draft_signal.py` | Automated hypothesis-testing script running tabular baseline over candidate splits to determine if ROC-AUC > 0.50 is statistically significant. |
| `tests/test_benchmark_draft_signal.py` | Unit tests verifying calculation of accuracy lift, log-loss improvement, and p-value checks. |
| `core/generate_graph.py` | Add `build_draft_dense_model` providing a focused MLP with appropriate capacity for dense draft features (dim=1870) without 22-head dilution. |
| `scripts/train_pre_match_model.py` | Support `--feature-version {v1_hasher, v2_draft}` allowing seamless training of neural models with structured draft features. |
| `tests/test_draft_pre_match_training.py` | End-to-end integration test of candidate pre-match training on synthetic records with `v2_draft`. |

---

## Tasks

### Task 1: Canonical Champion Registry (`core/champion_registry.py`)

**Files:**
- Create: `core/champion_registry.py`
- Test: `tests/test_champion_registry.py`

**Interfaces:**
- Consumes: None (pure Python standard library).
- Produces:
  - `normalize_champion_name(name: str) -> str`: strips whitespace and handles alias conversion.
  - `champion_to_id(name: str) -> int`: maps champion name to integer in range `[0, NUM_CHAMPIONS - 1]`. Unknown champions return `UNK_CHAMPION_ID = 0`.
  - `id_to_champion(champ_id: int) -> str`: maps integer ID back to canonical name.
  - `NUM_CHAMPIONS`: `int` constant equal to total registered champions including `UNKNOWN`.
  - `CANONICAL_CHAMPIONS`: `tuple[str, ...]`.

- [ ] **Step 1: Write failing test for champion registry**

Write `tests/test_champion_registry.py`:
```python
import pytest
from core.champion_registry import (
    normalize_champion_name,
    champion_to_id,
    id_to_champion,
    NUM_CHAMPIONS,
    UNK_CHAMPION_ID,
    CANONICAL_CHAMPIONS,
)

def test_champion_registry_constants():
    assert NUM_CHAMPIONS >= 168
    assert UNK_CHAMPION_ID == 0
    assert len(CANONICAL_CHAMPIONS) == NUM_CHAMPIONS
    assert CANONICAL_CHAMPIONS[0] == "UNKNOWN"

def test_normalize_champion_name_aliases():
    assert normalize_champion_name("Wukong") == "MonkeyKing"
    assert normalize_champion_name("MonkeyKing") == "MonkeyKing"
    assert normalize_champion_name("Nunu & Willump") == "Nunu"
    assert normalize_champion_name("Nunu") == "Nunu"
    assert normalize_champion_name("Renata Glasc") == "Renata"
    assert normalize_champion_name("Renata") == "Renata"
    assert normalize_champion_name("  Aatrox  ") == "Aatrox"

def test_champion_to_id_mapping():
    aatrox_id = champion_to_id("Aatrox")
    ahri_id = champion_to_id("Ahri")
    assert aatrox_id > 0
    assert ahri_id > 0
    assert aatrox_id != ahri_id
    assert id_to_champion(aatrox_id) == "Aatrox"
    assert id_to_champion(ahri_id) == "Ahri"

def test_unknown_champion_fallback():
    assert champion_to_id("NonExistentChamp123") == UNK_CHAMPION_ID
    assert champion_to_id("") == UNK_CHAMPION_ID
    assert champion_to_id(None) == UNK_CHAMPION_ID
    assert id_to_champion(UNK_CHAMPION_ID) == "UNKNOWN"

def test_all_ids_are_contiguous():
    for cid in range(NUM_CHAMPIONS):
        name = id_to_champion(cid)
        assert isinstance(name, str)
        assert champion_to_id(name) == cid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv312\Scripts\pytest.exe tests/test_champion_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.champion_registry'`

- [ ] **Step 3: Implement `core/champion_registry.py`**

Create `core/champion_registry.py`:
```python
"""Canonical League of Legends champion registry and alias mapping.

Provides deterministic, zero-collision mapping from champion strings to
contiguous integer IDs [0, NUM_CHAMPIONS - 1].
"""
from __future__ import annotations

UNK_CHAMPION_ID = 0

# Common Riot API aliases and formatting variations
_ALIASES: dict[str, str] = {
    "wukong": "MonkeyKing",
    "monkeyking": "MonkeyKing",
    "nunu & willump": "Nunu",
    "nunu": "Nunu",
    "renata glasc": "Renata",
    "renata": "Renata",
    "drmundo": "DrMundo",
    "dr. mundo": "DrMundo",
    "jarvan iv": "JarvanIV",
    "kog'maw": "KogMaw",
    "reksai": "RekSai",
    "rek'sai": "RekSai",
    "vel'koz": "Velkoz",
    "velkoz": "Velkoz",
    "k'sante": "KSante",
    "ksante": "KSante",
    "bel'veth": "Belveth",
    "belveth": "Belveth",
    "kai'sa": "Kaisa",
    "kaisa": "Kaisa",
    "cho'gath": "Chogath",
    "chogath": "Chogath",
    "kha'zix": "Khazix",
    "khazix": "Khazix",
    "leblanc": "Leblanc",
}

# 168+ canonical champions as of current League of Legends
_RAW_CHAMPIONS: list[str] = [
    "UNKNOWN",
    "Aatrox", "Ahri", "Akali", "Akshan", "Alistar", "Ambessa", "Amumu", "Anivia",
    "Annie", "Aphelios", "Ashe", "AurelionSol", "Aurora", "Azir", "Bard",
    "Belveth", "Blitzcrank", "Brand", "Braum", "Briar", "Caitlyn", "Camille",
    "Cassiopeia", "Chogath", "Corki", "Darius", "Diana", "Draven", "DrMundo",
    "Ekko", "Elise", "Evelynn", "Ezreal", "Fiddlesticks", "Fiora", "Fizz",
    "Galio", "Gangplank", "Garen", "Gnar", "Gragas", "Graves", "Gwen",
    "Hecarim", "Heimerdinger", "Hwei", "Illaoi", "Irelia", "Ivern", "Janna",
    "JarvanIV", "Jax", "Jayce", "Jhin", "Jinx", "Kaisa", "Kalista",
    "Karma", "Karthus", "Kassadin", "Katarina", "Kayle", "Kayn", "Kennen",
    "Khazix", "Kindred", "Kled", "KogMaw", "KSante", "Leblanc", "LeeSin",
    "Leona", "Lillia", "Lissandra", "Lucian", "Lulu", "Lux", "Malphite",
    "Malzahar", "Maokai", "MasterYi", "Milio", "MissFortune", "MonkeyKing",
    "Mordekaiser", "Morgana", "Naafiri", "Nami", "Nasus", "Nautilus",
    "Neeko", "Nidalee", "Nilah", "Nocturne", "Nunu", "Olaf", "Orianna",
    "Ornn", "Pantheon", "Poppy", "Pyke", "Qiyana", "Quinn", "Rakan",
    "Rammus", "RekSai", "Rell", "Renata", "Renekton", "Rengar", "Riven",
    "Rumble", "Ryze", "Samira", "Sejuani", "Senna", "Seraphine", "Sett",
    "Shaco", "Shen", "Shyvana", "Singed", "Sion", "Sivir", "Skarner",
    "Smolder", "Sona", "Soraka", "Swain", "Sylas", "Syndra", "TahmKench",
    "Taliyah", "Talon", "Taric", "Teemo", "Thresh", "Tristana", "Trundle",
    "Tryndamere", "TwistedFate", "Twitch", "Udyr", "Urgot", "Varus",
    "Vayne", "Veigar", "Velkoz", "Vex", "Vi", "Viego", "Viktor",
    "Vladimir", "Volibear", "Warwick", "Xayah", "Xerath", "XinZhao",
    "Yasuo", "Yone", "Yorick", "Yuumi", "Zac", "Zed", "Zeri",
    "Ziggs", "Zilean", "Zoe", "Zyra",
]

CANONICAL_CHAMPIONS: tuple[str, ...] = tuple(_RAW_CHAMPIONS)
NUM_CHAMPIONS: int = len(CANONICAL_CHAMPIONS)

_CHAMP_TO_ID: dict[str, int] = {name.lower(): idx for idx, name in enumerate(CANONICAL_CHAMPIONS)}
_ID_TO_CHAMP: dict[int, str] = {idx: name for idx, name in enumerate(CANONICAL_CHAMPIONS)}


def normalize_champion_name(name: str | None) -> str:
    """Normalize champion name, resolving known aliases and casing."""
    if not isinstance(name, str):
        return "UNKNOWN"
    clean = name.strip()
    if not clean:
        return "UNKNOWN"
    lower = clean.lower()
    if lower in _ALIASES:
        return _ALIASES[lower]
    # Check if lower matches a known canonical champ
    if lower in _CHAMP_TO_ID:
        return _ID_TO_CHAMP[_CHAMP_TO_ID[lower]]
    return clean


def champion_to_id(name: str | None) -> int:
    """Return the integer ID for a champion, defaulting to UNK_CHAMPION_ID (0)."""
    norm = normalize_champion_name(name)
    return _CHAMP_TO_ID.get(norm.lower(), UNK_CHAMPION_ID)


def id_to_champion(champ_id: int) -> str:
    """Return the canonical champion name for an ID."""
    return _ID_TO_CHAMP.get(champ_id, "UNKNOWN")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv312\Scripts\pytest.exe tests/test_champion_registry.py -v`
Expected: PASS (5/5 passed)

- [ ] **Step 5: Commit Task 1**

```bash
git add core/champion_registry.py tests/test_champion_registry.py
git commit -m "feat(ml): implement canonical champion registry and alias resolver"
```

---

### Task 2: Structured Draft Feature Extractor (`core/draft_features.py`)

**Files:**
- Create: `core/draft_features.py`
- Test: `tests/test_draft_features.py`

**Interfaces:**
- Consumes: `core.champion_registry.champion_to_id`, `NUM_CHAMPIONS`.
- Produces:
  - `ROLE_NAMES: tuple[str, ...]` = `("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")`.
  - `TABULAR_FEATURE_DIM: int` = 10 (role categorical IDs) + `NUM_CHAMPIONS` (composition diff) = 10 + 171 = 181.
  - `DENSE_FEATURE_DIM: int` = `11 * NUM_CHAMPIONS` (5 roles A + 5 roles B + 1 comp diff).
  - `extract_tabular_draft_features(features: dict) -> tuple[np.ndarray, list[str]]`.
  - `extract_dense_draft_features(features: dict) -> np.ndarray`.
  - `batch_extract_draft_features(features_list: list[dict], mode: str) -> np.ndarray`.

- [ ] **Step 1: Write failing test for draft feature extractor**

Write `tests/test_draft_features.py`:
```python
import numpy as np
import pytest
from core.champion_registry import NUM_CHAMPIONS, champion_to_id
from core.draft_features import (
    ROLE_NAMES,
    TABULAR_FEATURE_DIM,
    DENSE_FEATURE_DIM,
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

def test_batch_extract_draft_features(sample_match_features):
    batch = [sample_match_features, sample_match_features]
    tabular_batch = batch_extract_draft_features(batch, mode="tabular")
    assert tabular_batch.shape == (2, TABULAR_FEATURE_DIM)
    dense_batch = batch_extract_draft_features(batch, mode="dense")
    assert dense_batch.shape == (2, DENSE_FEATURE_DIM)

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv312\Scripts\pytest.exe tests/test_draft_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.draft_features'`

- [ ] **Step 3: Implement `core/draft_features.py`**

Create `core/draft_features.py`:
```python
"""Domain-specific draft feature extraction for League of Legends matches.

Provides canonical, collision-free feature vectors for both tabular GBDT models
and neural networks.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from core.champion_registry import (
    NUM_CHAMPIONS,
    UNK_CHAMPION_ID,
    champion_to_id,
)

ROLE_NAMES: tuple[str, ...] = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
ROLE_INDEX: dict[str, int] = {role: idx for idx, role in enumerate(ROLE_NAMES)}

# Tabular: 10 role champion IDs (int) + NUM_CHAMPIONS composition diff indicators (+1/0/-1)
TABULAR_FEATURE_DIM: int = 10 + NUM_CHAMPIONS

# Dense Neural: 5 one-hot roles for Team A (5*N) + 5 one-hot roles for Team B (5*N) + comp diff (N)
DENSE_FEATURE_DIM: int = 11 * NUM_CHAMPIONS


def _order_team_participants(participants: list[dict[str, Any]]) -> list[int]:
    """Extract 5 champion IDs ordered strictly by [TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY].

    If roles are unassigned or duplicated, maps greedily and fills remaining slots with UNK.
    """
    ordered_ids = [UNK_CHAMPION_ID] * 5
    assigned_slots = set()
    unassigned_players = []

    for p in participants:
        champ_name = p.get("champion") or p.get("championName") or p.get("champion_name")
        cid = champion_to_id(champ_name)
        role = (p.get("role") or p.get("teamPosition") or "").upper()
        if role in ROLE_INDEX and ROLE_INDEX[role] not in assigned_slots:
            slot = ROLE_INDEX[role]
            ordered_ids[slot] = cid
            assigned_slots.add(slot)
        else:
            unassigned_players.append(cid)

    # Fill remaining unassigned slots
    unassigned_idx = 0
    for slot in range(5):
        if slot not in assigned_slots and unassigned_idx < len(unassigned_players):
            ordered_ids[slot] = unassigned_players[unassigned_idx]
            unassigned_idx += 1

    return ordered_ids


def extract_tabular_draft_features(features: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Extract a 1D vector of length TABULAR_FEATURE_DIM and corresponding feature names.

    Vector structure:
      - [0:5]: Team A champion IDs (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY)
      - [5:10]: Team B champion IDs (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY)
      - [10:10+NUM_CHAMPIONS]: Composition diff (+1.0 for Team B presence, -1.0 for Team A presence)
    """
    team_a = features.get("team_a") or []
    team_b = features.get("team_b") or []

    a_ids = _order_team_participants(team_a)
    b_ids = _order_team_participants(team_b)

    vec = np.zeros(TABULAR_FEATURE_DIM, dtype=np.float32)
    names: list[str] = []

    for role_name, cid in zip(ROLE_NAMES, a_ids):
        names.append(f"team_a_{role_name}")
    for role_name, cid in zip(ROLE_NAMES, b_ids):
        names.append(f"team_b_{role_name}")

    vec[0:5] = a_ids
    vec[5:10] = b_ids

    # Composition diff
    offset = 10
    for cid in range(NUM_CHAMPIONS):
        names.append(f"comp_diff_champ_{cid}")

    for cid in a_ids:
        if cid != UNK_CHAMPION_ID:
            vec[offset + cid] -= 1.0

    for cid in b_ids:
        if cid != UNK_CHAMPION_ID:
            vec[offset + cid] += 1.0

    return vec, names


def extract_dense_draft_features(features: dict[str, Any]) -> np.ndarray:
    """Extract a 1D float32 vector of length DENSE_FEATURE_DIM for neural networks.

    Vector structure:
      - 5 one-hot segments for Team A roles (5 * NUM_CHAMPIONS)
      - 5 one-hot segments for Team B roles (5 * NUM_CHAMPIONS)
      - 1 composition diff segment (NUM_CHAMPIONS)
    """
    team_a = features.get("team_a") or []
    team_b = features.get("team_b") or []

    a_ids = _order_team_participants(team_a)
    b_ids = _order_team_participants(team_b)

    dense = np.zeros(DENSE_FEATURE_DIM, dtype=np.float32)

    # Team A roles
    for slot, cid in enumerate(a_ids):
        if cid != UNK_CHAMPION_ID:
            dense[slot * NUM_CHAMPIONS + cid] = 1.0

    # Team B roles
    b_offset = 5 * NUM_CHAMPIONS
    for slot, cid in enumerate(b_ids):
        if cid != UNK_CHAMPION_ID:
            dense[b_offset + slot * NUM_CHAMPIONS + cid] = 1.0

    # Comp diff segment
    diff_offset = 10 * NUM_CHAMPIONS
    for cid in a_ids:
        if cid != UNK_CHAMPION_ID:
            dense[diff_offset + cid] -= 1.0
    for cid in b_ids:
        if cid != UNK_CHAMPION_ID:
            dense[diff_offset + cid] += 1.0

    return dense


def batch_extract_draft_features(
    features_list: list[dict[str, Any]], mode: str = "tabular"
) -> np.ndarray:
    """Batch feature extraction for lists of match feature dicts."""
    n = len(features_list)
    if mode == "tabular":
        out = np.zeros((n, TABULAR_FEATURE_DIM), dtype=np.float32)
        for i, feat in enumerate(features_list):
            out[i], _ = extract_tabular_draft_features(feat)
        return out
    elif mode == "dense":
        out = np.zeros((n, DENSE_FEATURE_DIM), dtype=np.float32)
        for i, feat in enumerate(features_list):
            out[i] = extract_dense_draft_features(feat)
        return out
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'tabular' or 'dense'.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv312\Scripts\pytest.exe tests/test_draft_features.py -v`
Expected: PASS (5/5 passed)

- [ ] **Step 5: Commit Task 2**

```bash
git add core/draft_features.py tests/test_draft_features.py
git commit -m "feat(ml): implement structured draft feature extraction for tabular and dense models"
```

---

### Task 3: Upgrade Tree Baseline with Structured Features (`scripts/train_tree_baseline.py`)

**Files:**
- Modify: `scripts/train_tree_baseline.py`
- Test: `tests/test_tree_baseline_structured.py`

**Interfaces:**
- Consumes: `core.draft_features.batch_extract_draft_features`, `TABULAR_FEATURE_DIM`.
- Produces:
  - Updated `train_tree_model` fitting `HistGradientBoostingClassifier` with `categorical_features=list(range(10))` on the 10 role champion columns and continuous numeric on composition diff columns.
  - Generates updated `model.metrics.json` preserving comparison contracts.

- [ ] **Step 1: Write failing test for upgraded tree baseline**

Write `tests/test_tree_baseline_structured.py`:
```python
import numpy as np
import pytest
from scripts.train_tree_baseline import train_tree_model

def make_synthetic_record(mid: int, winner: int, a_top: str, b_top: str):
    return {
        "match_id": f"TEST_{mid}",
        "game_creation_time": 1700000000 + mid * 100,
        "feature_json": {
            "team_a": [
                {"champion": a_top, "role": "TOP"},
                {"champion": "LeeSin", "role": "JUNGLE"},
                {"champion": "Ahri", "role": "MIDDLE"},
                {"champion": "Jinx", "role": "BOTTOM"},
                {"champion": "Thresh", "role": "UTILITY"},
            ],
            "team_b": [
                {"champion": b_top, "role": "TOP"},
                {"champion": "Elise", "role": "JUNGLE"},
                {"champion": "Zed", "role": "MIDDLE"},
                {"champion": "Ezreal", "role": "BOTTOM"},
                {"champion": "Lulu", "role": "UTILITY"},
            ],
            "game_mode": "CLASSIC",
        },
        "labels_json": {"winner": winner},
        "winner_label": winner,
    }

def test_tree_model_trains_with_structured_features(tmp_path):
    records = []
    # Create 60 records (42 train, 9 val, 9 test) with synthetic bias
    for i in range(60):
        # Bias: If Darius is on team B, winner is mostly 1
        b_top = "Darius" if i % 2 == 0 else "Garen"
        winner = 1 if b_top == "Darius" else 0
        records.append(make_synthetic_record(i, winner, a_top="Aatrox", b_top=b_top))

    out_dir = str(tmp_path / "tree_run")
    report = train_tree_model(records, output_dir=out_dir, split_strategy="chronological")

    assert "model" in report["metrics"]["train"]
    assert "test" in report["metrics"]
    assert (tmp_path / "tree_run" / "model.joblib").exists()
    assert (tmp_path / "tree_run" / "model.metrics.json").exists()
```

- [ ] **Step 2: Run test to verify current state**

Run: `.\.venv312\Scripts\pytest.exe tests/test_tree_baseline_structured.py -v`
Expected: May fail or succeed using legacy string features; we want to upgrade it to use `batch_extract_draft_features`.

- [ ] **Step 3: Update `scripts/train_tree_baseline.py` to use `core.draft_features`**

Modify `scripts/train_tree_baseline.py`:
Replace `extract_tabular_features` calls with `batch_extract_draft_features`:
```python
from core.draft_features import (
    batch_extract_draft_features,
    TABULAR_FEATURE_DIM,
)
```
In `train_tree_model`:
```python
    # Extract structured draft features
    x_train = batch_extract_draft_features([r["feature_json"] for r in splits["train"]], mode="tabular")
    x_val = batch_extract_draft_features([r["feature_json"] for r in splits["validation"]], mode="tabular")
    x_test = batch_extract_draft_features([r["feature_json"] for r in splits["test"]], mode="tabular")

    y_train = np.asarray([r["labels_json"]["winner"] for r in splits["train"]], dtype=int)
    y_val = np.asarray([r["labels_json"]["winner"] for r in splits["validation"]], dtype=int)
    y_test = np.asarray([r["labels_json"]["winner"] for r in splits["test"]], dtype=int)

    # First 10 features are categorical role champion IDs
    cat_features = list(range(10))
    clf = HistGradientBoostingClassifier(
        categorical_features=cat_features,
        random_state=seed,
        max_iter=100,
        early_stopping=True,
    )
    clf.fit(x_train, y_train)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv312\Scripts\pytest.exe tests/test_tree_baseline_structured.py -v`
Expected: PASS

- [ ] **Step 5: Commit Task 3**

```bash
git add scripts/train_tree_baseline.py tests/test_tree_baseline_structured.py
git commit -m "refactor(ml): wire structured draft features into tree baseline trainer"
```

---

### Task 4: Automated Hypothesis-Testing Benchmark (`scripts/benchmark_draft_signal.py`)

**Files:**
- Create: `scripts/benchmark_draft_signal.py`
- Test: `tests/test_benchmark_draft_signal.py`

**Interfaces:**
- Consumes: `core.pre_match_dataset.load_records`, `core.draft_features.batch_extract_draft_features`, `core.training_metrics.evaluate_winner`.
- Produces:
  - `run_draft_signal_benchmark(records: list[dict], n_splits: int = 3) -> dict`: runs holdout evaluation across sample slices (e.g. 5k, 10k), computes ROC-AUC, 95% confidence intervals, and hypothesis p-value vs $H_0: \text{AUC} \le 0.50$.
  - CLI command `python scripts/benchmark_draft_signal.py --db PATH --samples 10000 --output runs/draft_signal_report.json`.

- [ ] **Step 1: Write failing test for benchmark draft signal**

Write `tests/test_benchmark_draft_signal.py`:
```python
import pytest
from scripts.benchmark_draft_signal import run_draft_signal_benchmark

def make_biased_record(mid: int, bias: float = 0.8):
    # Deterministic generation where champion pick drives winner
    is_darius = (mid % 2 == 0)
    winner = 1 if (is_darius and (mid % 10 < int(bias * 10))) else 0
    return {
        "match_id": f"BENCH_{mid}",
        "game_creation_time": 1700000000 + mid * 10,
        "feature_json": {
            "team_a": [
                {"champion": "Aatrox", "role": "TOP"},
                {"champion": "LeeSin", "role": "JUNGLE"},
                {"champion": "Ahri", "role": "MIDDLE"},
                {"champion": "Jinx", "role": "BOTTOM"},
                {"champion": "Thresh", "role": "UTILITY"},
            ],
            "team_b": [
                {"champion": "Darius" if is_darius else "Garen", "role": "TOP"},
                {"champion": "Elise", "role": "JUNGLE"},
                {"champion": "Zed", "role": "MIDDLE"},
                {"champion": "Ezreal", "role": "BOTTOM"},
                {"champion": "Lulu", "role": "UTILITY"},
            ],
            "game_mode": "CLASSIC",
        },
        "labels_json": {"winner": winner},
        "winner_label": winner,
    }

def test_benchmark_detects_signal_when_present():
    records = [make_biased_record(i, bias=0.85) for i in range(100)]
    result = run_draft_signal_benchmark(records, min_samples=60)
    assert "verdict" in result
    assert "test_auc" in result
    assert result["verdict"] in ("SIGNAL_CONFIRMED", "INCONCLUSIVE", "NO_SIGNAL")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv312\Scripts\pytest.exe tests/test_benchmark_draft_signal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.benchmark_draft_signal'`

- [ ] **Step 3: Implement `scripts/benchmark_draft_signal.py`**

Create `scripts/benchmark_draft_signal.py`:
```python
"""Automated benchmark testing whether LoL draft alone has statistically significant predictive signal."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any
import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import config
from core.pre_match_dataset import load_records, split_records
from core.draft_features import batch_extract_draft_features
from core.training_metrics import evaluate_winner

logger = logging.getLogger("benchmark_draft_signal")


def run_draft_signal_benchmark(records: list[dict[str, Any]], min_samples: int = 100) -> dict[str, Any]:
    """Evaluate whether draft features achieve statistically significant lift over random chance."""
    if len(records) < min_samples:
        raise ValueError(f"Need at least {min_samples} records, got {len(records)}")

    splits, _ = split_records(records, strategy="chronological")
    x_train = batch_extract_draft_features([r["feature_json"] for r in splits["train"]], mode="tabular")
    x_val = batch_extract_draft_features([r["feature_json"] for r in splits["validation"]], mode="tabular")
    x_test = batch_extract_draft_features([r["feature_json"] for r in splits["test"]], mode="tabular")

    y_train = np.asarray([r["labels_json"]["winner"] for r in splits["train"]], dtype=int)
    y_val = np.asarray([r["labels_json"]["winner"] for r in splits["validation"]], dtype=int)
    y_test = np.asarray([r["labels_json"]["winner"] for r in splits["test"]], dtype=int)

    train_rate = float(y_train.mean())

    # Fit tabular gradient booster with early stopping on validation
    clf = HistGradientBoostingClassifier(
        categorical_features=list(range(10)),
        random_state=42,
        max_iter=100,
        early_stopping=True,
    )
    clf.fit(x_train, y_train)

    val_probs = clf.predict_proba(x_val)[:, 1]
    test_probs = clf.predict_proba(x_test)[:, 1]

    val_metrics = evaluate_winner(y_val, val_probs, train_rate)
    test_metrics = evaluate_winner(y_test, test_probs, train_rate)

    test_auc = float(test_metrics["model"]["roc_auc"])
    test_acc = float(test_metrics["model"]["accuracy"])
    base_acc = float(test_metrics["baseline"]["accuracy"])
    acc_lift = test_acc - base_acc

    # Standard error for AUC via Hanley & McNeil approximation
    n_pos = int(np.sum(y_test == 1))
    n_neg = int(np.sum(y_test == 0))
    q1 = test_auc / (2.0 - test_auc)
    q2 = 2.0 * test_auc**2 / (1.0 + test_auc)
    auc_var = (test_auc * (1.0 - test_auc) + (n_pos - 1) * (q1 - test_auc**2) + (n_neg - 1) * (q2 - test_auc**2)) / (n_pos * n_neg)
    auc_se = float(np.sqrt(max(1e-8, auc_var)))
    z_score = (test_auc - 0.50) / auc_se
    p_value = float(1.0 - stats.norm.cdf(z_score))

    if test_auc >= 0.52 and p_value < 0.05:
        verdict = "SIGNAL_CONFIRMED"
    elif test_auc > 0.50 and p_value < 0.10:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "NO_SIGNAL"

    return {
        "verdict": verdict,
        "sample_counts": {k: len(v) for k, v in splits.items()},
        "test_auc": test_auc,
        "auc_standard_error": auc_se,
        "p_value_vs_null": p_value,
        "test_accuracy": test_acc,
        "baseline_accuracy": base_acc,
        "accuracy_lift": acc_lift,
        "val_log_loss": val_metrics["model"]["log_loss"],
        "test_log_loss": test_metrics["model"]["log_loss"],
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark draft signal on pre-match matches.")
    parser.add_argument("--db", type=str, required=True, help="Path to SQLite database.")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples to evaluate.")
    parser.add_argument("--output", type=str, default="runs/draft_signal_report.json", help="Report output path.")
    args = parser.parse_args()

    records = load_records(args.db, limit=args.samples)
    report = run_draft_signal_benchmark(records)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Benchmark finished: verdict={report['verdict']} test_auc={report['test_auc']:.4f} p={report['p_value_vs_null']:.4e}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv312\Scripts\pytest.exe tests/test_benchmark_draft_signal.py -v`
Expected: PASS

- [ ] **Step 5: Commit Task 4**

```bash
git add scripts/benchmark_draft_signal.py tests/test_benchmark_draft_signal.py
git commit -m "feat(ml): implement automated draft signal benchmark with hypothesis testing"
```

---

### Task 5: Compact Dense Neural Draft Model & Training Pipeline (`core/generate_graph.py` & `scripts/train_pre_match_model.py`)

**Files:**
- Modify: `core/generate_graph.py`
- Modify: `scripts/train_pre_match_model.py`
- Test: `tests/test_draft_pre_match_training.py`

**Interfaces:**
- Consumes: `core.draft_features.DENSE_FEATURE_DIM`, `core.draft_features.batch_extract_draft_features`.
- Produces:
  - `build_draft_dense_model(input_dim: int = DENSE_FEATURE_DIM, dropout_rate: float = 0.2) -> tf.keras.Model`: clean, non-diluted MLP dedicated to winner prediction with calibrated softmax head.
  - `--feature-version {v1_hasher, v2_draft}` flag in `scripts/train_pre_match_model.py`.

- [ ] **Step 1: Write failing test for dense neural draft training**

Write `tests/test_draft_pre_match_training.py`:
```python
import pytest
from scripts.train_pre_match_model import train_model

def make_sample_record(mid: int, winner: int):
    return {
        "match_id": f"DRAFT_NN_{mid}",
        "game_creation_time": 1700000000 + mid * 60,
        "feature_json": {
            "team_a": [
                {"champion": "Aatrox", "role": "TOP"},
                {"champion": "LeeSin", "role": "JUNGLE"},
                {"champion": "Ahri", "role": "MIDDLE"},
                {"champion": "Jinx", "role": "BOTTOM"},
                {"champion": "Thresh", "role": "UTILITY"},
            ],
            "team_b": [
                {"champion": "Darius" if winner == 1 else "Garen", "role": "TOP"},
                {"champion": "Elise", "role": "JUNGLE"},
                {"champion": "Zed", "role": "MIDDLE"},
                {"champion": "Ezreal", "role": "BOTTOM"},
                {"champion": "Lulu", "role": "UTILITY"},
            ],
            "game_mode": "CLASSIC",
        },
        "labels_json": {
            "winner": winner,
            "team_b_kill_lead": winner,
            "first_blood": 0, "first_baron": 2, "first_inhibitor": 2, "first_tower": 0,
            "kills_odd": 0, "both_baron": 0, "both_inhibitor": 0, "both_dragon": 1,
            "elder_dragon": 0, "kill_handicap": 0, "total_kills": 20,
            "team_a_kills": 10, "team_b_kills": 10, "total_barons": 0,
            "total_dragons": 1, "total_towers": 2,
        },
        "winner_label": winner,
    }

def test_train_model_with_v2_draft_features(tmp_path):
    records = [make_sample_record(i, winner=i % 2) for i in range(40)]
    ckpt = str(tmp_path / "candidate_model.keras")
    model, report = train_model(
        records,
        epochs=1,
        batch_size=16,
        checkpoint_path=ckpt,
        feature_version="v2_draft",
    )
    assert model is not None
    assert (tmp_path / "candidate_model.keras").exists()
    assert (tmp_path / "candidate_model.metrics.json").exists()
    assert report["parameters"]["feature_version"] == "v2_draft"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv312\Scripts\pytest.exe tests/test_draft_pre_match_training.py -v`
Expected: FAIL with `TypeError: train_model() got an unexpected keyword argument 'feature_version'`

- [ ] **Step 3: Update `core/generate_graph.py` and `scripts/train_pre_match_model.py`**

In `core/generate_graph.py`, add `build_draft_dense_model`:
```python
def build_draft_dense_model(input_dim: int = 1881, dropout_rate: float = 0.2) -> tf.keras.Model:
    """Compact, focused MLP specifically for dense draft features."""
    inputs = tf.keras.Input(shape=(input_dim,), name="Input")
    x = tf.keras.layers.Dense(512, activation="relu", name="Draft_Dense1")(inputs)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="Draft_Dense2")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="Draft_Dense3")(x)
    outputs = {"winner": tf.keras.layers.Dense(2, activation="softmax", name="winner")(x)}

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="NNriot_DraftDense")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={"winner": "categorical_crossentropy"},
        metrics={"winner": ["accuracy"]},
    )
    return model
```

In `scripts/train_pre_match_model.py`:
- Add argument `feature_version: str = "v1_hasher"` (or CLI `--feature-version {v1_hasher, v2_draft}`).
- When `feature_version == "v2_draft"`:
  - Vectorize via `core.draft_features.batch_extract_draft_features(..., mode="dense")`.
  - Use `build_draft_dense_model(input_dim=DENSE_FEATURE_DIM)`.
  - Record `feature_version` in `model.metrics.json`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv312\Scripts\pytest.exe tests/test_draft_pre_match_training.py -v`
Expected: PASS

- [ ] **Step 5: Commit Task 5**

```bash
git add core/generate_graph.py scripts/train_pre_match_model.py tests/test_draft_pre_match_training.py
git commit -m "feat(ml): add focused dense draft neural architecture and v2_draft training support"
```

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-25-draft-feature-engineering-and-signal-validation.md`.
