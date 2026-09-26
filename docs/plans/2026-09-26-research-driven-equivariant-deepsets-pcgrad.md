# Equivariant Siamese DeepSets & PCGrad Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an Equivariant Siamese DeepSets architecture with Projecting Conflicting Gradients (PCGrad) deconfliction to enforce mathematical team-swap anti-symmetry and eliminate multi-task gradient interference in NNriot.

**Architecture:** 
1. Replace flat dense MLPs with a Siamese DeepSets encoder that embeds 10 participants (champion ID + role + Bayesian priors), aggregates each 5-player team via permutation-invariant pooling $\sum \phi(x_i)$, and computes an anti-symmetric difference embedding $Z_{\text{diff}} = T_B - T_A$ and symmetric sum embedding $Z_{\text{sum}} = T_A + T_B$.
2. Implement PCGrad gradient projection during training to eliminate negative transfer from secondary regression and noisy heads onto the primary winner logit.
3. Decouple heads hierarchically into core match outcomes and auxiliary markets.

**Tech Stack:** Python 3.10+, TensorFlow / Keras 2.x, NumPy, SQLite, PyTest.

## Academic Literature Foundations
- **Equivariance & Set Invariance:** Zaheer et al. (NIPS 2017) *Deep Sets* ([arXiv:1703.06114](https://arxiv.org/abs/1703.06114)); Lee et al. (ICML 2019) *Set Transformer* ([arXiv:1810.00825](https://arxiv.org/abs/1810.00825)).
- **Multi-Task Deconfliction:** Yu et al. (NeurIPS 2020) *Gradient Surgery for Multi-Task Learning (PCGrad)*; Shi & Li (2023) *Recon: Reducing Conflicting Gradients* ([arXiv:2302.11289](https://arxiv.org/abs/2302.11289)).
- **Esports Skill & Graph Modeling:** De Bois & Parmentier (2025) *PandaSkill* ([arXiv:2501.10049](https://arxiv.org/abs/2501.10049)); Bisberg & Ferrara (2022) *GCN-WP* ([arXiv:2207.13191](https://arxiv.org/abs/2207.13191)).

## Global Constraints
- Must run cleanly on CPU (NumPy / TensorFlow CPU without CUDA requirements).
- Zero breaking changes to existing SQLite schema (version 12) or database interfaces.
- The model must strictly enforce team-swap anti-symmetry: $P(\text{winner}=1 \mid B, A) = 1 - P(\text{winner}=1 \mid A, B)$ up to machine precision ($\epsilon < 10^{-6}$).
- Candidate evaluation must follow the existing chronological holdout and validation selection protocol.

---

### Task 1: Permutation-Invariant & Siamese DeepSets Layers

**Files:**
- Create: `core/deepsets_layers.py`
- Test: `tests/test_deepsets_layers.py`

**Interfaces:**
- Consumes: `champion_registry.NUM_CHAMPIONS` (int), `player_priors.PLAYER_PRIOR_FEATURE_COUNT` (int).
- Produces: `SiameseTeamEncoder(tf.keras.layers.Layer)`: takes team tensors of shape `(batch, 5, feature_dim)` and outputs `(T_diff, T_sum)` representations where $T_{\text{diff}}(B, A) = -T_{\text{diff}}(A, B)$.

- [x] **Step 1: Write failing tests for Siamese anti-symmetry and permutation invariance**

```python
# tests/test_deepsets_layers.py
import pytest
import numpy as np
import tensorflow as tf
from core.deepsets_layers import SiameseTeamEncoder

def test_permutation_invariance_within_team():
    layer = SiameseTeamEncoder(embed_dim=32, latent_dim=64)
    team_a = tf.random.normal((2, 5, 10))
    team_b = tf.random.normal((2, 5, 10))
    
    diff1, sum1 = layer((team_a, team_b))
    
    # Permute order of players in team_a (e.g. swap slot 0 and 3)
    perm = [3, 1, 2, 0, 4]
    team_a_perm = tf.gather(team_a, perm, axis=1)
    diff2, sum2 = layer((team_a_perm, team_b))
    
    np.testing.assert_allclose(diff1.numpy(), diff2.numpy(), atol=1e-5)
    np.testing.assert_allclose(sum1.numpy(), sum2.numpy(), atol=1e-5)

def test_siamese_team_swap_antisymmetry():
    layer = SiameseTeamEncoder(embed_dim=32, latent_dim=64)
    team_a = tf.random.normal((2, 5, 10))
    team_b = tf.random.normal((2, 5, 10))
    
    diff_ab, sum_ab = layer((team_a, team_b))
    diff_ba, sum_ba = layer((team_b, team_a))
    
    # T_diff must be anti-symmetric: diff(B, A) == -diff(A, B)
    np.testing.assert_allclose(diff_ba.numpy(), -diff_ab.numpy(), atol=1e-5)
    # T_sum must be symmetric: sum(B, A) == sum(A, B)
    np.testing.assert_allclose(sum_ba.numpy(), sum_ab.numpy(), atol=1e-5)
```

- [x] **Step 2: Run pytest to confirm failure**
```powershell
python -m pytest tests/test_deepsets_layers.py
```

- [x] **Step 3: Implement `SiameseTeamEncoder` in `core/deepsets_layers.py`**
- Implement per-entity MLP $\phi(x)$ shared across all 10 slots.
- Sum pooling $\sum_{i=1}^5 \phi(x_i)$ for Team A and Team B.
- Linear projection $\rho$ with anti-symmetric combination $T_{\text{diff}} = \rho(T_B) - \rho(T_A)$ and symmetric combination $T_{\text{sum}} = \rho(T_A) + \rho(T_B)$.

- [x] **Step 4: Run tests and verify they pass**
```powershell
python -m pytest tests/test_deepsets_layers.py
```

- [x] **Step 5: Commit**
```powershell
git add core/deepsets_layers.py tests/test_deepsets_layers.py
git commit -m "feat(deepsets): add SiameseTeamEncoder with strict anti-symmetry and permutation invariance"
```

---

### Task 2: Equivariant Multi-Task Model Factory

**Files:**
- Create: `core/equivariant_model.py`
- Modify: `core/model_factory.py`
- Test: `tests/test_equivariant_model.py`

**Interfaces:**
- Consumes: `SiameseTeamEncoder` from Task 1, `core.feature_labels.ALL_LABEL_KEYS`.
- Produces: `build_equivariant_deepsets_model(embed_dim=32, latent_dim=64) -> tf.keras.Model`: Keras multi-output model taking `(team_a_inputs, team_b_inputs)` and yielding all multi-task heads with anti-symmetric winner logits.

- [x] **Step 1: Write failing test verifying exact win-probability reflection under team swap**

```python
# tests/test_equivariant_model.py
import pytest
import numpy as np
import tensorflow as tf
from core.equivariant_model import build_equivariant_deepsets_model

def test_winner_probability_inversion_under_team_swap():
    model = build_equivariant_deepsets_model(embed_dim=16, latent_dim=32)
    
    # 2 matches, 5 players, champ_id(1) + role(1) + priors(6) = 8 inputs
    team_a = tf.constant(np.random.randint(0, 100, size=(2, 5, 8)), dtype=tf.float32)
    team_b = tf.constant(np.random.randint(0, 100, size=(2, 5, 8)), dtype=tf.float32)
    
    pred_ab = model({"team_a": team_a, "team_b": team_b})
    pred_ba = model({"team_a": team_b, "team_b": team_a})
    
    win_ab = pred_ab["winner"].numpy() # shape (2, 2), [P(team_a_wins), P(team_b_wins)]
    win_ba = pred_ba["winner"].numpy()
    
    # P(Team B wins when B is team_b) == P(Team A wins when B is team_a)
    np.testing.assert_allclose(win_ab[:, 1], win_ba[:, 0], atol=1e-5)
    np.testing.assert_allclose(win_ab[:, 0], win_ba[:, 1], atol=1e-5)
```

- [x] **Step 2: Run pytest to confirm failure**
```powershell
python -m pytest tests/test_equivariant_model.py
```

- [x] **Step 3: Implement `build_equivariant_deepsets_model`**
- Input layers: `team_a: (None, 5, 8)`, `team_b: (None, 5, 8)`.
- Pass through `SiameseTeamEncoder`.
- Route $T_{\text{diff}}$ to directional heads (`winner`, `team_b_kill_lead`, `kill_handicap`, `first_*`).
- Route $T_{\text{sum}}$ to symmetric aggregate heads (`total_kills`, `total_towers`, `total_dragons`, `total_barons`).
- Note: Winner head uses single odd logit $s(T_{\text{diff}})$ where $s(-T_{\text{diff}}) = -s(T_{\text{diff}})$, outputting softmax $[ \sigma(-s), \sigma(s) ]$.

- [x] **Step 4: Run tests to verify**
```powershell
python -m pytest tests/test_equivariant_model.py
```

- [x] **Step 5: Commit**
```powershell
git add core/equivariant_model.py tests/test_equivariant_model.py
git commit -m "feat(model): build equivariant multi-output DeepSets architecture"
```

---

### Task 3: PCGrad (Projecting Conflicting Gradients) Optimizer Wrapper

**Files:**
- Create: `core/pcgrad.py`
- Test: `tests/test_pcgrad.py`

**Interfaces:**
- Consumes: TensorFlow gradient computations across task loss functions.
- Produces: `pcgrad_step(model, optimizer, x_batch, target_dict, task_weights=None) -> dict[str, float]`: computes gradients for each active head independently, projects conflicting gradients ($g_i \cdot g_j < 0 \implies g_i = g_i - \frac{g_i \cdot g_j}{\|g_j\|^2} g_j$), and applies the deconflicted sum to model variables.

- [x] **Step 1: Write unit tests for PCGrad projection mathematics**

```python
# tests/test_pcgrad.py
import numpy as np
import tensorflow as tf
from core.pcgrad import project_conflicting_gradients

def test_project_conflicting_gradients_orthogonal():
    # g1 and g2 point in opposite directions
    g1 = tf.constant([1.0, 0.0])
    g2 = tf.constant([-0.5, 0.0]) # dot < 0
    
    deconflicted = project_conflicting_gradients([g1, g2])
    # g1 projected onto normal of g2 will have its opposing component removed
    dot = tf.reduce_sum(deconflicted[0] * deconflicted[1])
    assert dot.numpy() >= -1e-6
```

- [x] **Step 2: Run pytest to confirm failure**
```powershell
python -m pytest tests/test_pcgrad.py
```

- [x] **Step 3: Implement `project_conflicting_gradients` and `pcgrad_step` in `core/pcgrad.py`**
- Follow Yu et al. (2020) and Shi & Li (2023) algorithms.
- Vectorize deconfliction across shared model parameters.

- [x] **Step 4: Verify test suite passes**
```powershell
python -m pytest tests/test_pcgrad.py
```

- [x] **Step 5: Commit**
```powershell
git add core/pcgrad.py tests/test_pcgrad.py
git commit -m "feat(optimization): implement PCGrad gradient deconfliction wrapper"
```

---

### Task 4: Structured Participant Tensor Extraction & Dataset Feeder

**Files:**
- Create: `core/deepsets_dataset.py`
- Test: `tests/test_deepsets_dataset.py`
- Modify: `core/point_in_time_dataset.py`

**Interfaces:**
- Consumes: SQLite match participants and `PlayerPriorsTracker`.
- Produces: `extract_team_tensors(feature_json, tracker) -> tuple[np.ndarray, np.ndarray]`: outputs `(team_a, team_b)` where each tensor has shape `(5, 8)` representing `[champ_id, role_id, smooth_wr, smooth_cwr, champ_exp, rec_wr, role_fam, tot_exp]`.

- [x] **Step 1: Write failing tests for structured tensor feeder**

```python
# tests/test_deepsets_dataset.py
import pytest
import numpy as np
from core.deepsets_dataset import extract_team_tensors

def test_extract_team_tensors_shape_and_types():
    sample_feat = {
        "team_a": [{"champion": "Ahri", "role": "MIDDLE"}] * 5,
        "team_b": [{"champion": "Zed", "role": "MIDDLE"}] * 5,
    }
    ta, tb = extract_team_tensors(sample_feat)
    assert ta.shape == (5, 8)
    assert tb.shape == (5, 8)
    assert ta.dtype == np.float32
```

- [x] **Step 2: Run pytest to confirm failure**
```powershell
python -m pytest tests/test_deepsets_dataset.py
```

- [x] **Step 3: Implement `extract_team_tensors` in `core/deepsets_dataset.py`**
- Maps champion name to integer index using `champion_to_id`.
- Maps role string to canonical slot `[0..4]`.
- Pulls point-in-time player priors from `PlayerPriorsTracker`.

- [x] **Step 4: Run tests and ensure 100% pass**
```powershell
python -m pytest tests/test_deepsets_dataset.py
```

- [x] **Step 5: Commit**
```powershell
git add core/deepsets_dataset.py tests/test_deepsets_dataset.py
git commit -m "feat(dataset): extract structured team tensors for DeepSets"
```

---

### Task 5: Candidate Training Integration & Benchmark Verification

**Files:**
- Modify: `scripts/train_pre_match_model.py`
- Modify: `core/experiment.py`
- Test: `tests/test_pre_match_training.py`

**Interfaces:**
- Consumes: `--architecture {mlp,deepsets}` CLI option in `scripts/train_pre_match_model.py`.
- Produces: Complete candidate run producing `model.keras`, `model.metrics.json`, and validation report comparing baseline vs Equivariant DeepSets.

- [x] **Step 1: Add `--architecture` flag to `scripts/train_pre_match_model.py`**
- Default `--architecture deepsets` (with backward-compatible `--architecture mlp` fallback).
- Integrate `pcgrad_step` into the training epoch loop when training multi-head DeepSets.

- [x] **Step 2: Run small end-to-end integration test (100 matches, 1 epoch)**
```powershell
python scripts/train_pre_match_model.py --db runs/training_data_23h_20260922_191434.db --samples 100 --epochs 1 --batch-size 32 --architecture deepsets --output runs/pilot_deepsets/model.keras
```

- [x] **Step 3: Verify metrics report schema & anti-symmetry verification log**
- Confirm `model.metrics.json` reflects test AUC, accuracy lift, and calibration slope.
- Verify `is_anti_symmetric: true` diagnostic flag in report.

- [x] **Step 4: Run full test suite**
```powershell
python -m pytest
```

- [x] **Step 5: Commit**
```powershell
git add scripts/train_pre_match_model.py core/experiment.py tests/test_pre_match_training.py
git commit -m "feat(training): integrate Equivariant DeepSets candidate training with PCGrad"
```
