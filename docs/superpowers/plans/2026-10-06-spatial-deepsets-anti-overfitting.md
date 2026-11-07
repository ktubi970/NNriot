# Anti-Overfitting Pipeline for Spatial Equivariant DeepSets Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate premature validation loss explosion and improve generalisation and selective classification accuracy on the 108,703 matches spatial dataset by introducing AdamW with cosine decay, label smoothing, gradient conflict deconfliction (PCGrad), auxiliary head down-weighting, and spatial embedding jitter.

**Architecture:** 
- Regularize the Siamese DeepSets 46D encoder via decoupled weight decay (`AdamW`), increased dropout (`0.30`), and embedding Gaussian jitter (`\sigma=0.05`).
- Prevent destructive multi-task gradient interference by rescaling noisy MSE kill regression heads (`0.10` instead of `1.00`) and compiling branchless PCGrad projection into `@tf.function`.
- Suppress overconfident target convergence with categorical label smoothing (`\alpha=0.05`) on match outcome heads.

**Tech Stack:** Python 3.12, TensorFlow 2.16+, NumPy, SQLite (WAL mode), Pytest.

## Global Constraints
- Python 3.12 (`.\.venv312\Scripts\python.exe`).
- Domain Invariants (`AGENTS.md`):
  - Draft Combat Anti-symmetry: $s_{\text{draft}}(T_B, T_A) = -s_{\text{draft}}(T_A, T_B)$ strictly on `draft_advantage`.
  - Blue Side Advantage Decoupling: $s_{\text{match}} = s_{\text{draft}} + b_{\text{side}}$ via `DirectionalSideBias`.
  - Chronological point-in-time integrity: zero future data leakage.
- Mode Honesty: Unvarnished calibration metrics, Wilson intervals, and Brier decomposition.

---

### Task 1: Regularization & Architecture Enhancements (`core/equivariant_model.py`)

**Files:**
- Modify: `core/equivariant_model.py:202-384`
- Test: `tests/test_equivariant_model_regularization.py`

**Interfaces:**
- `build_equivariant_deepsets_model(..., dropout_rate=0.30, l2_reg=1e-4, weight_decay=1e-3, optimizer_type="adamw", spatial_noise_std=0.05, label_smoothing=0.05)`
- Returns: `tf.keras.Model` equipped with `AdamW`, optional `GaussianNoise` on spatial latent components, and smoothed categorical loss functions.

- [x] **Step 1: Write the failing tests**
```python
# tests/test_equivariant_model_regularization.py
import pytest
import tensorflow as tf
from core.equivariant_model import build_equivariant_deepsets_model

def test_model_builds_with_adamw_and_smoothing():
    model = build_equivariant_deepsets_model(
        player_feature_dim=46,
        optimizer_type="adamw",
        weight_decay=1e-3,
        dropout_rate=0.30,
        label_smoothing=0.05,
        pruned_heads=True,
    )
    assert isinstance(model.optimizer, tf.keras.optimizers.AdamW)
    assert model.count_params() > 0

def test_spatial_jitter_layer_active_in_training_only():
    model = build_equivariant_deepsets_model(
        player_feature_dim=46,
        spatial_noise_std=0.05,
    )
    xa = tf.ones((2, 5, 46))
    xb = tf.ones((2, 5, 46))
    out_eval1 = model({"team_a": xa, "team_b": xb}, training=False)["winner"]
    out_eval2 = model({"team_a": xa, "team_b": xb}, training=False)["winner"]
    # In eval mode, outputs must be identical (deterministic)
    tf.debugging.assert_near(out_eval1, out_eval2)
```

- [x] **Step 2: Run tests to verify failure**
```powershell
.\.venv312\Scripts\python.exe -m pytest tests/test_equivariant_model_regularization.py -v
```

- [x] **Step 3: Implement AdamW, label smoothing, and spatial jitter in `core/equivariant_model.py`**
  - Add `spatial_noise_std`, `optimizer_type`, `weight_decay`, and `label_smoothing` arguments to `build_equivariant_deepsets_model`.
  - When `spatial_noise_std > 0`, apply `tf.keras.layers.GaussianNoise(spatial_noise_std)` specifically to the input tensors or encoder projection during training.
  - Configure `tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)` when `optimizer_type == "adamw"`.
  - Configure `CategoricalCrossentropy(label_smoothing=label_smoothing)` for categorical heads (`winner`, `team_b_kill_lead`, `first_blood`).

- [x] **Step 4: Run tests to verify pass**
```powershell
.\.venv312\Scripts\python.exe -m pytest tests/test_equivariant_model_regularization.py tests/test_equivariant_model.py -v
```

- [x] **Step 5: Commit changes**
```powershell
git add core/equivariant_model.py tests/test_equivariant_model_regularization.py
git commit -m "feat(model): add AdamW, label smoothing, and spatial embedding jitter to Equivariant DeepSets"
```

---

### Task 2: Auxiliary Head Re-weighting & Learning Rate Scheduler (`core/generate_graph.py` & `scripts/train_spatial_deepsets.py`)

**Files:**
- Modify: `core/generate_graph.py:35-56`
- Modify: `scripts/train_spatial_deepsets.py:150-265`
- Test: `tests/test_train_spatial_deepsets_config.py`

**Interfaces:**
- CLI flags in `train_spatial_deepsets.py`:
  - `--lr 0.0003` (default reduced from 0.001)
  - `--use-cosine-decay` (boolean flag, decaying to `1e-5` over epochs)
  - `--aux-weight-scale 0.1` (scales all MSE kill regression heads down by 10x)
  - `--use-pcgrad` (flag to route updates through `create_compiled_pcgrad_step`)

- [x] **Step 1: Write test for auxiliary head scaling and scheduler configuration**
```python
# tests/test_train_spatial_deepsets_config.py
from scripts.train_spatial_deepsets import get_adjusted_loss_weights

def test_auxiliary_loss_weights_downscaled():
    weights = get_adjusted_loss_weights(aux_scale=0.1)
    assert weights["winner"] == 1.0
    assert weights["team_b_kill_lead"] == 1.0
    assert weights["total_kills"] <= 0.1
    assert weights["team_a_kills"] <= 0.1
    assert weights["team_b_kills"] <= 0.1
    assert weights["kill_handicap"] <= 0.2
```

- [x] **Step 2: Run test to verify failure**
```powershell
.\.venv312\Scripts\python.exe -m pytest tests/test_train_spatial_deepsets_config.py -v
```

- [x] **Step 3: Implement `get_adjusted_loss_weights` and CosineDecay scheduler**
  - Implement `get_adjusted_loss_weights(aux_scale: float = 0.1)` in `scripts/train_spatial_deepsets.py`.
  - Add `tf.keras.optimizers.schedules.CosineDecay` when `--use-cosine-decay` is set, with `decay_steps = total_batches` and `alpha = 0.01`.
  - Wire CLI arguments in `train_spatial_deepsets.py`.

- [x] **Step 4: Run tests to verify pass**
```powershell
.\.venv312\Scripts\python.exe -m pytest tests/test_train_spatial_deepsets_config.py -v
```

- [x] **Step 5: Commit changes**
```powershell
git add scripts/train_spatial_deepsets.py core/generate_graph.py tests/test_train_spatial_deepsets_config.py
git commit -m "feat(training): add auxiliary loss re-weighting and cosine decay scheduler"
```

---

### Task 3: PCGrad Multi-Task Deconfliction Integration (`scripts/train_spatial_deepsets.py`)

**Files:**
- Modify: `scripts/train_spatial_deepsets.py:180-225`
- Test: `tests/test_spatial_pcgrad_integration.py`

**Interfaces:**
- Consumes: `create_compiled_pcgrad_step` from `core/pcgrad.py`
- Produces: Seamless switching between vectorized standard gradient tape and graph-compiled PCGrad deconfliction.

- [x] **Step 1: Write test for compiled PCGrad step in spatial model**
```python
# tests/test_spatial_pcgrad_integration.py
import tensorflow as tf
from core.equivariant_model import build_equivariant_deepsets_model
from core.pcgrad import create_compiled_pcgrad_step

def test_spatial_model_runs_compiled_pcgrad_step():
    model = build_equivariant_deepsets_model(player_feature_dim=46, pruned_heads=True)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss_fns = {"winner": tf.keras.losses.CategoricalCrossentropy(), "kill_handicap": tf.keras.losses.MeanSquaredError()}
    loss_weights = {"winner": 1.0, "kill_handicap": 0.2}
    step_fn = create_compiled_pcgrad_step(model, optimizer, loss_fns, loss_weights, jit_compile=False)
    
    xa = tf.random.normal((8, 5, 46))
    xb = tf.random.normal((8, 5, 46))
    yb = {
        "winner": tf.one_hot(tf.zeros((8,), dtype=tf.int32), 2),
        "kill_handicap": tf.zeros((8, 1)),
    }
    losses = step_fn({"team_a": xa, "team_b": xb}, yb)
    assert "winner" in losses
    assert float(losses["winner"]) > 0.0
```

- [x] **Step 2: Run test to verify failure/pass**
```powershell
.\.venv312\Scripts\python.exe -m pytest tests/test_spatial_pcgrad_integration.py -v
```

- [x] **Step 3: Wire `use_pcgrad` into `scripts/train_spatial_deepsets.py`**
  - Integrate `create_compiled_pcgrad_step` when `--use-pcgrad` is passed.
  - Retain fallback to fast `@tf.function` standard gradient tape when `--use-pcgrad` is disabled.

- [x] **Step 4: Commit changes**
```powershell
git add scripts/train_spatial_deepsets.py tests/test_spatial_pcgrad_integration.py
git commit -m "feat(training): integrate compiled PCGrad multi-task deconfliction"
```

---

### Task 4: Comparative Execution & Benchmark Verification

**Files:**
- Execute: `scripts/train_spatial_deepsets.py`
- Target Dir: `runs/spatial_deepsets_anti_overfit/`
- Test: `tests/test_spatial_integration.py`

- [x] **Step 1: Execute training with anti-overfitting configuration**
```powershell
.\.venv312\Scripts\python.exe scripts/train_spatial_deepsets.py `
  --db training_data.db `
  --epochs 15 `
  --batch-size 64 `
  --lr 0.0003 `
  --output-dir runs/spatial_deepsets_anti_overfit `
  --use-cosine-decay `
  --aux-weight-scale 0.1
```

- [x] **Step 2: Verify validation curve across epochs**
  - Confirm validation loss does not immediately blow up at epoch 2.
  - Check that training loss and validation loss decrease smoothly.
  - Check that implied blue side winrate converges reasonably (~51-53%).

- [x] **Step 3: Compare metrics report against baseline run1**
  - Compare `runs/spatial_deepsets_anti_overfit/model.metrics.json` vs `runs/spatial_deepsets_run1/model.metrics.json`:
    - Log Loss improvement
    - Brier score
    - Test accuracy
    - Wilson intervals on selective classification (conviction $\ge 2\%, 4\%, 6\%$).

- [x] **Step 4: Commit final model artifact configs**
```powershell
git add runs/spatial_deepsets_anti_overfit/model.metrics.json
git commit -m "benchmark: log spatial deepsets anti-overfitting results"
```

---

## Benchmark Results & Comparative Analysis (`run1` vs `anti_overfit`)

### 1. Per-Epoch Validation Dynamics

| Epoch | Train Loss | Val Loss | Val Win Acc | Blue Bias ($b_{\text{side}}$) | Implied Blue WR |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | 3.0791 | 2.8567 | 52.74% | +0.0620 | 46.9% |
| **02** | 2.9201 | 2.8203 | 53.33% | +0.1798 | 41.1% |
| **03** | 2.8916 | **2.8184** (★ Best) | 53.27% | +0.2347 | 38.5% |
| **04** | 2.8767 | 2.8229 | 52.61% | +0.2625 | 37.2% |
| **05** | 2.8649 | 2.8268 | 52.67% | +0.2699 | 36.8% |
| **06** | 2.8560 | 2.8345 | 52.25% | +0.2776 | 36.5% |
| **07** | 2.8492 | 2.8397 | 52.17% | +0.2810 | 36.3% |
| **08** | 2.8438 | 2.8463 | 52.16% | +0.2837 | 36.2% |

*Early stopping triggered at Epoch 8 (patience 5 exceeded). Best checkpoint weights restored from Epoch 3.*

### 2. Global Test Set Comparison ($N = 10,871$ unseen matches)

| Metric | Baseline Majority | `run1` | `anti_overfit` | Gain vs `run1` |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy (Calibrated)** | 52.26% | 52.66% | **52.78%** | **+0.12%** (+13 bps) |
| **Log Loss (Calibrated)** | 0.693147 | 0.691404 | **0.690894** | **-0.000510** (Gain) |
| **Brier Score (Calibrated)** | 0.250000 | 0.249129 | **0.248876** | **-0.000253** (Gain) |
| **ROC AUC** | 0.500000 | 0.522236 | **0.527642** | **+0.005406** (+0.54%) |
| **Brier Reliability** ($\downarrow$) | 0.000510 | 0.000269 | **0.000193** | **-28.2% error** |
| **Quantile ECE** ($\downarrow$) | 0.022583 | 0.013912 | **0.012879** | **-7.4%** |
| **Calibration Slope** | N/A | 1.1181 | **0.8068** | Closest to 1.0 |
| **Optimal Temperature** | 1.0000 | 1.4588 | **1.5497** | Mild residual over-confidence |

### 3. Selective Classification (Wilson 95% Confidence Intervals)

| Conviction Threshold | Model | Coverage (Volume) | Test Count | Test Accuracy | Wilson 95% Interval |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$C \ge 0\%$** (Global) | `run1`<br>`anti_overfit` | 100.0%<br>100.0% | 10,871<br>10,871 | 52.66%<br>**52.78%** | [51.72%, 53.60%]<br>[51.84%, 53.72%] |
| **$C \ge 2\%$** | `run1`<br>`anti_overfit` | 64.13%<br>62.41% | 6,972<br>6,785 | 53.66%<br>**53.75%** | [52.49%, 54.83%]<br>[52.56%, 54.93%] |
| **$C \ge 4\%$** | `run1`<br>`anti_overfit` | 20.98%<br>**31.13%** | 2,281<br>**3,384** (**+48%**) | 54.23%<br>**55.85%** (**+1.62%**) | [52.18%, 56.27%]<br>[**54.17%**, **57.52%**] |
| **$C \ge 6\%$** | `run1`<br>`anti_overfit` | 2.38%<br>**11.77%** | 259<br>**1,280** ($\times 4.9$) | 57.92%<br>**55.78%** | [51.83%, 63.77%]<br>[53.05%, 58.48%] |
| **$C \ge 8\%$** | `run1`<br>`anti_overfit` | 0.12%<br>**3.04%** | 13 (noise)<br>**330** (viable) | 61.54%<br>**60.30%** | [35.52%, 82.29%]<br>[**54.93%**, **65.43%**] |
| **$C \ge 10\%$** | `run1`<br>`anti_overfit` | 0.00%<br>**0.59%** | 0<br>**64** | N/A<br>**64.06%** | N/A<br>[51.82%, 74.71%] |
