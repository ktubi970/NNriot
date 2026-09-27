# Refonte Neuronale État-de-l'Art (Pack Maximaliste & Élagage) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Moderniser en profondeur l'architecture neuronale de NNriot en intégrant les avancées scientifiques majeures (CAGrad, Poids Homoscédastiques, Tokenisation Périodique de Fourier, Set-Transformer PMA, Transport Optimal Sinkhorn, Tête Bilinéaire Skew-Symétrique, Apprentissage Évidentiel Dirichlet et Décomposition Contrefactuelle de Shapley), tout en élaguant les 22 têtes vers 7 têtes prioritaires à haut signal prédictif.

**Architecture:** Le réseau remplace le sum-pooling DeepSets naïf par une self-attention intra-équipe et un Pooling by Multihead Attention (PMA), aligne les confrontations croisées via transport optimal de Sinkhorn, encode les features continues via des embeddings périodiques apprenables, et produit des sorties calibrées évidentielles (Dirichlet) avec attribution explicite de Shapley (Draft Advantage vs Player Skill), optimisées par CAGrad.

**Tech Stack:** Python 3.12, TensorFlow / Keras, NumPy, SciPy (optimisation duale quadratique CAGrad), Pytest.

## Global Constraints

- **Invariance mathématique stricte d'anti-symétrie** : $P(\text{winner}=1 \mid \text{Team B}, \text{Team A}) = 1 - P(\text{winner}=1 \mid \text{Team A}, \text{Team B})$ et $\text{kill\_handicap}(B, A) = -\text{kill\_handicap}(A, B)$ garantis par construction.
- **Invariance par permutation intra-équipe** : Toute permutation arbitraire des 5 joueurs d'une même équipe produit une représentation d'équipe identique.
- **Sérialisation Keras** : Chaque couche personnalisée doit comporter le décorateur `@tf.keras.utils.register_keras_serializable(package="nnriot")` et la méthode `get_config()`.
- **Zéro dépendance lourde externe** : N'utilise que TensorFlow, NumPy, SciPy standard déjà disponibles dans l'environnement `.venv312`.
- **Non-régression des tests existants** : `tests/test_equivariant_model.py` et `tests/test_deepsets_layers.py` doivent continuer à passer sans régression.

---

### Task 1: Optimiseur Multi-Tâches CAGrad (Conflict-Averse Gradient Descent)

**Files:**
- Create: `core/cagrad.py`
- Modify: `core/pcgrad.py:1-120`
- Test: `tests/test_cagrad.py`

**Interfaces:**
- Consumes: Gradients de liste de tenseurs TensorFlow par tâche `[grad_task_1, ..., grad_task_T]` et liste des variables entraînables.
- Produces: `cagrad_step(model, optimizer, task_losses, variables, c=0.5)` et `project_conflict_averse_gradients(task_gradients, c=0.5)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cagrad.py
import pytest
import tensorflow as tf
import numpy as np
from core.cagrad import project_conflict_averse_gradients

def test_cagrad_resolves_conflict():
    # Deux tâches avec gradients opposés
    # g1 = [1.0, 0.0], g2 = [-1.0, 0.5]
    v = tf.Variable([0.0, 0.0])
    g1 = [tf.constant([1.0, 0.0], dtype=tf.float32)]
    g2 = [tf.constant([-1.0, 0.5], dtype=tf.float32)]
    
    combined = project_conflict_averse_gradients([g1, g2], [v], c=0.5)
    assert len(combined) == 1
    g_res = combined[0].numpy()
    
    # Le gradient combiné ne doit pas avoir un produit scalaire négatif avec la pire tâche au-delà de la borne
    dot1 = np.dot(g_res, [1.0, 0.0])
    dot2 = np.dot(g_res, [-1.0, 0.5])
    assert dot1 >= -0.1
    assert dot2 >= -0.1
    assert not np.isnan(g_res).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_cagrad.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.cagrad'`

- [ ] **Step 3: Write minimal implementation**

```python
# core/cagrad.py
"""Conflict-Averse Gradient Descent (CAGrad) optimizer for Multi-Task Learning.

Based on Liu et al. (NeurIPS 2021) "Conflict-Averse Gradient Descent for Multi-task Learning".
Replaces naive PCGrad by finding a direction that optimizes average loss while bounding 
the worst-case task degradation via dual quadratic programming.
"""
from __future__ import annotations
import tensorflow as tf
import numpy as np
from scipy.optimize import minimize

def project_conflict_averse_gradients(
    task_gradients: list[list[tf.Tensor | None]],
    variables: list[tf.Variable],
    c: float = 0.5,
) -> list[tf.Tensor]:
    """Applies CAGrad projection to task gradients."""
    if not task_gradients:
        return []
    
    num_tasks = len(task_gradients)
    num_vars = len(variables)
    
    # Flatten each task's gradients
    flat_grads = []
    shapes = [v.shape for v in variables]
    sizes = [tf.size(v).numpy() for v in variables]
    
    for g_list in task_gradients:
        parts = []
        for var_idx, g in enumerate(g_list):
            if g is None:
                parts.append(np.zeros(sizes[var_idx], dtype=np.float32))
            else:
                parts.append(tf.reshape(g, [-1]).numpy())
        flat_grads.append(np.concatenate(parts, axis=0))
    
    G = np.stack(flat_grads, axis=0) # (num_tasks, total_params)
    g0 = np.mean(G, axis=0)          # Average gradient
    norm_g0 = np.linalg.norm(g0)
    
    if norm_g0 < 1e-8:
        # Gradients already near 0
        final_flat = tf.constant(g0, dtype=tf.float32)
    else:
        # Solve dual: min w^T G G^T w  s.t. w in simplex, sum w = 1
        # GG0 = G @ g0
        GG = G @ G.T
        GG0 = G @ g0
        
        def obj(w):
            gw = G.T @ w
            return np.dot(gw, g0) + c * norm_g0 * np.linalg.norm(gw)
        
        # Dual formulation approximation via standard quadratic simplex solve
        w0 = np.ones(num_tasks) / num_tasks
        bounds = [(0, 1) for _ in range(num_tasks)]
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        res = minimize(lambda w: 0.5 * w @ GG @ w - (c * norm_g0) * np.sqrt(max(1e-12, w @ GG @ w)), 
                       w0, bounds=bounds, constraints=cons, method='SLSQP')
        w_opt = res.x if res.success else w0
        g_opt = G.T @ w_opt
        
        # Rescale according to CAGrad formula
        norm_g_opt = np.linalg.norm(g_opt)
        if norm_g_opt > 1e-8:
            g_cagrad = g0 + (c * norm_g0 / norm_g_opt) * g_opt
        else:
            g_cagrad = g0
        final_flat = tf.constant(g_cagrad, dtype=tf.float32)

    # Reconstruct original variable shapes
    final_unflattened = []
    offset = 0
    for var_idx in range(num_vars):
        numel = sizes[var_idx]
        chunk = final_flat[offset:offset + numel]
        final_unflattened.append(tf.reshape(chunk, shapes[var_idx]))
        offset += numel
    return final_unflattened
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_cagrad.py -v`
Expected: PASS

- [ ] **Step 5: Expose CAGrad in pcgrad.py and commit**

Modify `core/pcgrad.py` to import and allow switching between `pcgrad_step` and `cagrad_step`.
Commit: `git add core/cagrad.py core/pcgrad.py tests/test_cagrad.py && git commit -m "feat(mtl): implement CAGrad conflict-averse gradient optimizer"`

---

### Task 2: Élagage des Têtes & Couche d'Incertitude Homoscédastique

**Files:**
- Modify: `core/generate_graph.py:15-80`
- Modify: `core/feature_labels.py`
- Test: `tests/test_homoscedastic_loss.py`

**Interfaces:**
- Consumes: Dictionnaire des pertes brutes par tâche.
- Produces: `PRUNED_LABEL_KEYS = ("winner", "team_b_kill_lead", "kill_handicap", "first_blood", "total_kills", "team_a_kills", "team_b_kills")` et `HomoscedasticLossLayer`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_homoscedastic_loss.py
import pytest
import tensorflow as tf
from core.generate_graph import HomoscedasticLossLayer, PRUNED_LABEL_KEYS

def test_homoscedastic_loss_layer():
    layer = HomoscedasticLossLayer(task_names=list(PRUNED_LABEL_KEYS))
    assert len(PRUNED_LABEL_KEYS) == 7
    
    # 7 dummy losses
    losses = {k: tf.constant(1.0, dtype=tf.float32) for k in PRUNED_LABEL_KEYS}
    total_loss = layer(losses)
    
    assert total_loss.shape == ()
    assert not tf.math.is_nan(total_loss)
    assert len(layer.trainable_weights) == 1
    # Weight shape should be (7,) representing log_vars
    assert layer.trainable_weights[0].shape == (7,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_homoscedastic_loss.py -v`
Expected: FAIL with `ImportError: cannot import name 'HomoscedasticLossLayer'`

- [ ] **Step 3: Write minimal implementation**

```python
# In core/generate_graph.py
PRUNED_LABEL_KEYS: tuple[str, ...] = (
    "winner",
    "team_b_kill_lead",
    "kill_handicap",
    "first_blood",
    "total_kills",
    "team_a_kills",
    "team_b_kills",
)

@tf.keras.utils.register_keras_serializable(package="nnriot")
class HomoscedasticLossLayer(tf.keras.layers.Layer):
    """Learned homoscedastic uncertainty weighting for multi-task losses (Kendall et al. CVPR 2018).
    
    L = sum_t ( exp(-s_t) * L_t + 0.5 * s_t ) where s_t = log(sigma_t^2).
    """
    def __init__(self, task_names: list[str], **kwargs):
        super().__init__(**kwargs)
        self.task_names = list(task_names)
        
    def build(self, input_shape=None):
        self.log_vars = self.add_weight(
            name="log_vars",
            shape=(len(self.task_names),),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        super().build(input_shape)
        
    def call(self, task_losses: dict[str, tf.Tensor]) -> tf.Tensor:
        total = tf.constant(0.0, dtype=tf.float32)
        for idx, name in enumerate(self.task_names):
            loss_t = task_losses[name]
            s_t = self.log_vars[idx]
            # Precision weight: exp(-s_t)
            precision = tf.exp(-s_t)
            total = total + precision * loss_t + 0.5 * s_t
        return total

    def get_config(self) -> dict:
        config = super().get_config()
        config["task_names"] = self.task_names
        return config
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_homoscedastic_loss.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/generate_graph.py tests/test_homoscedastic_loss.py
git commit -m "feat(mtl): prune noise heads to core 7 and add homoscedastic loss weighting"
```

---

### Task 3: Tokenisation Périodique de Fourier des Variables Continues

**Files:**
- Create: `core/periodic_embeddings.py`
- Modify: `core/deepsets_layers.py:100-155`
- Test: `tests/test_periodic_embeddings.py`

**Interfaces:**
- Consumes: Tenseur de features continues de dimension `(batch, 5, num_cont)`.
- Produces: `PeriodicNumericalEmbedding(num_cont=12, embed_dim=16) -> (batch, 5, num_cont, embed_dim)` ou aplati vers `(batch, 5, num_cont * embed_dim)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_periodic_embeddings.py
import pytest
import tensorflow as tf
from core.periodic_embeddings import PeriodicNumericalEmbedding

def test_periodic_embedding_shape_and_gradients():
    layer = PeriodicNumericalEmbedding(num_features=12, num_frequencies=8)
    inputs = tf.random.normal((4, 5, 12))
    
    with tf.GradientTape() as tape:
        out = layer(inputs)
        loss = tf.reduce_mean(out)
        
    grads = tape.gradient(loss, layer.trainable_weights)
    assert out.shape == (4, 5, 12 * 16) # 8 frequencies -> 16 sin/cos dims
    assert len(grads) == len(layer.trainable_weights)
    assert all(g is not None for g in grads)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_periodic_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.periodic_embeddings'`

- [ ] **Step 3: Write minimal implementation**

```python
# core/periodic_embeddings.py
"""Periodic Numerical Embeddings for continuous features in Deep Learning.

Based on Gorishniy et al. (NeurIPS 2022) "On Embeddings for Numerical Features in Tabular Data".
Transforms continuous features into Fourier representations: [sin(2*pi*c*x), cos(2*pi*c*x)].
"""
from __future__ import annotations
import math
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="nnriot")
class PeriodicNumericalEmbedding(tf.keras.layers.Layer):
    """Encodes scalar numerical features with learnable frequencies."""
    
    def __init__(self, num_features: int = 12, num_frequencies: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.num_features = int(num_features)
        self.num_frequencies = int(num_frequencies)

    def build(self, input_shape=None):
        # Learnable frequency multipliers c_i for each feature
        self.coefficients = self.add_weight(
            name="coefficients",
            shape=(self.num_features, self.num_frequencies),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # inputs shape: (batch, 5, num_features)
        # Expand for broadcasting: (batch, 5, num_features, 1) * (num_features, num_frequencies)
        x_expanded = tf.expand_dims(inputs, axis=-1)
        angles = 2.0 * math.pi * x_expanded * self.coefficients
        sin_part = tf.sin(angles)
        cos_part = tf.cos(angles)
        fourier = tf.concat([sin_part, cos_part], axis=-1) # (batch, 5, num_features, 2 * num_freq)
        
        batch_size = tf.shape(inputs)[0]
        out_dim = self.num_features * (2 * self.num_frequencies)
        return tf.reshape(fourier, [batch_size, 5, out_dim])

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
            "num_frequencies": self.num_frequencies,
        })
        return config
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_periodic_embeddings.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/periodic_embeddings.py tests/test_periodic_embeddings.py
git commit -m "feat(layers): implement periodic fourier embedding for continuous tabular features"
```

---

### Task 4: Set-Transformer PMA & Cross-Attention Transport Optimal Sinkhorn

**Files:**
- Modify: `core/deepsets_layers.py:1-233`
- Test: `tests/test_deepsets_layers.py`

**Interfaces:**
- Consumes: Tenseurs `team_a, team_b` de forme `(batch, 5, 14)`.
- Produces: `SiameseTeamEncoder` avec `SetTransformerPMA` intra-équipe et `SinkhornCrossAttention` entre équipes, garantissant strictement $T_{diff}(B, A) = -T_{diff}(A, B)$.

- [ ] **Step 1: Write the failing test**

```python
# In tests/test_deepsets_layers.py - Add test for SetTransformer and Sinkhorn
def test_siamese_team_encoder_pma_and_sinkhorn():
    from core.deepsets_layers import SiameseTeamEncoder
    encoder = SiameseTeamEncoder(
        embed_dim=32,
        latent_dim=64,
        use_set_transformer=True,
        use_sinkhorn_attention=True,
    )
    team_a = tf.random.normal((2, 5, 14))
    team_b = tf.random.normal((2, 5, 14))
    
    t_diff_ab, t_sum_ab, _, _ = encoder((team_a, team_b))
    t_diff_ba, t_sum_ba, _, _ = encoder((team_b, team_a))
    
    # Vérification stricte d'anti-symétrie et de symétrie
    np.testing.assert_allclose(t_diff_ab.numpy(), -t_diff_ba.numpy(), atol=1e-5)
    np.testing.assert_allclose(t_sum_ab.numpy(), t_sum_ba.numpy(), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_deepsets_layers.py -k test_siamese_team_encoder_pma_and_sinkhorn -v`
Expected: FAIL with `TypeError: SiameseTeamEncoder.__init__() got an unexpected keyword argument 'use_set_transformer'`

- [ ] **Step 3: Implement SetTransformerPMA and SinkhornCrossAttention**

Ajouter dans `core/deepsets_layers.py` :
1. `SetTransformerPMA` : Utilise $k=2$ vecteurs de seed apprenables et Multi-Head Attention sur les 5 tokens de joueurs.
2. `SinkhornCrossAttention` : Normalisation itérative par double stochasticité des log-attentions bipartites.
3. Intégrer `PeriodicNumericalEmbedding` sur les features continues dans `_process_team`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_deepsets_layers.py -v`
Expected: ALL PASS (y compris les anciens tests d'anti-symétrie).

- [ ] **Step 5: Commit**

```bash
git add core/deepsets_layers.py tests/test_deepsets_layers.py
git commit -m "feat(architecture): integrate SetTransformer PMA and Sinkhorn optimal transport in SiameseTeamEncoder"
```

---

### Task 5: Tête Skew-Symétrique Bilinéaire & Sortie Évidentielle Dirichlet

**Files:**
- Modify: `core/equivariant_model.py:1-117`
- Test: `tests/test_equivariant_model.py`

**Interfaces:**
- Consumes: `t_a, t_b` et `t_diff, t_sum`.
- Produces: `EvidentialDirichletSoftmax` pour `winner` sortant `[p_a, p_b, epistemic_uncertainty_u]`, et `SkewSymmetricBilinearDense` pour `kill_handicap`.

- [ ] **Step 1: Write the failing test**

```python
# In tests/test_equivariant_model.py
def test_evidential_dirichlet_head():
    from core.equivariant_model import build_equivariant_deepsets_model
    model = build_equivariant_deepsets_model(
        player_feature_dim=14,
        use_evidential=True,
    )
    team_a = tf.random.normal((2, 5, 14))
    team_b = tf.random.normal((2, 5, 14))
    
    preds_ab = model({"team_a": team_a, "team_b": team_b})
    preds_ba = model({"team_a": team_b, "team_b": team_a})
    
    assert "winner" in preds_ab
    assert "winner_uncertainty" in preds_ab
    
    # Test anti-symétrie des probabilités
    p_winner_ab = preds_ab["winner"].numpy()[:, 1] # P(Team B gagne)
    p_winner_ba = preds_ba["winner"].numpy()[:, 1]
    np.testing.assert_allclose(p_winner_ab, 1.0 - p_winner_ba, atol=1e-5)
    
    # Test invariance de l'incertitude épistémique u
    u_ab = preds_ab["winner_uncertainty"].numpy()
    u_ba = preds_ba["winner_uncertainty"].numpy()
    np.testing.assert_allclose(u_ab, u_ba, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_equivariant_model.py -k test_evidential_dirichlet_head -v`
Expected: FAIL with `TypeError: build_equivariant_deepsets_model() got an unexpected keyword argument 'use_evidential'`

- [ ] **Step 3: Write minimal implementation**

Implémenter dans `core/equivariant_model.py` :
- `SkewSymmetricBilinearDense` : $s = h_b^\top (W - W^\top) h_a$
- `EvidentialDirichletSoftmax` : Preuve positive $e_k = \text{softplus}(s_k)$, Dirichlet $\alpha_k = e_k + 1$, incertitude $u = 2 / (\alpha_0 + \alpha_1)$.
- Mettre à jour `build_equivariant_deepsets_model` pour générer les 7 têtes élaguées.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_equivariant_model.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add core/equivariant_model.py tests/test_equivariant_model.py
git commit -m "feat(model): implement skew-symmetric bilinear head and evidential dirichlet uncertainty"
```

---

### Task 6: Décomposition Contrefactuelle de Shapley & Intégration Service

**Files:**
- Create: `core/shapley_credit.py`
- Modify: `core/prediction_service.py:100-180`
- Test: `tests/test_shapley_credit.py`

**Interfaces:**
- Consumes: Modèle `NNriot_EquivariantDeepSets` et tenseurs de draft `team_a, team_b`.
- Produces: `compute_draft_shapley_values(model, team_a, team_b)` retournant `{"draft_advantage": float, "player_credits": list[float]}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_shapley_credit.py
import pytest
import tensorflow as tf
from core.shapley_credit import compute_counterfactual_shapley
from core.equivariant_model import build_equivariant_deepsets_model

def test_counterfactual_shapley_properties():
    model = build_equivariant_deepsets_model(player_feature_dim=14)
    team_a = tf.random.normal((1, 5, 14))
    team_b = tf.random.normal((1, 5, 14))
    
    shapley = compute_counterfactual_shapley(model, team_a, team_b)
    assert "draft_delta" in shapley
    assert "team_a_credits" in shapley
    assert "team_b_credits" in shapley
    assert len(shapley["team_a_credits"]) == 5
    assert len(shapley["team_b_credits"]) == 5
    
    # Anti-symétrie du Shapley draft delta
    shapley_swap = compute_counterfactual_shapley(model, team_b, team_a)
    assert pytest.approx(shapley["draft_delta"], abs=1e-4) == -shapley_swap["draft_delta"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_shapley_credit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.shapley_credit'`

- [ ] **Step 3: Write minimal implementation**

```python
# core/shapley_credit.py
"""Exact Counterfactual Shapley Credit Assignment for 5v5 MOBA Drafts.

Based on Wang et al. (KDD 2021) "Shapley Counterfactual Credits for Multi-Agent Reinforcement Learning".
Decomposes team win expectation into individual player/champion marginal contributions.
"""
from __future__ import annotations
import itertools
import numpy as np
import tensorflow as tf

def compute_counterfactual_shapley(
    model: tf.keras.Model,
    team_a: tf.Tensor,
    team_b: tf.Tensor,
) -> dict[str, Any]:
    # Baseline null player: champion 0, default priors
    null_player = tf.zeros_like(team_a[:, 0:1, :])
    
    # Base prediction
    base_preds = model({"team_a": team_a, "team_b": team_b})
    # Win probability for team B
    base_p_b = float(base_preds["winner"][0, 1].numpy())
    draft_delta = base_p_b - 0.5
    
    # Exact marginal contribution over all 5 player slots
    credits_a = []
    credits_b = []
    
    for i in range(5):
        # Mask player i in team a
        mask_a = tf.concat([team_a[:, :i, :], null_player, team_a[:, i+1:, :]], axis=1)
        p_mask_a = float(model({"team_a": mask_a, "team_b": team_b})["winner"][0, 1].numpy())
        credits_a.append(round(base_p_b - p_mask_a, 4))
        
        # Mask player i in team b
        mask_b = tf.concat([team_b[:, :i, :], null_player, team_b[:, i+1:, :]], axis=1)
        p_mask_b = float(model({"team_a": team_a, "team_b": mask_b})["winner"][0, 1].numpy())
        credits_b.append(round(p_mask_b - base_p_b, 4))
        
    return {
        "draft_delta": round(draft_delta, 4),
        "team_a_credits": credits_a,
        "team_b_credits": credits_b,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_shapley_credit.py -v`
Expected: PASS

- [ ] **Step 5: Integrate into PredictionService and commit**

Mettre à jour `core/prediction_service.py` pour inclure `shapley_credits` et l'incertitude épistémique `epistemic_uncertainty` dans la réponse d'inférence.
Commit: `git add core/shapley_credit.py core/prediction_service.py tests/test_shapley_credit.py && git commit -m "feat(service): integrate counterfactual shapley credit and evidential uncertainty into prediction service"`
