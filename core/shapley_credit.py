"""Leave-one-player-out counterfactual credits for NNriot's 5v5 model.

Inspired by Li, Kuang, Wang et al. (KDD 2021), "Shapley Counterfactual
Credits for Multi-Agent Reinforcement Learning", https://arxiv.org/abs/2106.00285.
The public name follows the architecture plan. These single-player ablations
are NOT exact Shapley values: they do not average over coalitions, and their
sum need not equal the match margin. They measure joint champion/player effects,
not isolated player mastery or a causal effect.
"""
from __future__ import annotations

from typing import TypedDict

import numpy as np
import tensorflow as tf

from core.features.draft_composition import extract_macro_clash_vector


class CounterfactualCredits(TypedDict):
    draft_delta: float
    team_a_credits: list[float]
    team_b_credits: list[float]


def compute_counterfactual_shapley(
    model: tf.keras.Model,
    team_a: tf.Tensor,
    team_b: tf.Tensor,
) -> CounterfactualCredits:
    """Explain one match with ten neutral-player counterfactuals.

    Inputs have matching shape ``(1, 5, features)`` (also accepts NumPy arrays).
    Each baseline clears champion ID and continuous features but preserves role
    ID (column 1), avoiding duplicate roles during the encoder's canonical sort.
    Credits follow input row order; pipeline inputs use TOP/JUNGLE/MID/BOT/SUPPORT.

    With ``p = P(Team B wins)``, the plan's sign convention is:
    ``draft_delta = p - .5``, ``credit_a[i] = p - p_without_a_i`` and
    ``credit_b[i] = p_without_b_i - p``. Thus NEGATIVE credits favor the player's
    own team. For an anti-symmetric model, swapping teams negates draft_delta
    and exchanges the credit lists without negation.

    draft_delta is the full *raw* model margin, including player priors, not an
    isolated intrinsic composition advantage. No calibration or rounding is
    applied. Zero is a reference baseline, not a learned average player.
    Macro-clash inputs, when required, are recomputed for each masked draft.
    This eager inference helper returns Python scalars for a JSON response;
    multi-match batches are rejected rather than silently taking the first row.
    """
    a = np.array(team_a, dtype=np.float32, copy=True)
    b = np.array(team_b, dtype=np.float32, copy=True)
    if (a.ndim != 3 or a.shape[:2] != (1, 5) or a.shape[-1] < 2
            or b.shape != a.shape):
        raise ValueError("team_a and team_b must have matching shape (1, 5, features >= 2)")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("team_a and team_b must contain only finite features")

    # Batch order: base, A without players 0..4, B without players 0..4.
    batch_a = np.repeat(a, 11, axis=0)
    batch_b = np.repeat(b, 11, axis=0)
    for slot in range(5):
        batch_a[1 + slot, slot, :] = 0.0
        batch_a[1 + slot, slot, 1] = a[0, slot, 1]
        batch_b[6 + slot, slot, :] = 0.0
        batch_b[6 + slot, slot, 1] = b[0, slot, 1]

    inputs = {"team_a": tf.convert_to_tensor(batch_a), "team_b": tf.convert_to_tensor(batch_b)}
    input_names = set(getattr(model, "input_names", None) or [])
    input_names.update(t.name.split(":")[0] for t in getattr(model, "inputs", []) or [])
    if "macro_clash" in input_names:
        inputs["macro_clash"] = tf.convert_to_tensor(np.stack([
            extract_macro_clash_vector(a_row[:, 0].astype(int), b_row[:, 0].astype(int))
            for a_row, b_row in zip(batch_a, batch_b)
        ]))

    winner = np.asarray(model(inputs, training=False)["winner"], dtype=np.float64)
    if winner.shape != (11, 2) or not np.isfinite(winner).all():
        raise ValueError("winner must contain finite probabilities with shape (11, 2)")
    p_b = winner[:, 1]
    return {
        "draft_delta": float(p_b[0] - 0.5),
        "team_a_credits": (p_b[0] - p_b[1:6]).tolist(),
        "team_b_credits": (p_b[6:11] - p_b[0]).tolist(),
    }
