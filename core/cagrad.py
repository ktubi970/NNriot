"""Conflict-Averse Gradient Descent (CAGrad) optimizer for Multi-Task Learning.

Based on Liu et al. (NeurIPS 2021) "Conflict-Averse Gradient Descent for Multi-task Learning".
Replaces naive PCGrad by finding a direction that optimizes average loss while bounding 
the worst-case individual task degradation via dual quadratic programming on a simplex.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize


def project_conflict_averse_gradients(
    task_gradients: list[list[tf.Tensor | None]],
    variables: Sequence[tf.Variable] | None = None,
    c: float = 0.5,
) -> list[tf.Tensor]:
    """Applies CAGrad projection to multi-task gradients.

    Args:
        task_gradients: List of T lists, where each sublist contains gradients
            corresponding to the model's trainable variables for that task.
        variables: Optional sequence of trainable variables to determine shapes
            for variables where gradient is None.
        c: Hyperparameter in [0, 1] controlling conflict aversion. When c=0,
            reverts to standard mean gradient descent. Higher c enforces stronger
            worst-case task protection.

    Returns:
        Deconflicted combined gradient list matching the variables structure.
    """
    if not task_gradients:
        return []

    num_vars = len(task_gradients[0])
    if num_vars == 0:
        return []

    # Build reference shapes and sizes
    shapes: list[Sequence[int]] = []
    sizes: list[int] = []
    for var_idx in range(num_vars):
        if variables is not None:
            v = variables[var_idx]
            shapes.append(tuple(v.shape.as_list()))
            sizes.append(int(tf.size(v).numpy()))
        else:
            sample_g = next((t_grads[var_idx] for t_grads in task_gradients if t_grads[var_idx] is not None), None)
            if sample_g is not None:
                shapes.append(tuple(sample_g.shape.as_list()))
                sizes.append(int(tf.size(sample_g).numpy()))
            else:
                shapes.append((1,))
                sizes.append(1)

    # Flatten each task's gradients on device, executing only T transfers total
    flat_grads: list[np.ndarray] = []
    for g_list in task_gradients:
        parts: list[tf.Tensor] = []
        for var_idx, g in enumerate(g_list):
            if g is None:
                parts.append(tf.zeros([sizes[var_idx]], dtype=tf.float32))
            else:
                parts.append(tf.reshape(tf.cast(g, tf.float32), [-1]))
        if parts:
            task_flat = tf.concat(parts, axis=0).numpy()
        else:
            task_flat = np.zeros(0, dtype=np.float32)
        flat_grads.append(task_flat)

    num_tasks = len(flat_grads)
    total_params = flat_grads[0].shape[0]
    if total_params == 0:
        return []

    if num_tasks == 1:
        final_flat = flat_grads[0]
    else:
        G = np.stack(flat_grads, axis=0)  # (num_tasks, total_params)
        g0 = np.mean(G, axis=0)  # Average gradient
        norm_g0 = float(np.linalg.norm(g0))

        if norm_g0 < 1e-12 or c <= 0.0:
            final_flat = g0
        else:
            # Scale-invariant normalization: G_tilde = G / norm_g0
            # Ensures O(1) conditioning and tolerance-invariance regardless of gradient scale
            G_tilde = G / norm_g0
            g0_tilde = g0 / norm_g0
            GG_tilde = G_tilde @ G_tilde.T  # (num_tasks, num_tasks), O(1)
            v_tilde = G_tilde @ g0_tilde    # (num_tasks,), O(1)

            def obj(w: np.ndarray) -> float:
                gw_norm = np.sqrt(max(1e-12, float(w @ GG_tilde @ w)))
                return float(w @ v_tilde + c * gw_norm)

            def grad(w: np.ndarray) -> np.ndarray:
                gw_norm = np.sqrt(max(1e-12, float(w @ GG_tilde @ w)))
                return v_tilde + (c / gw_norm) * (GG_tilde @ w)

            w0 = np.ones(num_tasks, dtype=np.float64) / float(num_tasks)
            bounds = [(0.0, 1.0) for _ in range(num_tasks)]
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0, 'jac': lambda w: np.ones_like(w)})

            res = minimize(
                obj,
                w0,
                jac=grad,
                bounds=bounds,
                constraints=cons,
                method='SLSQP',
                options={'maxiter': 100, 'ftol': 1e-9},
            )

            w_opt = res.x if res.success else w0
            # Ensure simplex validity
            w_opt = np.maximum(w_opt, 0.0)
            sum_w = np.sum(w_opt)
            if sum_w > 0:
                w_opt = w_opt / sum_w
            else:
                w_opt = w0

            g_opt_tilde = G_tilde.T @ w_opt
            norm_g_opt = float(np.linalg.norm(g_opt_tilde))

            if norm_g_opt > 1e-8:
                g_cagrad = norm_g0 * (g0_tilde + (c / norm_g_opt) * g_opt_tilde)
            else:
                g_cagrad = g0

            final_flat = g_cagrad.astype(np.float32)

    # Reconstruct original variable shapes as TensorFlow tensors
    final_unflattened: list[tf.Tensor] = []
    offset = 0
    for var_idx in range(num_vars):
        numel = sizes[var_idx]
        chunk = final_flat[offset : offset + numel]
        final_unflattened.append(tf.constant(chunk.reshape(shapes[var_idx]), dtype=tf.float32))
        offset += numel

    return final_unflattened


def cagrad_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    x_batch: Any,
    target_dict: dict[str, Any],
    loss_fns: dict[str, Callable[[Any, Any], tf.Tensor]],
    loss_weights: dict[str, float] | None = None,
    c: float = 0.5,
) -> dict[str, float]:
    """Execute a single multi-task training step using CAGrad conflict-averse optimization.

    Args:
        model: Multi-output Keras model.
        optimizer: Keras optimizer.
        x_batch: Input tensors (e.g. dict with team_a and team_b).
        target_dict: Dict of ground-truth target tensors per head.
        loss_fns: Dict mapping head name to loss function.
        loss_weights: Optional weighting per head (default 1.0).
        c: CAGrad conflict-aversion hyperparameter (default 0.5).

    Returns:
        Dict mapping head name to evaluated loss float.
    """
    if loss_weights is None:
        loss_weights = {k: 1.0 for k in loss_fns}

    active_heads = [k for k in loss_fns if k in target_dict and loss_weights.get(k, 1.0) > 0.0]
    if not active_heads:
        return {}

    task_gradients: list[list[tf.Tensor | None]] = []
    losses_record: dict[str, float] = {}
    variables = model.trainable_variables

    # Execute forward pass for all heads on a persistent tape
    with tf.GradientTape(persistent=True) as tape:
        preds = model(x_batch, training=True)
        head_losses = {}
        for head in active_heads:
            loss_fn = loss_fns[head]
            weight = float(loss_weights.get(head, 1.0))
            raw_loss = loss_fn(target_dict[head], preds[head])
            head_losses[head] = tf.reduce_mean(raw_loss) * weight

    for head in active_heads:
        grads = tape.gradient(head_losses[head], variables)
        task_gradients.append(grads)
        losses_record[head] = float(head_losses[head].numpy())
    del tape

    # Project conflict-averse gradients
    cagrad_grads = project_conflict_averse_gradients(task_gradients, variables=variables, c=c)

    # Apply deconflicted gradients
    optimizer.apply_gradients(zip(cagrad_grads, variables))

    return losses_record
