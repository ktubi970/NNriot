"""Projecting Conflicting Gradients (PCGrad) optimizer wrapper.

Based on Yu et al. (NeurIPS 2020) "Gradient Surgery for Multi-Task Learning"
and Shi & Li (2023) "Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning".

When two tasks have conflicting gradient directions (<g_i, g_j> < 0), PCGrad
projects g_i onto the normal plane of g_j, eliminating negative transfer
without dropping secondary heads.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Sequence
import tensorflow as tf


def project_conflicting_gradients(
    task_gradients: list[list[tf.Tensor | None]],
    variables: list[tf.Variable] | None = None,
) -> list[tf.Tensor]:
    """Project conflicting gradients across tasks.
    
    Args:
        task_gradients: list of T lists, where each sublist contains gradients
                        corresponding to the model's trainable variables.
        variables: Optional list of trainable variables to determine shapes
                   for variables where gradient is None.
                   
    Returns:
        Deconflicted combined gradient list matching the variables structure.
    """
    if not task_gradients:
        return []
        
    num_vars = len(task_gradients[0])

    # Build reference shapes and sizes
    shapes = []
    sizes = []
    for var_idx in range(num_vars):
        if variables is not None:
            v = variables[var_idx]
            shapes.append(tf.shape(v))
            sizes.append(tf.size(v))
        else:
            # Find first non-None gradient for this variable
            sample_g = next((t_grads[var_idx] for t_grads in task_gradients if t_grads[var_idx] is not None), None)
            if sample_g is not None:
                shapes.append(tf.shape(sample_g))
                sizes.append(tf.size(sample_g))
            else:
                shapes.append(tf.constant([1]))
                sizes.append(tf.constant(1))

    # Flatten each task's gradients, replacing None with zeros
    flat_grads = []
    for g_list in task_gradients:
        flat_parts = []
        for var_idx, g in enumerate(g_list):
            if g is None:
                flat_parts.append(tf.zeros([sizes[var_idx]], dtype=tf.float32))
            else:
                flat_parts.append(tf.reshape(tf.cast(g, tf.float32), [-1]))
        flat_grads.append(tf.concat(flat_parts, axis=0))

    num_tasks = len(flat_grads)
    deconflicted = [tf.identity(g) for g in flat_grads]

    indices = list(range(num_tasks))
    for i in indices:
        shuffled_j = [j for j in indices if j != i]
        random.shuffle(shuffled_j)
        for j in shuffled_j:
            g_i = deconflicted[i]
            g_j = flat_grads[j]

            dot = tf.reduce_sum(g_i * g_j)
            if dot < 0.0:
                norm_sq = tf.reduce_sum(g_j * g_j) + 1e-8
                proj = (dot / norm_sq) * g_j
                deconflicted[i] = g_i - proj

    # Sum all deconflicted gradients
    final_flat = tf.add_n(deconflicted)

    # Reconstruct original variable shapes
    final_unflattened = []
    offset = 0
    for var_idx in range(num_vars):
        numel = sizes[var_idx]
        chunk = final_flat[offset:offset + numel]
        final_unflattened.append(tf.reshape(chunk, shapes[var_idx]))
        offset += numel

    return final_unflattened


def pcgrad_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    x_batch: Any,
    target_dict: dict[str, Any],
    loss_fns: dict[str, Callable[[Any, Any], tf.Tensor]],
    loss_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Execute a single multi-task training step using PCGrad gradient deconfliction.
    
    Args:
        model: Multi-output Keras model.
        optimizer: Keras optimizer.
        x_batch: Input tensors (e.g. dict with team_a and team_b).
        target_dict: Dict of ground-truth target tensors per head.
        loss_fns: Dict mapping head name to loss function.
        loss_weights: Optional weighting per head (default 1.0).
        
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

    for head in active_heads:
        loss_fn = loss_fns[head]
        weight = float(loss_weights.get(head, 1.0))
        y_true = target_dict[head]

        with tf.GradientTape() as tape:
            preds = model(x_batch, training=True)
            y_pred = preds[head]
            raw_loss = loss_fn(y_true, y_pred)
            head_loss = tf.reduce_mean(raw_loss) * weight

        grads = tape.gradient(head_loss, variables)
        task_gradients.append(grads)
        losses_record[head] = float(head_loss.numpy())

    # Project conflicting gradients passing trainable variables
    deconflicted_grads = project_conflicting_gradients(task_gradients, variables=variables)

    # Apply deconflicted gradients
    optimizer.apply_gradients(zip(deconflicted_grads, variables))

    return losses_record
