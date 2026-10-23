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
    variables: Sequence[tf.Variable] | None = None,
) -> list[tf.Tensor]:
    """Project conflicting gradients across tasks.
    
    Branchless and graph-compilable implementation of PCGrad projection.
    
    Args:
        task_gradients: list of T lists, where each sublist contains gradients
                        corresponding to the model's trainable variables.
        variables: Optional sequence of trainable variables to determine shapes
                   for variables where gradient is None.
                   
    Returns:
        Deconflicted combined gradient list matching the variables structure.
    """
    if not task_gradients:
        return []

    num_vars = len(task_gradients[0])

    shapes = []
    sizes = []
    for var_idx in range(num_vars):
        if variables is not None:
            v = variables[var_idx]
            shapes.append(v.shape)
            if v.shape.is_fully_defined():
                sizes.append(v.shape.num_elements())
            else:
                sizes.append(tf.size(v))
        else:
            sample_g = next((t_grads[var_idx] for t_grads in task_gradients if t_grads[var_idx] is not None), None)
            if sample_g is not None:
                if sample_g.shape.is_fully_defined():
                    shapes.append(sample_g.shape)
                    sizes.append(sample_g.shape.num_elements())
                else:
                    shapes.append(tf.shape(sample_g))
                    sizes.append(tf.size(sample_g))
            else:
                shapes.append(tf.TensorShape([1]))
                sizes.append(1)

    flat_grads = []
    for g_list in task_gradients:
        flat_parts = []
        for var_idx, g in enumerate(g_list):
            if g is None:
                sz = sizes[var_idx]
                flat_parts.append(tf.zeros([sz], dtype=tf.float32))
            else:
                flat_parts.append(tf.reshape(tf.cast(g, tf.float32), [-1]))
        flat_grads.append(tf.concat(flat_parts, axis=0))

    num_tasks = len(flat_grads)
    deconflicted = [tf.identity(g) for g in flat_grads]

    for i in range(num_tasks):
        for j in range(num_tasks):
            if i != j:
                g_i = deconflicted[i]
                g_j = flat_grads[j]
                dot = tf.reduce_sum(g_i * g_j)
                norm_sq = tf.reduce_sum(g_j * g_j) + 1e-8
                scale = tf.minimum(0.0, dot) / norm_sq
                deconflicted[i] = g_i - scale * g_j

    final_flat = tf.add_n(deconflicted)
    splits = tf.split(final_flat, sizes)
    return [tf.reshape(s, shape) for s, shape in zip(splits, shapes)]


def create_compiled_pcgrad_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fns: dict[str, Callable[[Any, Any], tf.Tensor]],
    loss_weights: dict[str, float] | None = None,
    active_heads: Sequence[str] | None = None,
    jit_compile: bool = False,
) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, tf.Tensor]]:
    """Create a compiled @tf.function multi-task training step using PCGrad.
    
    Encapsulates the persistent forward/backward pass, gradient deconfliction,
    and optimizer update inside a single compiled TensorFlow execution graph.
    
    Args:
        model: Multi-output Keras model.
        optimizer: Keras optimizer.
        loss_fns: Mapping of head name to loss function.
        loss_weights: Optional mapping of head name to scalar weight (default 1.0).
        active_heads: Optional sequence of active heads to optimize. If None,
                      derived from loss_fns where weight > 0.
        jit_compile: Whether to request XLA compilation.
        
    Returns:
        Callable step_fn(x_batch, target_dict) returning a dict of scalar loss Tensors.
    """
    if loss_weights is None:
        loss_weights = {k: 1.0 for k in loss_fns}

    if active_heads is None:
        heads_tuple = tuple(k for k in loss_fns if loss_weights.get(k, 1.0) > 0.0)
    else:
        heads_tuple = tuple(active_heads)

    variables = list(model.trainable_variables)
    static_weights = {k: tf.constant(float(loss_weights.get(k, 1.0)), dtype=tf.float32) for k in heads_tuple}

    @tf.function(jit_compile=jit_compile)
    def train_step(x_batch: dict[str, tf.Tensor], target_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        with tf.GradientTape(persistent=True) as tape:
            preds = model(x_batch, training=True)
            head_losses = {}
            for head in heads_tuple:
                loss_fn = loss_fns[head]
                raw_loss = loss_fn(target_dict[head], preds[head])
                head_losses[head] = tf.reduce_mean(raw_loss) * static_weights[head]

        task_gradients = []
        for head in heads_tuple:
            grads = tape.gradient(head_losses[head], variables)
            task_gradients.append(grads)
        del tape

        deconflicted_grads = project_conflicting_gradients(task_gradients, variables=variables)
        optimizer.apply_gradients(zip(deconflicted_grads, variables))
        return head_losses

    return train_step


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

    # Execute a single forward pass for all heads on a persistent tape
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

    # Project conflicting gradients passing trainable variables
    deconflicted_grads = project_conflicting_gradients(task_gradients, variables=variables)

    # Apply deconflicted gradients
    optimizer.apply_gradients(zip(deconflicted_grads, variables))

    return losses_record


# Re-export CAGrad optimizer functions for clean backwards compatibility
from core.cagrad import cagrad_step, project_conflict_averse_gradients

__all__ = [
    "project_conflicting_gradients",
    "create_compiled_pcgrad_step",
    "pcgrad_step",
    "cagrad_step",
    "project_conflict_averse_gradients",
]
