import time
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import tensorflow as tf
from core.equivariant_model import build_equivariant_deepsets_model
from core.generate_graph import LOSS_PER_HEAD, LOSS_WEIGHTS
from core.pcgrad import pcgrad_step, project_conflicting_gradients

model = build_equivariant_deepsets_model(player_feature_dim=18, macro_feature_dim=15, include_macro_clash=True)
optimizer = tf.keras.optimizers.Adam(0.001)
loss_fns = {k: tf.keras.losses.get(LOSS_PER_HEAD[k]) for k in LOSS_PER_HEAD if k in model.output_names}

batch_size = 128
xb = {
    'team_a': tf.random.normal((batch_size, 5, 18)),
    'team_b': tf.random.normal((batch_size, 5, 18)),
    'macro_clash': tf.random.normal((batch_size, 15)),
}
yb = {
    'winner': tf.one_hot(tf.random.uniform((batch_size,), 0, 2, dtype=tf.int32), 2),
    'team_b_kill_lead': tf.one_hot(tf.random.uniform((batch_size,), 0, 2, dtype=tf.int32), 2),
    'first_blood': tf.one_hot(tf.random.uniform((batch_size,), 0, 3, dtype=tf.int32), 3),
    'kill_handicap': tf.random.normal((batch_size, 1)),
    'total_kills': tf.random.normal((batch_size, 1)),
    'team_a_kills': tf.random.normal((batch_size, 1)),
    'team_b_kills': tf.random.normal((batch_size, 1)),
}

print("1. Running Eager PCGrad warmup...")
pcgrad_step(model, optimizer, xb, yb, loss_fns, loss_weights=LOSS_WEIGHTS)

t0 = time.time()
n_iters = 5
for _ in range(n_iters):
    pcgrad_step(model, optimizer, xb, yb, loss_fns, loss_weights=LOSS_WEIGHTS)
t_eager = (time.time() - t0) / n_iters
print(f"Eager step time: {t_eager * 1000:.1f} ms ({1.0 / t_eager:.1f} steps/s)")

# Now test compiled step
variables = model.trainable_variables
var_shapes = [v.shape for v in variables]
var_sizes = [v.shape.num_elements() for v in variables]
active_heads = [k for k in loss_fns if k in yb and LOSS_WEIGHTS.get(k, 1.0) > 0.0]

def project_conflicting_gradients_graph(task_gradients, variables):
    flat_grads = []
    for g_list in task_gradients:
        flat_parts = []
        for g, v in zip(g_list, variables):
            if g is None:
                flat_parts.append(tf.zeros([v.shape.num_elements()], dtype=tf.float32))
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
    splits = tf.split(final_flat, [v.shape.num_elements() for v in variables])
    return [tf.reshape(s, v.shape) for s, v in zip(splits, variables)]

@tf.function
def compiled_pcgrad_step(xb_step, yb_step):
    with tf.GradientTape(persistent=True) as tape:
        preds = model(xb_step, training=True)
        head_losses = {}
        for head in active_heads:
            loss_fn = loss_fns[head]
            weight = tf.constant(LOSS_WEIGHTS.get(head, 1.0), dtype=tf.float32)
            raw = loss_fn(yb_step[head], preds[head])
            head_losses[head] = tf.reduce_mean(raw) * weight

    task_gradients = []
    for head in active_heads:
        grads = tape.gradient(head_losses[head], variables)
        task_gradients.append(grads)
    del tape

    deconflicted = project_conflicting_gradients_graph(task_gradients, variables)
    optimizer.apply_gradients(zip(deconflicted, variables))
    return {k: head_losses[k] for k in active_heads}

print("2. Running Compiled @tf.function PCGrad warmup...")
compiled_pcgrad_step(xb, yb)

t0 = time.time()
n_iters = 10
for _ in range(n_iters):
    res = compiled_pcgrad_step(xb, yb)
t_compiled = (time.time() - t0) / n_iters
print(f"Compiled @tf.function step time: {t_compiled * 1000:.1f} ms ({1.0 / t_compiled:.1f} steps/s)")
print(f"Speedup: {t_eager / t_compiled:.2f}x faster!")
