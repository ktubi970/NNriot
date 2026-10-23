import time
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import tensorflow as tf
from core.equivariant_model import build_equivariant_deepsets_model
from core.generate_graph import LOSS_PER_HEAD, LOSS_WEIGHTS

model = build_equivariant_deepsets_model(player_feature_dim=18, macro_feature_dim=15, include_macro_clash=True)
optimizer = tf.keras.optimizers.Adam(0.001)
loss_fns = {k: tf.keras.losses.get(LOSS_PER_HEAD[k]) for k in LOSS_PER_HEAD if k in model.output_names}

batch_size = 128
n_samples = 128 * 10  # 10 batches

x_data = {
    'team_a': tf.random.normal((n_samples, 5, 18)),
    'team_b': tf.random.normal((n_samples, 5, 18)),
    'macro_clash': tf.random.normal((n_samples, 15)),
}
y_data = {
    'winner': tf.one_hot(tf.random.uniform((n_samples,), 0, 2, dtype=tf.int32), 2),
    'team_b_kill_lead': tf.one_hot(tf.random.uniform((n_samples,), 0, 2, dtype=tf.int32), 2),
    'first_blood': tf.one_hot(tf.random.uniform((n_samples,), 0, 3, dtype=tf.int32), 3),
    'kill_handicap': tf.random.normal((n_samples, 1)),
    'total_kills': tf.random.normal((n_samples, 1)),
    'team_a_kills': tf.random.normal((n_samples, 1)),
    'team_b_kills': tf.random.normal((n_samples, 1)),
}

dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

variables = model.trainable_variables
active_heads = [k for k in loss_fns if k in y_data and LOSS_WEIGHTS.get(k, 1.0) > 0.0]

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
def step_fn(xb_step, yb_step):
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
    total_step_loss = tf.add_n([head_losses[k] for k in active_heads])
    return total_step_loss

@tf.function
def train_epoch_dataset(ds):
    total_loss = tf.constant(0.0, dtype=tf.float32)
    count = tf.constant(0.0, dtype=tf.float32)
    for xb_b, yb_b in ds:
        l = step_fn(xb_b, yb_b)
        total_loss += l
        count += 1.0
    return total_loss / count

print("1. Benchmarking step-by-step in Python loop (10 batches)...")
# Warmup
for xb_b, yb_b in dataset.take(1):
    step_fn(xb_b, yb_b)

t0 = time.time()
for xb_b, yb_b in dataset:
    step_fn(xb_b, yb_b)
t_py_loop = time.time() - t0
print(f"Python loop 10 batches: {t_py_loop:.3f} s ({t_py_loop/10*1000:.1f} ms/step)")

print("2. Benchmarking dataset loop inside @tf.function (10 batches)...")
# Warmup
train_epoch_dataset(dataset)

t0 = time.time()
avg_loss = train_epoch_dataset(dataset)
t_graph_loop = time.time() - t0
print(f"Graph loop 10 batches: {t_graph_loop:.3f} s ({t_graph_loop/10*1000:.1f} ms/step)")
print(f"Speedup inside graph epoch loop: {t_py_loop / t_graph_loop:.2f}x faster!")
