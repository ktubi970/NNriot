"""Inspect fixed-side composition invariants using synthetic inference inputs."""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO_ROOT / "runs/candidate_deepsets_v4_compiled_gpu/model.keras")
    args = parser.parse_args(argv)

    import tensorflow as tf
    from core.equivariant_model import NNRIOT_CUSTOM_OBJECTS

    model = tf.keras.models.load_model(args.checkpoint, custom_objects=NNRIOT_CUSTOM_OBJECTS)
    input_shapes = {tensor.name.split(":")[0]: tuple(tensor.shape) for tensor in model.inputs}
    dim = int(input_shapes["team_a"][-1])
    rng = np.random.default_rng(42)
    xa = rng.uniform(0.1, 0.9, (2, 5, dim)).astype(np.float32)
    xb = rng.uniform(0.1, 0.9, (2, 5, dim)).astype(np.float32)
    xa[:, :, 0], xb[:, :, 0] = np.arange(1, 6), np.arange(6, 11)
    xa[:, :, 1] = xb[:, :, 1] = np.arange(5)
    original = {"team_a": xa, "team_b": xb}
    swapped = {"team_a": xb, "team_b": xa}
    mirror = {"team_a": xa, "team_b": xa}
    if "macro_clash" in input_shapes:
        macro = rng.normal(size=(2, int(input_shapes["macro_clash"][-1]))).astype(np.float32)
        original["macro_clash"] = macro
        swapped["macro_clash"] = -macro
        mirror["macro_clash"] = np.zeros_like(macro)

    pred_orig = model(original, training=False)
    pred_swap = model(swapped, training=False)
    pred_mirror = model(mirror, training=False)
    p_draft_orig = np.asarray(pred_orig["draft_advantage"])
    p_draft_swap = np.asarray(pred_swap["draft_advantage"])
    p_draft_mirror = np.asarray(pred_mirror["draft_advantage"])
    p_winner_mirror = np.asarray(pred_mirror["winner"])
    tolerance = 1e-5
    mirror_error = float(np.max(np.abs(p_draft_mirror - 0.5)))
    swap_error = float(np.max(np.abs(p_draft_orig - p_draft_swap[:, ::-1])))
    print(f"1. draft_advantage mirror: {p_draft_mirror[0]} (max error={mirror_error:.7g}, tolerance={tolerance:g})")
    print(f"2. draft_advantage composition swap at fixed sides: max error={swap_error:.7g}, tolerance={tolerance:g}")
    try:
        bias = float(model.get_layer("winner_side_bias").bias.numpy()[0])
    except ValueError:
        bias = 0.0
    # Read the actual winner head: an evidential head does not use sigmoid(-2b).
    print(f"3. Learned side bias b: {bias:.5f} -> Mirror Blue Winrate: {p_winner_mirror[0, 0] * 100:.2f}%")
    print(f"4. Winner mirror at fixed Blue/Red sides: {p_winner_mirror[0]}")
    return 0 if mirror_error <= tolerance and swap_error <= tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())
