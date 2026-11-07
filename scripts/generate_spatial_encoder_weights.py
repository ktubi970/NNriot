"""Write the pinned spatial trajectory encoder weights (one-off, never overwrites).

The committed asset is the source of truth for stored embeddings; this script only
documents how it was drawn. A new asset means a new ``encoder_id``, so every stored
embedding must then be recomputed with ``compute_spatial_embeddings.py --recompute-all``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.spatial_rnn import (
    SPATIAL_ENCODER_SEED,
    SPATIAL_ENCODER_WEIGHTS_PATH,
    generate_spatial_encoder_weights,
    spatial_encoder_id,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SPATIAL_ENCODER_SEED)
    parser.add_argument("--output", type=Path, default=SPATIAL_ENCODER_WEIGHTS_PATH)
    args = parser.parse_args(argv)
    payload = generate_spatial_encoder_weights(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")
    print(f"Wrote {args.output} ({spatial_encoder_id(payload)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
