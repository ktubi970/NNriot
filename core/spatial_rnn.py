"""Dual-Phase Trajectory GRU Encoder and spatial sequence utilities.

Processes spatio-temporal trajectories (x, y, t) from League of Legends matches:
- Phase 1 (0-15 min): Early laning, jungle pathing, and initial roams (15 timesteps).
- Phase 2 (15-25 min): Mid-game rotations, neutral objective presence, and grouping (10 timesteps, masked).
Guarantees side-invariance through coordinate normalization:
- Red side coordinates are inverted so base is mapped to (0, 0) identically to Blue side.

The encoder is a fixed, untrained projection. Its weights are committed in
``core/assets/spatial_encoder_v1.json`` and identified by ``encoder_id``; every
stored embedding records that id so embeddings from different weights are never mixed.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

MAP_SIZE: float = 15000.0

EARLY_TIMESTEPS: int = 15
MID_TIMESTEPS: int = 10
SPATIAL_FEATURE_DIM: int = 4  # [x_norm, y_norm, delta_dist, in_combat]

EARLY_EMBEDDING_DIM: int = 16
MID_EMBEDDING_DIM: int = 16
TOTAL_SPATIAL_EMBEDDING_DIM: int = EARLY_EMBEDDING_DIM + MID_EMBEDDING_DIM  # 32

SPATIAL_ENCODER_FORMAT: str = "nnriot.spatial_encoder.v1"
SPATIAL_ENCODER_NAME: str = "dual_phase_gru_v1"
SPATIAL_ENCODER_SEED: int = 20261023
SPATIAL_ENCODER_WEIGHTS_PATH: Path = Path(__file__).resolve().parent / "assets" / "spatial_encoder_v1.json"


class MixedSpatialEncoderError(ValueError):
    """Raised when embeddings from different encoder weights would share one database."""


def normalize_spatial_coordinates(
    x: float | None,
    y: float | None,
    is_red_side: bool = False,
) -> tuple[float, float]:
    """Normalize raw Riot coordinates [0, 15000] into [0.0, 1.0] with side-invariance.

    For Red side, (15000, 15000) becomes (0, 0) so both sides share the same
    relative base-to-river frame of reference.
    """
    if x is None or np.isnan(x):
        x = 0.0
    if y is None or np.isnan(y):
        y = 0.0

    x = float(np.clip(x, 0.0, MAP_SIZE))
    y = float(np.clip(y, 0.0, MAP_SIZE))

    if is_red_side:
        x_norm = (MAP_SIZE - x) / MAP_SIZE
        y_norm = (MAP_SIZE - y) / MAP_SIZE
    else:
        x_norm = x / MAP_SIZE
        y_norm = y / MAP_SIZE

    return float(x_norm), float(y_norm)


def extract_dual_phase_trajectory(
    timeline_data: dict[str, Any] | None,
    puuid: str,
    is_red_side: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract early-game (0-15m) and mid-game (15-25m) sequence arrays for a player.

    Returns:
        early_seq: array of shape (15, 4)
        mid_seq: array of shape (10, 4), zero-padded if match ended before 25 min.
    """
    early_seq = np.zeros((EARLY_TIMESTEPS, SPATIAL_FEATURE_DIM), dtype=np.float32)
    mid_seq = np.zeros((MID_TIMESTEPS, SPATIAL_FEATURE_DIM), dtype=np.float32)

    if not isinstance(timeline_data, dict):
        return early_seq, mid_seq

    info = timeline_data.get("info", timeline_data)
    participants = info.get("participants") or []

    # Map puuid to participantId
    pid = None
    for p in participants:
        if p.get("puuid") == puuid:
            pid = p.get("participantId")
            break

    if pid is None:
        return early_seq, mid_seq

    pid_str = str(pid)
    frames = info.get("frames") or []

    last_x, last_y = 0.0, 0.0

    for idx, frame in enumerate(frames):
        # We process minute 0 to 24 (total 25 minutes)
        if idx >= EARLY_TIMESTEPS + MID_TIMESTEPS:
            break

        p_frames = frame.get("participantFrames") or {}
        p_data = p_frames.get(pid_str) or {}
        pos = p_data.get("position") or {}
        raw_x = pos.get("x")
        raw_y = pos.get("y")

        if raw_x is None or raw_y is None:
            continue

        x_norm, y_norm = normalize_spatial_coordinates(raw_x, raw_y, is_red_side=is_red_side)

        # Delta movement from previous frame
        if idx == 0:
            delta_dist = 0.0
        else:
            delta_dist = float(np.sqrt((x_norm - last_x) ** 2 + (y_norm - last_y) ** 2))

        last_x, last_y = x_norm, y_norm

        # Check combat events in current frame
        events = frame.get("events") or []
        in_combat = 0.0
        for ev in events:
            ev_type = ev.get("type", "")
            if ev_type in ("CHAMPION_KILL", "ELITE_MONSTER_KILL", "BUILDING_KILL"):
                killer = ev.get("killerId")
                victim = ev.get("victimId")
                assists = ev.get("assistingParticipantIds") or []
                if pid in (killer, victim) or pid in assists:
                    in_combat = 1.0
                    break

        feat = np.array([x_norm, y_norm, delta_dist, in_combat], dtype=np.float32)

        if idx < EARLY_TIMESTEPS:
            early_seq[idx] = feat
        else:
            mid_seq[idx - EARLY_TIMESTEPS] = feat

    return early_seq, mid_seq


def build_dual_phase_trajectory_encoder() -> tf.keras.Model:
    """Build the dual-phase GRU encoder architecture with random Keras weights.

    Stored embeddings must come from ``load_pinned_spatial_encoder()`` instead.
    """
    # Early phase input (0-15m)
    early_input = tf.keras.layers.Input(
        shape=(EARLY_TIMESTEPS, SPATIAL_FEATURE_DIM),
        name="early_trajectory",
    )
    early_gru = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(8, return_sequences=False, name="early_gru"),
        name="bi_early_gru",
    )(early_input)
    early_emb = tf.keras.layers.Dense(EARLY_EMBEDDING_DIM, activation="relu", name="early_dense")(early_gru)

    # Mid phase input (15-25m) with Masking for shorter games
    mid_input = tf.keras.layers.Input(
        shape=(MID_TIMESTEPS, SPATIAL_FEATURE_DIM),
        name="mid_trajectory",
    )
    mid_masked = tf.keras.layers.Masking(mask_value=0.0, name="mid_masking")(mid_input)
    mid_gru = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(8, return_sequences=False, name="mid_gru"),
        name="bi_mid_gru",
    )(mid_masked)
    mid_emb = tf.keras.layers.Dense(MID_EMBEDDING_DIM, activation="relu", name="mid_dense")(mid_gru)

    # Concatenate both phase representations
    fusion = tf.keras.layers.Concatenate(name="spatial_trajectory_embedding")([early_emb, mid_emb])

    model = tf.keras.Model(
        inputs=[early_input, mid_input],
        outputs=fusion,
        name="DualPhaseTrajectoryGRUEncoder",
    )
    return model


def _initial_weight(path: str, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    # Same laws as the Keras defaults, drawn from NumPy so a seed does not depend on TF's RNG.
    if path.endswith("bias"):
        return np.zeros(shape, dtype=np.float32)
    if path.endswith("recurrent_kernel"):
        rows, cols = shape
        q, r = np.linalg.qr(rng.standard_normal((max(rows, cols), min(rows, cols))))
        q *= np.sign(np.diag(r))
        return (q.T if rows < cols else q).astype(np.float32)
    limit = np.sqrt(6.0 / (shape[0] + shape[1]))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


def generate_spatial_encoder_weights(seed: int = SPATIAL_ENCODER_SEED) -> dict[str, Any]:
    """Draw encoder weights from ``seed`` in the committed asset's JSON layout."""
    rng = np.random.default_rng(seed)
    model = build_dual_phase_trajectory_encoder()
    weights = []
    for weight in model.weights:
        values = _initial_weight(weight.path, tuple(weight.shape), rng)
        weights.append({"path": weight.path, "shape": list(weight.shape), "values": values.ravel().tolist()})
    return {"format": SPATIAL_ENCODER_FORMAT, "encoder": SPATIAL_ENCODER_NAME, "seed": seed, "weights": weights}


def spatial_encoder_id(payload: dict[str, Any]) -> str:
    """Identify encoder weights by their values, independently of the JSON file bytes."""
    digest = hashlib.sha256()
    for weight in payload["weights"]:
        digest.update(json.dumps([weight["path"], list(weight["shape"])]).encode("utf-8"))
        digest.update(np.asarray(weight["values"], dtype="<f4").tobytes())
    return f"{payload['encoder']}-{digest.hexdigest()[:16]}"


def load_spatial_encoder_weights(path: str | Path = SPATIAL_ENCODER_WEIGHTS_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != SPATIAL_ENCODER_FORMAT or payload.get("encoder") != SPATIAL_ENCODER_NAME:
        raise ValueError(f"unsupported spatial encoder weights file: {path}")
    return payload


def load_pinned_spatial_encoder(
    path: str | Path = SPATIAL_ENCODER_WEIGHTS_PATH,
) -> tuple[tf.keras.Model, str]:
    """Build the encoder with the committed weights and return it with its ``encoder_id``."""
    payload = load_spatial_encoder_weights(path)
    model = build_dual_phase_trajectory_encoder()
    expected = [(w.path, tuple(w.shape)) for w in model.weights]
    if [(w["path"], tuple(w["shape"])) for w in payload["weights"]] != expected:
        raise ValueError(f"spatial encoder weights do not match the encoder architecture: {path}")
    model.set_weights([np.asarray(w["values"], dtype=np.float32).reshape(w["shape"]) for w in payload["weights"]])
    return model, spatial_encoder_id(payload)


def foreign_spatial_encoder(conn, encoder_id: str) -> str | None:
    """Return another encoder found among stored embeddings ("unknown" for legacy rows), or None."""
    row = conn.execute(
        "SELECT COALESCE(encoder_id, 'unknown') FROM player_spatial_embeddings WHERE encoder_id IS NOT ? LIMIT 1",
        (encoder_id,),
    ).fetchone()
    return row[0] if row else None


_PINNED_SPATIAL_ENCODER: tuple[tf.keras.Model, str] | None = None
_ENCODER_COMPATIBLE_DATABASES: dict[str, bool] = {}


def get_pinned_spatial_encoder() -> tuple[tf.keras.Model, str]:
    global _PINNED_SPATIAL_ENCODER
    if _PINNED_SPATIAL_ENCODER is None:
        _PINNED_SPATIAL_ENCODER = load_pinned_spatial_encoder()
    return _PINNED_SPATIAL_ENCODER


def _database_accepts_encoder(db_path: str, encoder_id: str) -> bool:
    # Checked once per process: a compatible database only receives pinned rows from here on.
    key = str(Path(db_path).resolve())
    if key not in _ENCODER_COMPATIBLE_DATABASES:
        from core.database import get_connection
        with get_connection(db_path) as conn:
            foreign = foreign_spatial_encoder(conn, encoder_id)
        if foreign:
            logger.warning(
                "Skipping spatial embeddings for %s: it already holds embeddings from encoder %s. "
                "Run scripts/compute_spatial_embeddings.py --recompute-all, then restart collection.",
                db_path, foreign,
            )
        _ENCODER_COMPATIBLE_DATABASES[key] = foreign is None
    return _ENCODER_COMPATIBLE_DATABASES[key]


def extract_and_store_timeline_embeddings(
    match_id: str,
    timeline_data: dict[str, Any],
    game_creation_time: int,
    db_path: str | None = None,
) -> int:
    """Extract and persist 32D spatial embeddings for all participants in a match timeline.

    Returns 0 without writing when the database already holds embeddings from other encoder weights.
    """
    from core.database import save_player_spatial_embedding, DB_PATH
    db = db_path or DB_PATH

    info = timeline_data.get("info", timeline_data)
    participants = info.get("participants") or []
    if not participants:
        return 0

    encoder, encoder_id = get_pinned_spatial_encoder()
    if not _database_accepts_encoder(db, encoder_id):
        return 0
    early_batch = []
    mid_batch = []
    puuids = []

    for p in participants:
        puuid = p.get("puuid")
        pid = p.get("participantId")
        if not puuid or pid is None:
            continue

        is_red = int(pid) > 5
        early_seq, mid_seq = extract_dual_phase_trajectory(timeline_data, puuid=puuid, is_red_side=is_red)
        early_batch.append(early_seq)
        mid_batch.append(mid_seq)
        puuids.append(puuid)

    if not puuids:
        return 0

    early_np = np.stack(early_batch, axis=0)
    mid_np = np.stack(mid_batch, axis=0)
    emb_matrix = encoder([early_np, mid_np]).numpy().astype(np.float32)

    for puuid, emb_vec in zip(puuids, emb_matrix):
        save_player_spatial_embedding(
            match_id=match_id,
            puuid=puuid,
            game_creation_time=int(game_creation_time),
            embedding=emb_vec.tobytes(),
            encoder_id=encoder_id,
            db_path=db,
        )

    return len(puuids)
