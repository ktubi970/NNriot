"""Tests for DualPhaseTrajectoryGRU and spatial sequence extraction."""
import numpy as np
import pytest
import tensorflow as tf

from core.spatial_rnn import (
    normalize_spatial_coordinates,
    extract_dual_phase_trajectory,
    build_dual_phase_trajectory_encoder,
    EARLY_TIMESTEPS,
    MID_TIMESTEPS,
    SPATIAL_FEATURE_DIM,
    EARLY_EMBEDDING_DIM,
    MID_EMBEDDING_DIM,
    TOTAL_SPATIAL_EMBEDDING_DIM,
)


def test_spatial_normalization_blue_side():
    # Base blue (0, 0)
    x_norm, y_norm = normalize_spatial_coordinates(0, 0, is_red_side=False)
    assert x_norm == 0.0
    assert y_norm == 0.0

    # Top right (15000, 15000)
    x_norm, y_norm = normalize_spatial_coordinates(15000, 15000, is_red_side=False)
    assert x_norm == 1.0
    assert y_norm == 1.0


def test_spatial_normalization_red_side_invariance():
    # Base red (15000, 15000) for a red player should map to (0, 0)
    x_norm, y_norm = normalize_spatial_coordinates(15000, 15000, is_red_side=True)
    assert x_norm == 0.0
    assert y_norm == 0.0

    # Base blue (0, 0) for a red player should map to (1.0, 1.0)
    x_norm, y_norm = normalize_spatial_coordinates(0, 0, is_red_side=True)
    assert x_norm == 1.0
    assert y_norm == 1.0


def test_extract_dual_phase_trajectory_shapes():
    # Mock Riot timeline data with 20 frames
    frames = []
    for m in range(20):
        frames.append({
            "participantFrames": {
                "1": {
                    "participantId": 1,
                    "position": {"x": 500 * m, "y": 500 * m},
                }
            },
            "events": [],
        })
    timeline_data = {
        "info": {
            "participants": [{"participantId": 1, "puuid": "PLAYER_1"}],
            "frames": frames,
        }
    }

    early_seq, mid_seq = extract_dual_phase_trajectory(timeline_data, puuid="PLAYER_1", is_red_side=False)

    assert early_seq.shape == (EARLY_TIMESTEPS, SPATIAL_FEATURE_DIM)
    assert mid_seq.shape == (MID_TIMESTEPS, SPATIAL_FEATURE_DIM)

    # Frame 0 to 14 should be populated in early_seq
    assert not np.all(early_seq == 0.0)

    # Frame 15 to 19 populated, 20 to 24 should be padded with zeros (masking)
    assert not np.all(mid_seq[:5] == 0.0)
    assert np.all(mid_seq[5:] == 0.0)


def test_dual_phase_trajectory_encoder_forward():
    model = build_dual_phase_trajectory_encoder()

    batch_size = 4
    early_inputs = np.random.uniform(0.0, 1.0, size=(batch_size, EARLY_TIMESTEPS, SPATIAL_FEATURE_DIM)).astype(np.float32)
    mid_inputs = np.random.uniform(0.0, 1.0, size=(batch_size, MID_TIMESTEPS, SPATIAL_FEATURE_DIM)).astype(np.float32)
    # Mask half of the batch in mid game
    mid_inputs[2:, 5:, :] = 0.0

    out = model([early_inputs, mid_inputs])
    assert out.shape == (batch_size, TOTAL_SPATIAL_EMBEDDING_DIM)
    assert TOTAL_SPATIAL_EMBEDDING_DIM == EARLY_EMBEDDING_DIM + MID_EMBEDDING_DIM
    assert not np.isnan(out.numpy()).any()
