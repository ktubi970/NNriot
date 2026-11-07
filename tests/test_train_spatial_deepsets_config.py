import pytest
import tensorflow as tf
from scripts.train_spatial_deepsets import get_adjusted_loss_weights


def test_auxiliary_loss_weights_downscaled():
    weights = get_adjusted_loss_weights(aux_scale=0.1)
    assert weights["winner"] == 1.0
    assert weights["team_b_kill_lead"] == 1.0
    assert weights["total_kills"] <= 0.1
    assert weights["team_a_kills"] <= 0.1
    assert weights["team_b_kills"] <= 0.1
    assert weights["kill_handicap"] <= 0.2


def test_auxiliary_loss_weights_default_preserves_ratios():
    weights = get_adjusted_loss_weights(aux_scale=1.0)
    assert weights["winner"] == 1.0
    assert weights["total_kills"] == 1.0
    assert weights["first_blood"] == 0.7


def test_cosine_decay_scheduler():
    steps = 100
    schedule = tf.keras.optimizers.schedules.CosineDecay(0.0003, decay_steps=steps, alpha=0.01)
    lr_0 = float(schedule(0))
    lr_end = float(schedule(steps))
    assert lr_0 == pytest.approx(0.0003)
    assert lr_end == pytest.approx(0.0003 * 0.01)
