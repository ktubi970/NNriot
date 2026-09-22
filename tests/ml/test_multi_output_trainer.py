import pytest

pytestmark = [pytest.mark.ml, pytest.mark.slow]

from core.feature_labels import LABEL_KEYS, ALL_LABEL_KEYS


@pytest.fixture(scope="module")
def trainer_module():
    from scripts import continuous_trainer

    return continuous_trainer


def _make_record(labels: dict) -> dict:
    """Mimics the record shape returned by database.get_untrained_records."""
    return {"id": 1, "match_id": "M1", "feature_json": {"foo": "bar"}, "labels_json": labels}


def _full_labels(**overrides) -> dict:
    """Default-valid labels dict, overridable for specific tests.

    Includes all 22 ALL_LABEL_KEYS — the 4 timeline keys default to class 2
    ("neither team reached the threshold"), matching what _build_targets
    produces for records without timeline data.
    """
    base = {
        "winner": 0, "team_b_kill_lead": 0,
        "kill_handicap": 5, "total_kills": 30, "team_a_kills": 17, "team_b_kills": 13,
        "kills_odd": 0,
        "first_blood": 0, "first_baron": 0, "first_inhibitor": 0, "first_tower": 0,
        "total_barons": 1, "total_dragons": 3, "total_towers": 11,
        "both_baron": 0, "both_inhibitor": 0, "both_dragon": 1, "elder_dragon": 0,
        # Timeline kill-threshold defaults
        "first_to_5_kills": 2, "first_to_10_kills": 2,
        "first_to_15_kills": 2, "first_to_20_kills": 2,
    }
    base.update(overrides)
    return base


@pytest.mark.unit
def test_build_targets_shapes_and_keys(trainer_module):
    """All 22 head names present with correct shapes."""
    records = [_make_record(_full_labels()) for _ in range(3)]
    t = trainer_module._build_targets(records)
    assert set(t.keys()) == set(ALL_LABEL_KEYS), "missing or extra head"
    # softmax-2
    for name in ("winner", "team_b_kill_lead"):
        assert t[name].shape == (3, 2)
    # softmax-3 (first_* events + timeline kill thresholds)
    for name in ("first_blood", "first_baron", "first_inhibitor", "first_tower",
                 "first_to_5_kills", "first_to_10_kills",
                 "first_to_15_kills", "first_to_20_kills"):
        assert t[name].shape == (3, 3)
    # sigmoid-1 + regression
    for name in ("kills_odd", "both_baron", "both_inhibitor", "both_dragon", "elder_dragon",
                 "kill_handicap", "total_kills", "team_a_kills", "team_b_kills",
                 "total_barons", "total_dragons", "total_towers"):
        assert t[name].shape == (3, 1)


@pytest.mark.unit
def test_build_targets_defaults_missing_timeline_to_class_2(trainer_module):
    """Records without timeline keys in labels_json get class 2 ('neither')."""
    import numpy as np

    # Build a labels dict with only the 18 original keys (no timeline keys)
    legacy_labels = {k: 0 for k in LABEL_KEYS}
    rec = _make_record(legacy_labels)
    t = trainer_module._build_targets([rec])
    # Each of the 4 timeline heads should be one-hot at class 2
    for name in ("first_to_5_kills", "first_to_10_kills",
                 "first_to_15_kills", "first_to_20_kills"):
        np.testing.assert_array_equal(t[name][0], [0, 0, 1])


@pytest.mark.unit
def test_build_targets_one_hot_encoding(trainer_module):
    """Categorical heads are one-hot encoded correctly."""
    import numpy as np

    r0 = _make_record(_full_labels(winner=0, first_baron=2, team_b_kill_lead=1, first_blood=1))
    t = trainer_module._build_targets([r0])
    np.testing.assert_array_equal(t["winner"][0], [1, 0])
    np.testing.assert_array_equal(t["team_b_kill_lead"][0], [0, 1])
    np.testing.assert_array_equal(t["first_blood"][0], [0, 1, 0])
    np.testing.assert_array_equal(t["first_baron"][0], [0, 0, 1])


@pytest.mark.unit
def test_build_targets_regression_normalized(trainer_module):
    """Regression targets are normalized: (raw - mean) / std."""
    rec = _make_record(_full_labels(total_kills=42, team_a_kills=20, kill_handicap=-7))
    t = trainer_module._build_targets([rec])

    # total_kills: (42 - 30) / 12 = 1.0
    expected_total = (42 - trainer_module.REGRESSION_STATS["total_kills"][0]) / trainer_module.REGRESSION_STATS["total_kills"][1]
    assert abs(t["total_kills"][0, 0] - expected_total) < 1e-5

    # team_a_kills: (20 - 15) / 7
    expected_a = (20 - trainer_module.REGRESSION_STATS["team_a_kills"][0]) / trainer_module.REGRESSION_STATS["team_a_kills"][1]
    assert abs(t["team_a_kills"][0, 0] - expected_a) < 1e-5

    # kill_handicap: (-7 - 0) / 12
    expected_h = (-7 - trainer_module.REGRESSION_STATS["kill_handicap"][0]) / trainer_module.REGRESSION_STATS["kill_handicap"][1]
    assert abs(t["kill_handicap"][0, 0] - expected_h) < 1e-5


@pytest.mark.unit
def test_sparse_microbatch_generator_with_dict_targets(trainer_module):
    """Generator yields dict batches when given dict y."""
    import numpy as np

    from scipy.sparse import csr_matrix
    x_sparse = csr_matrix(np.eye(10, 5, dtype=np.float32))
    y_dict = {
        "winner": np.eye(10, 2, dtype=np.float32),
        "total_kills": np.arange(10, dtype=np.float32).reshape(-1, 1),
    }
    batches = list(trainer_module.sparse_microbatch_generator(x_sparse, y_dict, batch_size=3))
    assert len(batches) == 4  # 10 samples / batch_size 3 → 4 batches (last is 1)
    for x_batch, y_batch in batches:
        assert isinstance(y_batch, dict)
        assert set(y_batch.keys()) == {"winner", "total_kills"}
        assert y_batch["winner"].shape[0] == x_batch.shape[0]
        assert y_batch["total_kills"].shape[0] == x_batch.shape[0]


@pytest.mark.integration
def test_validate_checkpoint_input_dim_mismatch(tmp_path):
    """ModelFactory rebuilds fresh model if checkpoint has wrong input dim."""
    from core.model_factory import ModelFactory
    from core.generate_graph import build_multi_output_model

    # Save a model with the WRONG dim into tmp_path
    wrong = build_multi_output_model(input_dim=999)
    save_path = str(tmp_path / "model_v2.keras")
    wrong.save(save_path)

    # Calling ModelFactory with expected dim=20000 should detect mismatch and rebuild fresh model
    model, path = ModelFactory.get_model(version="v2", input_dim=20000, checkpoint_dir=str(tmp_path))
    assert model.input_shape[-1] == 20000


@pytest.mark.integration
def test_validate_checkpoint_correct_arch_passes(tmp_path):
    """ModelFactory loads correctly when input dim and heads match."""
    from core.model_factory import ModelFactory
    from core.generate_graph import build_multi_output_model

    correct = build_multi_output_model(input_dim=1000)
    save_path = str(tmp_path / "model_v2.keras")
    correct.save(save_path)

    model, path = ModelFactory.get_model(version="v2", input_dim=1000, checkpoint_dir=str(tmp_path))
    assert model.input_shape[-1] == 1000


@pytest.mark.slow
@pytest.mark.ml
@pytest.mark.integration
def test_continuous_trainer_initializes_all_output_heads():
    from core.feature_labels import ALL_LABEL_KEYS
    from scripts.continuous_trainer import ContinuousTrainer

    trainer = ContinuousTrainer()

    assert set(trainer.model.output_names) == set(ALL_LABEL_KEYS)
