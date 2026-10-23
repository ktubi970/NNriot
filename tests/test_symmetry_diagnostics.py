"""Executable regressions for diagnostics; no operational checkpoint is opened."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from scripts import train_pre_match_model as training


class DiagnosticModel:
    """Cheap training boundary: complementary draft, deliberately biased winner."""

    def __init__(self, *, draft=True, draft_error=0.0):
        self.output_names = ["winner", "draft_advantage"] if draft else ["winner"]
        self.draft = draft
        self.draft_error = draft_error
        self.prediction_inputs = []

    def train_on_batch(self, x, y, return_dict):
        return {"loss": 0.5}

    def save(self, path):
        Path(path).write_bytes(b"isolated diagnostic fixture")

    def predict_on_batch(self, x):
        self.prediction_inputs.append({key: value.copy() for key, value in x.items()})
        n = len(x["team_a"])
        result = {"winner": np.tile([0.7, 0.3], (n, 1))}
        if self.draft:
            direction = np.sign(x["team_b"][:, 0, 0] - x["team_a"][:, 0, 0])
            p = 0.5 + 0.1 * direction + self.draft_error
            result["draft_advantage"] = np.column_stack([1.0 - p, p])
        return result


def _run_training(tmp_path, monkeypatch, feature_version, model):
    from core import deepsets_dataset, equivariant_model, experiment

    dim = 14 if feature_version == "deepsets_v2" else 8
    records = [
        {"match_id": f"DIAG_{i}", "game_creation_time": 1700000000000 + i * 1000,
         "winner_label": i % 2, "labels_json": {"winner": i % 2},
         "feature_json": {"team_a": [], "team_b": [], "marker": i + 1}}
        for i in range(40)
    ]

    def extract(features):
        markers = np.asarray([row["marker"] for row in features], dtype=np.float32)
        a = np.broadcast_to(markers[:, None, None], (len(markers), 5, dim)).copy()
        return {"team_a": a, "team_b": -a}

    monkeypatch.setattr(deepsets_dataset, "batch_extract_team_tensors", extract)
    monkeypatch.setattr(experiment, "resolve_result_end_times", lambda *a, **kw: {})
    monkeypatch.setattr(
        training, "prepare_ordered_training",
        lambda splits, *a, **kw: (
            {name: extract([r["feature_json"] for r in rows]) for name, rows in splits.items()},
            {}, "test-feature-contract",
        ),
    )
    monkeypatch.setattr(equivariant_model, "build_equivariant_deepsets_model", lambda **kw: model)
    monkeypatch.setattr(tf.keras.models, "load_model", lambda path: model)
    return training.train_model(
        records, epochs=1, batch_size=64, feature_version=feature_version,
        checkpoint_path=str(tmp_path / "candidate" / "model.keras"), use_pcgrad=False,
    ).report


@pytest.mark.parametrize("feature_version", ["deepsets", "deepsets_v2"])
def test_training_diagnoses_validation_draft_and_does_not_repeat_test_inference(
    tmp_path, monkeypatch, feature_version,
):
    model = DiagnosticModel()
    report = _run_training(tmp_path, monkeypatch, feature_version, model)
    diagnostic = report["symmetry_diagnostics"]

    assert diagnostic is not None
    assert diagnostic["head"] == "draft_advantage"
    assert diagnostic["split"] == "validation"
    assert diagnostic["transformation"] == "composition_swap_fixed_sides"
    assert diagnostic["tolerance"] == 1e-5
    assert diagnostic["status"] == "passed"
    assert diagnostic["anti_symmetry_guarantee"] is True
    assert diagnostic["max_symmetry_violation"] < 1e-5
    # The test partition contains markers 35..40. Only its final winner pass
    # may reach inference; the diagnostic must instead swap validation 29..34.
    test_calls = [x for x in model.prediction_inputs if np.min(np.abs(x["team_a"])) >= 35]
    assert len(test_calls) == 1
    swapped = [x for x in model.prediction_inputs if np.all(x["team_a"] < 0)]
    assert len(swapped) == 1
    np.testing.assert_array_equal(-swapped[0]["team_a"][:, 0, 0], np.arange(29, 35))


def test_training_legacy_head_is_unavailable_instead_of_assuming_winner_symmetry(tmp_path, monkeypatch):
    report = _run_training(tmp_path, monkeypatch, "deepsets", DiagnosticModel(draft=False))
    diagnostic = report["symmetry_diagnostics"]
    assert diagnostic["status"] == "unavailable"
    assert diagnostic["reason"] == "draft_advantage_head_missing"
    assert diagnostic["head"] == "draft_advantage"
    assert diagnostic["anti_symmetry_guarantee"] is None
    assert diagnostic["max_symmetry_violation"] is None


def test_training_draft_violation_uses_one_e_minus_five_tolerance(tmp_path, monkeypatch):
    model = DiagnosticModel(draft_error=2e-5)
    report = _run_training(tmp_path, monkeypatch, "deepsets", model)
    diagnostic = report["symmetry_diagnostics"]
    assert diagnostic["status"] == "failed"
    assert diagnostic["anti_symmetry_guarantee"] is False
    assert diagnostic["max_symmetry_violation"] == pytest.approx(4e-5, abs=1e-7)


def _load_compiled_diagnostic():
    path = Path(__file__).parents[1] / "scripts" / "verify_compiled_invariants.py"
    spec = importlib.util.spec_from_file_location("compiled_diagnostic_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compiled_diagnostic_import_does_not_load_a_checkpoint(monkeypatch, capsys):
    def forbidden_load(*args, **kwargs):
        pytest.fail("Importing a diagnostic must not open any checkpoint")

    monkeypatch.setattr(tf.keras.models, "load_model", forbidden_load)
    _load_compiled_diagnostic()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("use_evidential", [True, False])
def test_compiled_diagnostic_reports_actual_mirror_probability(tmp_path, monkeypatch, capsys, use_evidential):
    from core.equivariant_model import build_equivariant_deepsets_model

    model = build_equivariant_deepsets_model(
        player_feature_dim=18, embed_dim=4, latent_dim=8,
        use_evidential=use_evidential, include_macro_clash=True,
    )
    model.get_layer("winner_side_bias").bias.assign([-0.4])
    # Isolated fake deserialization boundary avoids writing or opening a model.
    monkeypatch.setattr(tf.keras.models, "load_model", lambda *a, **kw: model)
    diagnostic = _load_compiled_diagnostic()
    output = capsys.readouterr().out
    if callable(getattr(diagnostic, "main", None)):
        assert diagnostic.main(["--checkpoint", str(tmp_path / "synthetic.keras")]) == 0
        output = capsys.readouterr().out
    probability_line = next(line for line in output.splitlines() if "Blue Winrate" in line)
    reported = float(probability_line.split("Blue Winrate:", 1)[1].strip().rstrip("%")) / 100
    # Independently derive each real head's mirror prediction at draft logit 0.
    if use_evidential:
        expected = (1 + np.logaddexp(0, 0.4)) / (2 + np.logaddexp(0, 0.4) + np.logaddexp(0, -0.4))
    else:
        expected = 1 / (1 + np.exp(-0.8))
    assert reported == pytest.approx(expected, abs=5e-5)
    assert callable(getattr(diagnostic, "main", None)), "Diagnostic needs an explicit entry point"
