import json
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.slow
@pytest.mark.ml
def test_report_includes_baselines_calibration_and_untouched_test(
    model_db, tmp_path
):
    from analytics.interpretable import train_and_evaluate

    report = train_and_evaluate(str(model_db), str(tmp_path / "artifacts"))

    assert report.dataset_version
    assert set(report.metrics) >= {"majority", "mean_probability", "logistic"}
    for name in ("log_loss", "brier", "calibration_error", "roc_auc"):
        assert name in report.metrics["logistic"]
        assert len(report.metrics["logistic"]["intervals_95"][name]) == 2
    assert "gold_delta*kills_delta" in report.coefficients
    assert len(report.coefficient_intervals["gold_delta*kills_delta"]) == 2
    assert report.coefficient_interval_method == (
        "centered_influence_function_bootstrap"
    )
    assert report.odds_ratio_scale == (
        "numeric_per_training_sd;categorical_vs_encoder_baseline"
    )
    assert report.direction_agreement["gold_delta"] in {
        "agrees",
        "conditional_only",
    }
    assert report.split_counts["test"] > 0
    saved = json.loads(
        (tmp_path / "artifacts" / "model_report.json").read_text()
    )
    assert saved["retrospective"] is True
    assert (tmp_path / "artifacts" / "interpretable_model.joblib").exists()


@pytest.mark.slow
@pytest.mark.ml
def test_grouped_result_reports_underpowered_instead_of_fallback(
    model_db, tmp_path
):
    from analytics.interpretable import train_and_evaluate

    report = train_and_evaluate(
        str(model_db),
        str(tmp_path / "artifacts"),
        min_grouped_test_matches=500,
    )

    assert report.player_disjoint["available"] is False
    assert report.player_disjoint["code"] == "INSUFFICIENT_MATCHES"


def test_underpowered_final_test_does_not_publish_artifacts(cohort_db, tmp_path):
    from analytics.interpretable import train_and_evaluate

    artifact_dir = tmp_path / "underpowered-artifacts"

    report = train_and_evaluate(str(cohort_db), str(artifact_dir))

    assert report.final_test == {
        "available": False,
        "code": "INSUFFICIENT_MATCHES",
    }
    assert report.metrics == {}
    assert not artifact_dir.exists()
