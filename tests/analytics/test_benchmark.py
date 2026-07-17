from analytics.benchmark import percentile, run_benchmark
from analytics.build import build_analytics
from analytics.cli import main


def test_percentile_is_nearest_rank():
    assert percentile([0.01, 0.02, 0.03, 0.04, 0.05], 95) == 0.05


def test_percentile_rejects_empty_values():
    try:
        percentile([], 95)
    except ValueError as exc:
        assert str(exc) == "EMPTY_BENCHMARK_SERIES"
    else:
        raise AssertionError("percentile() should reject an empty series")


def test_benchmark_reports_cache_analysis_and_chat(powered_analytics_pair):
    source_db, analytics_db = powered_analytics_pair

    report = run_benchmark(str(analytics_db), str(source_db), iterations=5)

    assert set(report["p95_ms"]) == {
        "cached_evidence",
        "single_metric",
        "chat_no_llm",
    }
    assert report["iterations"] == 5
    assert report["context"]["platform"]
    assert report["context"]["python"]
    assert report["targets_ms"] == {
        "cached_evidence": 100,
        "single_metric": 2000,
        "chat_no_llm": 3000,
    }
    assert report["passed"] is True
    assert report["available"] is True


def test_benchmark_reports_underpowered_mart(source_db, tmp_path):
    analytics_db = tmp_path / "analytics.db"
    build_analytics(str(source_db), str(analytics_db))

    report = run_benchmark(str(analytics_db), str(source_db), iterations=1)

    assert report["available"] is False
    assert report["code"] == "INSUFFICIENT_MATCHES"
    assert report["p95_ms"] == {}
    assert report["passed"] is False


def test_benchmark_cli_prints_json(powered_analytics_pair, capsys):
    source_db, analytics_db = powered_analytics_pair

    exit_code = main(
        [
            "--source",
            str(source_db),
            "--analytics",
            str(analytics_db),
            "benchmark",
            "--iterations",
            "1",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"cached_evidence"' in output
    assert '"passed": true' in output
