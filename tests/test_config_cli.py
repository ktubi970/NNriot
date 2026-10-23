"""Database selection and CLI boundary behavior."""

import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest


def test_db_paths_use_project_dotenv_and_environment_precedence(tmp_path, monkeypatch):
    from core import config

    (tmp_path / ".env").write_text(
        "NNRIOT_DB_PATH=from-env-file.db\n"
        "NNRIOT_ANALYTICS_DB_PATH=analytics-from-file.db\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("NNRIOT_DB_PATH", raising=False)
    monkeypatch.delenv("NNRIOT_ANALYTICS_DB_PATH", raising=False)
    monkeypatch.delenv("ANALYTICS_DB_PATH", raising=False)

    assert config.get_db_path() == str(tmp_path / "from-env-file.db")
    assert config.get_analytics_db_path() == str(tmp_path / "analytics-from-file.db")

    monkeypatch.setenv("NNRIOT_DB_PATH", "from-process.db")
    monkeypatch.setenv("NNRIOT_ANALYTICS_DB_PATH", "analytics-from-process.db")
    assert config.get_db_path() == str(tmp_path / "from-process.db")
    assert config.get_analytics_db_path() == str(tmp_path / "analytics-from-process.db")
    assert config.get_db_path("from-argument.db") == str(tmp_path / "from-argument.db")
    assert config.get_analytics_db_path("analytics-from-argument.db") == str(
        tmp_path / "analytics-from-argument.db"
    )


def test_db_path_requirement_rejects_missing_file_without_creating_it(tmp_path, monkeypatch):
    from core import config

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    missing = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError, match="missing.db"):
        config.get_db_path("missing.db", require_exists=True)
    assert not missing.exists()


def test_cli_train_forwards_resolved_database_and_training_options(tmp_path, monkeypatch):
    import cli

    source = tmp_path / "source.db"
    source.touch()
    received = []
    trainer = ModuleType("scripts.train_pre_match_model")
    trainer.main = lambda argv=None: received.append(argv)
    monkeypatch.setitem(sys.modules, "scripts.train_pre_match_model", trainer)

    cli.main([
        "train", "--db", str(source), "--samples", "120", "--epochs", "2",
        "--batch-size", "8", "--dim", "256", "--output", str(tmp_path / "candidate.keras"),
        "--seed", "7",
    ])

    assert received == [[
        "--db", str(source), "--samples", "120", "--epochs", "2",
        "--batch-size", "8", "--dim", "256", "--output", str(tmp_path / "candidate.keras"),
        "--seed", "7",
    ]]


def test_cli_train_forwards_split_strategy_and_manifest(tmp_path, monkeypatch):
    import cli

    source = tmp_path / "source.db"
    source.touch()
    received = []
    trainer = ModuleType("scripts.train_pre_match_model")
    trainer.main = lambda argv=None: received.append(argv)
    monkeypatch.setitem(sys.modules, "scripts.train_pre_match_model", trainer)

    cli.main([
        "train", "--db", str(source), "--split-strategy", "patch",
    ])
    assert ["--split-strategy", "patch"] == received[-1][-4:-2]

    manifest = tmp_path / "manifest.json"
    manifest.touch()
    cli.main([
        "train", "--db", str(source), "--split-manifest", str(manifest),
    ])
    assert ["--split-manifest", str(manifest)] == received[-1][-4:-2]

    cli.main([
        "train", "--db", str(source), "--calibrate",
    ])
    assert "--calibrate" in received[-1]

    cli.main([
        "train", "--db", str(source), "--feature-version", "v2_draft",
    ])
    assert ["--feature-version", "v2_draft"] == received[-1][-4:-2]


def test_cli_benchmark_signal_forwards_options(tmp_path, monkeypatch):
    import cli

    source = tmp_path / "source.db"
    source.touch()
    received = []
    bench = ModuleType("scripts.benchmark_draft_signal")
    bench.main = lambda argv=None: received.append(argv)
    monkeypatch.setitem(sys.modules, "scripts.benchmark_draft_signal", bench)

    cli.main([
        "benchmark-signal", "--db", str(source), "--samples", "500", "--output", str(tmp_path / "report.json")
    ])
    assert received == [[
        "--db", str(source), "--samples", "500", "--output", str(tmp_path / "report.json")
    ]]


def test_cli_train_rejects_mutually_exclusive_strategy_and_manifest(tmp_path):
    import cli

    source = tmp_path / "source.db"
    source.touch()
    manifest = tmp_path / "manifest.json"
    manifest.touch()

    with pytest.raises(SystemExit):
        cli.main([
            "train", "--db", str(source), "--split-strategy", "patch", "--split-manifest", str(manifest),
        ])


def test_script_train_rejects_mutually_exclusive_strategy_and_manifest_and_missing_manifest(tmp_path):
    from scripts.train_pre_match_model import main as train_main

    source = tmp_path / "source.db"
    source.touch()
    manifest = tmp_path / "manifest.json"
    manifest.touch()

    with pytest.raises(SystemExit) as exc:
        train_main(["--db", str(source), "--split-strategy", "patch", "--split-manifest", str(manifest)])
    assert exc.value.code == 2

    with pytest.raises(SystemExit) as exc2:
        train_main(["--db", str(source), "--split-manifest", str(tmp_path / "nonexistent.json")])
    assert exc2.value.code == 2


def test_train_script_help_does_not_import_tensorflow():
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    process = subprocess.run(
        [sys.executable, "-c", "import sys; from scripts.train_pre_match_model import main; "
         "assert 'tensorflow' not in sys.modules"],
        cwd=project_root, env=env, capture_output=True, text=True,
    )
    assert process.returncode == 0, process.stderr


def test_cli_rejects_missing_explicit_backfill_db_before_import(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    missing = tmp_path / "missing.db"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    process = subprocess.run(
        [sys.executable, "cli.py", "backfill", "--db", str(missing)],
        cwd=project_root, env=env, capture_output=True, text=True,
    )
    assert process.returncode != 0
    assert "missing.db" in process.stderr
    assert not missing.exists()


def test_cli_help_does_not_import_tensorflow(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    process = subprocess.run(
        [sys.executable, "-c", "import sys, cli; cli.main(['train', '--help']); "
         "assert 'tensorflow' not in sys.modules"],
        cwd=project_root, env=env, capture_output=True, text=True,
    )
    assert process.returncode == 0, process.stderr
    assert "--samples" in process.stdout

def test_factory_uses_injected_database_for_schema_and_routes(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    boot = tmp_path / "boot.db"
    boot.touch()
    selected = tmp_path / "selected.db"
    selected.touch()
    mart = tmp_path / "analytics.db"
    env = os.environ.copy()
    env["NNRIOT_DB_PATH"] = str(boot)
    env["NNRIOT_ANALYTICS_DB_PATH"] = str(mart)
    env["PYTHONPATH"] = str(project_root)
    script = (
        "import sqlite3, sys; from app import create_app; "
        f"a=create_app({{'TESTING': True, 'NNRIOT_DB_PATH': {str(selected)!r}, "
        f"'ANALYTICS_DB_PATH': {str(mart)!r}}}); "
        f"assert a.config['NNRIOT_DB_PATH'] == {str(selected)!r}; "
        f"assert a.config['ANALYTICS_DB_PATH'] == {str(mart)!r}; "
        "assert 'tensorflow' not in sys.modules; "
        f"assert sqlite3.connect({str(selected)!r}).execute("
        "\"SELECT name FROM sqlite_master WHERE name='schema_version'\").fetchone(); "
        "assert a.test_client().get('/api/db-health').status_code == 200; "
        f"from core import database; database.save_player('chosen', 'Selected', 'EUW', db_path={str(selected)!r}); "
        "history=a.test_client().get('/api/player/history?name=Selected&tag=EUW').get_json(); "
        "assert history['player']['puuid'] == 'chosen'; "
        "found=a.test_client().get('/api/players/search?q=Selected').get_json(); "
        "assert found and found[0]['puuid'] == 'chosen'"
    )
    process = subprocess.run(
        [sys.executable, "-c", script], cwd=project_root, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert process.returncode == 0, process.stderr



def test_analytics_service_uses_shared_defaults_when_paths_are_omitted(tmp_path, monkeypatch):
    from core import config
    from analytics.service import AnalyticsService

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("NNRIOT_DB_PATH", "source.db")
    monkeypatch.setenv("NNRIOT_ANALYTICS_DB_PATH", "mart.db")
    service = AnalyticsService()
    assert service.source_path == str(tmp_path / "source.db")
    assert service.analytics_path == str(tmp_path / "mart.db")

def test_start_bat_dispatches_train_continuous_help():
    if os.name != "nt":
        pytest.skip("Windows launcher")
    project_root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        # Absolute path: cmd ignores the cwd when NoDefaultCurrentDirectoryInExePath is set.
        ["cmd", "/c", str(project_root / "start.bat"), "train-continuous", "--help"],
        cwd=project_root, capture_output=True, text=True, timeout=30,
    )
    assert process.returncode == 0, process.stderr
    assert "--db" in process.stdout

def test_factory_import_does_not_create_missing_default_database(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    missing_default = tmp_path / "default-missing.db"
    selected = tmp_path / "selected.db"
    selected.touch()
    env = os.environ.copy()
    env["NNRIOT_DB_PATH"] = str(missing_default)
    env["PYTHONPATH"] = str(project_root)
    script = (
        "from app import app, create_app; from app import core; "
        f"a=create_app({{'TESTING': True, 'NNRIOT_DB_PATH': {str(selected)!r}}}); "
        f"assert a.config['NNRIOT_DB_PATH'] == {str(selected)!r}; "
        "assert core.trainer_loading is False"
    )
    process = subprocess.run(
        [sys.executable, "-c", script], cwd=project_root, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert process.returncode == 0, process.stderr
    assert not missing_default.exists()

def test_cli_continuous_trainer_receives_explicit_database(tmp_path, monkeypatch):
    import cli

    source = tmp_path / "source.db"
    source.touch()
    received = []
    trainer = ModuleType("scripts.continuous_trainer")
    trainer.main = lambda db_path=None: received.append(db_path)
    monkeypatch.setitem(sys.modules, "scripts.continuous_trainer", trainer)
    monkeypatch.setenv("NNRIOT_DB_PATH", str(source))

    cli.main(["train-continuous", "--db", str(source)])
    assert received == [str(source)]

@pytest.mark.parametrize("module,extra", [
    ("scripts.backfill_labels", []),
    ("scripts.backfill_participants", ["--max", "0"]),
    ("scripts.backfill_timelines", ["--limit", "0"]),
    ("scripts.cleanup_malformed", []),
    ("scripts.migrate_winner_kills", []),
])
def test_maintenance_command_rejects_missing_explicit_db(module, extra, tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    missing = tmp_path / "missing.db"
    fallback = tmp_path / "fallback.db"
    from core import database
    database.init_db(str(fallback))
    env = os.environ.copy()
    env["NNRIOT_DB_PATH"] = str(fallback)
    env["PYTHONPATH"] = str(project_root)
    process = subprocess.run(
        [sys.executable, "-m", module, "--db", str(missing), *extra],
        cwd=project_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert process.returncode == 2
    assert "does not exist" in process.stderr
    assert not missing.exists()


def test_analytics_cli_rejects_missing_source_before_sqlite_open(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    missing = tmp_path / "missing.db"
    mart = tmp_path / "mart.db"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    process = subprocess.run(
        [sys.executable, "-m", "analytics", "--source", str(missing),
         "--analytics", str(mart), "health"],
        cwd=project_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert process.returncode == 2
    assert "does not exist" in process.stderr
    assert not missing.exists()
    assert not mart.exists()

@pytest.mark.parametrize("module_name,extra", [
    ("scripts.backfill_labels", []),
    ("scripts.backfill_participants", ["--max", "0"]),
    ("scripts.backfill_timelines", ["--limit", "0"]),
    ("scripts.cleanup_malformed", []),
    ("scripts.migrate_winner_kills", []),
])
def test_maintenance_relative_db_is_project_root_relative(
    module_name, extra, tmp_path, monkeypatch, capsys
):
    import importlib
    from core import config, database

    source = tmp_path / "source.db"
    database.init_db(str(source))
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(other)

    module = importlib.import_module(module_name)
    module.main(["--db", "source.db", *extra])
    assert str(source) in capsys.readouterr().out
    assert not (other / "source.db").exists()


def test_analytics_cli_resolves_relative_output_under_project_root(tmp_path, monkeypatch, capsys):
    from core import config, database
    from analytics.cli import main

    source = tmp_path / "source.db"
    database.init_db(str(source))
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(other)
    with pytest.raises(SystemExit) as exc:
        main(["--source", "source.db", "--analytics", "mart.db", "health"])
    assert exc.value.code == 2
    assert str(tmp_path / "mart.db") in capsys.readouterr().err
    assert not (other / "mart.db").exists()

@pytest.mark.parametrize("module", [
    "scripts.backfill_labels",
    "scripts.backfill_participants",
    "scripts.backfill_timelines",
    "scripts.cleanup_malformed",
    "scripts.migrate_winner_kills",
    "analytics",
])
def test_database_maintenance_help_is_read_only(module, tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["NNRIOT_DB_PATH"] = str(tmp_path / "missing.db")
    env["PYTHONPATH"] = str(project_root)
    process = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=project_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert process.returncode == 0, process.stderr
    assert not (tmp_path / "missing.db").exists()
    assert ("--source" if module == "analytics" else "--db") in process.stdout

@pytest.mark.parametrize("script", [
    "backfill_labels.py",
    "backfill_participants.py",
    "backfill_timelines.py",
    "cleanup_malformed.py",
    "migrate_winner_kills.py",
])
def test_maintenance_script_direct_help_without_pythonpath(script, tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["NNRIOT_DB_PATH"] = str(tmp_path / "missing.db")
    process = subprocess.run(
        [sys.executable, str(project_root / "scripts" / script), "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert process.returncode == 0, process.stderr
    assert "--db" in process.stdout
    assert not (tmp_path / "missing.db").exists()
