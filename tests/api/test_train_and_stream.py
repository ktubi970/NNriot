import os

import base64
import json
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

from unittest.mock import MagicMock, patch
from core import database


@pytest.fixture
def app_modules():
    from app import app as flask_app
    from app import core as app_core

    return flask_app, app_core


@pytest.fixture
def client(app_modules):
    flask_app, app_core = app_modules

    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def test_manual_train_fails_fast_without_marking_records(client, tmp_path, app_modules):
    flask_app, app_core = app_modules

    test_db = str(tmp_path / "test_train.db")
    database.init_db(test_db)
    with database.get_connection(test_db) as conn:
        conn.execute("INSERT INTO matches (match_id, game_mode) VALUES (?, ?)", ("MATCH_1", "CLASSIC"))
    database.save_training_record("MATCH_1", {"f": 1}, 0, db_path=test_db)

    with patch.dict("os.environ", {"NNRIOT_DB_PATH": test_db}):
        with patch.object(app_core, "tf_available", False):
            with patch.object(app_core, "global_trainer", None):
                resp = client.post("/api/train/manual")
                assert resp.status_code == 503
                data = resp.get_json()
                assert "unavailable" in data["error"].lower() or "not loaded" in data["error"].lower()

                untrained = database.get_untrained_records(limit=10, db_path=test_db)
                assert len(untrained) == 1
                assert untrained[0]["match_id"] == "MATCH_1"


def test_manual_train_success_with_trainer(client, app_modules):
    flask_app, app_core = app_modules

    fake_trainer = MagicMock()
    fake_trainer.run_training_step.return_value = 5
    with patch.object(app_core, "tf_available", True):
        with patch.object(app_core, "global_trainer", fake_trainer):
            resp = client.post("/api/train/manual")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["success"] is True
            assert data["records_processed"] == 5
            fake_trainer.run_training_step.assert_called_once()


def test_stream_ocr_unique_capture_filenames(client, tmp_path, monkeypatch, app_modules):
    flask_app, app_core = app_modules

    fake_img = "data:image/jpeg;base64," + base64.b64encode(b"dummy_jpeg_data").decode("utf-8")
    static_dir = tmp_path / "static"
    monkeypatch.setenv("MOCK_OCR_ENABLED", "1")
    monkeypatch.setattr(flask_app, "_static_folder", str(static_dir))
    resp1 = client.post("/api/stream/process_frame", json={"image": fake_img})
    resp2 = client.post("/api/stream/process_frame", json={"image": fake_img})
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    url1 = resp1.get_json()["screenshot_url"]
    url2 = resp2.get_json()["screenshot_url"]
    assert url1 != url2, f"Collision detected: {url1} == {url2}"
    assert "capture_" in url1
    assert "capture_" in url2
    assert (static_dir / "screenshots" / os.path.basename(url1)).exists()
    assert (static_dir / "screenshots" / os.path.basename(url2)).exists()
