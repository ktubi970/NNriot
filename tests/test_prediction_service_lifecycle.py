"""The shared facade loads once, reloads on artifact changes and fails closed."""
import json
import os
import threading
import time

import pytest

import core.prediction_service as serving
from core.serving_contract import ContractError
from core.serving_guards import ServiceLoading
from tests.serving_artifact_fixture import build_serving_artifacts, write_json
from tests.test_serving_validation_gates import FixedWinnerModel


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(serving, '_GLOBAL_SERVICE', None)
    monkeypatch.setattr(serving, '_LAST_FAILURE', None)
    monkeypatch.setattr('tensorflow.keras.models.load_model', lambda *a, **k: FixedWinnerModel())
    return build_serving_artifacts(tmp_path)


def _get(artifacts, **kwargs):
    return serving.get_prediction_service(
        checkpoint_path=artifacts['checkpoint'], manifest_path=artifacts['manifest'], **kwargs,
    )


def _count_manifest_loads(monkeypatch):
    calls = []
    original = serving.load_model_manifest
    def counting(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(serving, 'load_model_manifest', counting)
    return calls


def _rewrite_manifest(artifacts, change):
    manifest = json.loads(artifacts['manifest'].read_text())
    change(manifest)
    write_json(artifacts['manifest'], manifest)


def test_cached_service_is_reused_and_manifest_verified_once(artifacts, monkeypatch):
    calls = _count_manifest_loads(monkeypatch)
    first = _get(artifacts)
    assert _get(artifacts) is first
    assert len(calls) == 1


def test_changed_manifest_reloads_without_restart(artifacts):
    first = _get(artifacts)
    assert first.manifest['serving_validation']['status'] == 'validated'
    _rewrite_manifest(artifacts, lambda m: m.update(
        serving_validation={'status': 'unverified', 'independent_evaluation_report': None}))
    second = _get(artifacts)
    assert second is not first
    assert second.manifest['serving_validation']['status'] == 'unverified'


def test_failed_reload_drops_the_stale_service(artifacts):
    _get(artifacts)
    _rewrite_manifest(artifacts, lambda m: m['checkpoint'].update(sha256='0' * 64))
    with pytest.raises(ContractError):
        _get(artifacts)
    assert serving._GLOBAL_SERVICE is None


def test_load_failure_is_replayed_until_artifacts_change(artifacts, monkeypatch):
    original = artifacts['manifest'].read_bytes()
    _rewrite_manifest(artifacts, lambda m: m['checkpoint'].update(sha256='0' * 64))
    calls = _count_manifest_loads(monkeypatch)
    with pytest.raises(ContractError, match='SHA-256'):
        _get(artifacts)
    with pytest.raises(ContractError, match='cached load failure'):
        _get(artifacts)
    assert len(calls) == 1

    monkeypatch.setattr(serving, 'LOAD_FAILURE_RETRY_SECONDS', 0.0)
    with pytest.raises(ContractError):
        _get(artifacts)
    assert len(calls) == 2

    monkeypatch.setattr(serving, 'LOAD_FAILURE_RETRY_SECONDS', 60.0)
    with pytest.raises(ContractError, match='cached load failure'):
        _get(artifacts)
    # Repairing the manifest (same size, newer mtime) retries at once.
    artifacts['manifest'].write_bytes(original)
    stat = artifacts['manifest'].stat()
    os.utime(artifacts['manifest'], ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    assert _get(artifacts).is_ready()
    assert len(calls) == 3


def test_concurrent_first_requests_load_once(artifacts, monkeypatch):
    constructed = []

    class SlowService(serving.PredictionService):
        def __init__(self, *args, **kwargs):
            constructed.append(1)
            time.sleep(.2)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(serving, 'PredictionService', SlowService)
    results = []
    threads = [threading.Thread(target=lambda: results.append(_get(artifacts))) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(constructed) == 1
    assert len(results) == 3 and all(result is results[0] for result in results)


def test_nonblocking_caller_reports_an_ongoing_load(artifacts, monkeypatch):
    monkeypatch.setattr(serving, 'NONBLOCKING_WAIT_SECONDS', .01)
    serving._SERVICE_LOCK.acquire()
    try:
        with pytest.raises(ServiceLoading):
            _get(artifacts, blocking=False)
    finally:
        serving._SERVICE_LOCK.release()
    assert _get(artifacts, blocking=False).is_ready()
