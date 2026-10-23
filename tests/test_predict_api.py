import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pytest

from app import app as flask_app


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Use the real serving facade with verified synthetic artifacts."""
    from tests.serving_fixture import install_serving_fixture
    install_serving_fixture(monkeypatch, _fake_preds(), tmp_path)
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def _fake_preds():
    """Synthetic Keras-dict output for a single match (batch size 1)."""
    return {
        "winner":           np.array([[0.40, 0.60]], dtype=np.float32),
        "team_b_kill_lead":     np.array([[0.45, 0.55]], dtype=np.float32),
        "first_blood":      np.array([[0.50, 0.45, 0.05]], dtype=np.float32),
        "first_baron":      np.array([[0.40, 0.45, 0.15]], dtype=np.float32),
        "first_inhibitor":  np.array([[0.42, 0.43, 0.15]], dtype=np.float32),
        "first_tower":      np.array([[0.50, 0.48, 0.02]], dtype=np.float32),
        "kills_odd":        np.array([[0.51]], dtype=np.float32),
        "both_baron":       np.array([[0.22]], dtype=np.float32),
        "both_inhibitor":   np.array([[0.41]], dtype=np.float32),
        "both_dragon":      np.array([[0.93]], dtype=np.float32),
        "elder_dragon":     np.array([[0.18]], dtype=np.float32),
        "kill_handicap":    np.array([[-4.2]], dtype=np.float32),
        "total_kills":      np.array([[36.4]], dtype=np.float32),
        "team_a_kills":     np.array([[16.1]], dtype=np.float32),
        "team_b_kills":     np.array([[20.3]], dtype=np.float32),
        "total_barons":     np.array([[1.8]], dtype=np.float32),
        "total_dragons":    np.array([[3.6]], dtype=np.float32),
        "total_towers":     np.array([[14.2]], dtype=np.float32),
        # Timeline kill-threshold heads (3-class softmax; class 2 = "neither")
        "first_to_5_kills":  np.array([[0.30, 0.20, 0.50]], dtype=np.float32),
        "first_to_10_kills": np.array([[0.25, 0.25, 0.50]], dtype=np.float32),
        "first_to_15_kills": np.array([[0.10, 0.10, 0.80]], dtype=np.float32),
        "first_to_20_kills": np.array([[0.05, 0.05, 0.90]], dtype=np.float32),
    }


def _sample_match_payload():
    from tests.serving_fixture import full_match
    return full_match()



def test_predict_returns_multi_output_shape(client):
    resp=client.post('/api/predict',json=_sample_match_payload())
    assert resp.status_code==200
    data=resp.get_json()
    assert data['success']
    assert data['winner']['predicted']=='team_b'
    assert data['probabilities_raw']['team_b']==pytest.approx(.6)
    assert data['probabilities_calibrated']['team_b']==pytest.approx(.6)
    assert data['model']['feature_contract']=='nnriot.players14.v1'
    assert data['selective_decision']['threshold_applied']==.04
    assert set(data['heads_raw'])==set(_fake_preds())



def test_predict_winner_predicted_field(client):
    """A verified model can select team B when conviction clears the threshold."""
    resp = client.post("/api/predict",
                       data=json.dumps(_sample_match_payload()),
                       content_type="application/json")
    data = resp.get_json()
    # Raw probabilities are preserved by the verified T=1 calibration.
    assert data["winner"]["predicted"] == "team_b"
    assert data["winner"]["team_a"] == 0.4
    assert data["winner"]["team_b"] == 0.6


def test_predict_first_event_three_keys(client):
    data=client.post('/api/predict',json=_sample_match_payload()).get_json()
    for head in ('first_blood','first_baron','first_inhibitor','first_tower','first_to_5_kills','first_to_10_kills','first_to_15_kills','first_to_20_kills'):
        assert len(data['heads_raw'][head])==3
        assert sum(data['heads_raw'][head])==pytest.approx(1)



def test_predict_handles_missing_fields(client):
    """Missing 'participants' or 'teams' must return 400."""
    payload = {"participants": [{"teamId": 100}]}  # no 'teams'
    resp = client.post("/api/predict",
                       data=json.dumps(payload),
                       content_type="application/json")
    assert resp.status_code == 400


def test_predict_handles_model_unavailable(client,monkeypatch):
    import core.prediction_service as serving
    from core.serving_contract import ContractError
    def failed(**kwargs):raise ContractError('missing checkpoint manifest')
    monkeypatch.setattr(serving,'get_prediction_service',failed)
    resp=client.post('/api/predict',json=_sample_match_payload())
    assert resp.status_code==503
    assert resp.get_json()['error_code']=='model_contract'



def test_players_search_accepts_q_and_query_param(client):
    """Ensure /api/players/search accepts both ?q= and ?query= parameters and returns 200."""
    resp_q = client.get("/api/players/search?q=Faker")
    assert resp_q.status_code == 200
    data_q = resp_q.get_json()
    assert isinstance(data_q, list)

    resp_query = client.get("/api/players/search?query=Faker")
    assert resp_query.status_code == 200
    data_query = resp_query.get_json()
    assert isinstance(data_query, list)


def test_predict_custom_contract_aliases(client):
    """Ensure /api/predict/custom provides backward-compatible aliases and multi_output."""
    payload = {
        "blue_team": [{"puuid": f"b_{i}", "champion_name": "Aatrox", "role": ["TOP","JUNGLE","MIDDLE","BOTTOM","UTILITY"][i]} for i in range(5)],
        "red_team": [{"puuid": f"r_{i}", "champion_name": "Lux", "role": ["TOP","JUNGLE","MIDDLE","BOTTOM","UTILITY"][i]} for i in range(5)],
        "game_mode": "CLASSIC",
    }
    resp = client.post("/api/predict/custom",
                       data=json.dumps(payload),
                       content_type="application/json")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert "blue_win_prob" in data
    assert "red_win_prob" in data
    assert "blue_win_probability" in data
    assert "red_win_probability" in data
    assert data["blue_win_prob"] == data["blue_win_probability"]
    assert data["red_win_prob"] == data["red_win_probability"]
    assert "multi_output" in data


def test_predict_custom_empty_or_non_json_returns_400(client):
    """POST /api/predict/custom with empty or non-JSON body returns 400 Bad Request."""
    # Empty JSON object
    resp = client.post("/api/predict/custom",
                       data=json.dumps({}),
                       content_type="application/json")
    assert resp.status_code == 400
    assert "error" in resp.get_json()

    # Empty payload
    resp_empty = client.post("/api/predict/custom",
                             data="",
                             content_type="application/json")
    assert resp_empty.status_code == 400
    assert "error" in resp_empty.get_json()

    # Non-JSON content type / plain text
    resp_text = client.post("/api/predict/custom",
                            data="non-json payload",
                            content_type="text/plain")
    assert resp_text.status_code == 400
    assert "error" in resp_text.get_json()


def test_players_link_empty_or_non_json_returns_400(client):
    """POST /api/players/link with empty or non-JSON body returns 400 Bad Request."""
    # Empty JSON object
    resp = client.post("/api/players/link",
                       data=json.dumps({}),
                       content_type="application/json")
    assert resp.status_code == 400
    assert "error" in resp.get_json()

    # Empty payload
    resp_empty = client.post("/api/players/link",
                             data="",
                             content_type="application/json")
    assert resp_empty.status_code == 400
    assert "error" in resp_empty.get_json()

    # Non-JSON content type / plain text
    resp_text = client.post("/api/players/link",
                            data="non-json body",
                            content_type="text/plain")
    assert resp_text.status_code == 400
    assert "error" in resp_text.get_json()


def test_players_link_missing_arguments_returns_400(client):
    """POST /api/players/link with missing canonical_puuid or alias_puuid returns 400 Bad Request."""
    resp1 = client.post("/api/players/link",
                        data=json.dumps({"canonical_puuid": "main123"}),
                        content_type="application/json")
    assert resp1.status_code == 400
    assert "error" in resp1.get_json()

    resp2 = client.post("/api/players/link",
                        data=json.dumps({"alias_puuid": "smurf123"}),
                        content_type="application/json")
    assert resp2.status_code == 400
    assert "error" in resp2.get_json()


def test_players_unlink_missing_arguments_returns_400(client):
    """DELETE /api/players/unlink with missing alias_puuid returns 400 Bad Request."""
    resp = client.delete("/api/players/unlink")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_batch_collect_empty_or_non_json_returns_400(client):
    """POST /api/batch/collect with empty or non-JSON body returns 400 Bad Request."""
    resp = client.post("/api/batch/collect", data="", content_type="application/json")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_batch_status_preserves_logs_non_destructive(client):
    """GET /api/batch/status must preserve log history across repeated calls without destructive clearing."""
    from scripts import data_collector

    with data_collector.batch_lock:
        data_collector.batch_status["logs"] = [f"log entry {i}" for i in range(120)]
        data_collector.batch_status["status"] = "idle"

    # First call
    resp1 = client.get("/api/batch/status")
    assert resp1.status_code == 200
    data1 = resp1.get_json()
    assert len(data1["logs"]) == 100
    assert data1["logs"][-1] == "log entry 119"

    # Second call - logs must NOT be cleared to []
    resp2 = client.get("/api/batch/status")
    assert resp2.status_code == 200
    data2 = resp2.get_json()
    assert len(data2["logs"]) == 100
    assert data2["logs"][-1] == "log entry 119"

    # Underlying buffer still contains items
    with data_collector.batch_lock:
        assert len(data_collector.batch_status["logs"]) == 120


def test_predict_custom_uses_pre_match_extraction(client):
    from tests.serving_fixture import custom_match
    from unittest.mock import patch
    from core.pre_match_history import to_serving_features
    with patch('core.pre_match_history.to_serving_features',wraps=to_serving_features) as extract:
        res=client.post('/api/predict/custom',json=custom_match())
        assert res.status_code==200
        assert extract.called
        assert 'career_win_rate' not in extract.call_args[0][0]['info']['participants'][0]



@pytest.mark.parametrize('calibrated, validated, tier', [
    (False, True, 'UNCALIBRATED'),
    (True, False, 'UNVALIDATED_MODEL'),
])
@pytest.mark.parametrize('endpoint', ['/api/predict', '/api/predict/custom'])
def test_predict_endpoints_abstain_without_required_evidence(
    client, monkeypatch, tmp_path, calibrated, validated, tier, endpoint,
):
    from tests.serving_fixture import custom_match, install_serving_fixture
    install_serving_fixture(
        monkeypatch, _fake_preds(), tmp_path / 'missing-evidence',
        calibrated=calibrated, validated=validated,
    )
    payload = custom_match() if endpoint.endswith('/custom') else _sample_match_payload()
    response = client.post(endpoint, json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert data['probabilities_raw']['team_b'] == pytest.approx(.6)
    assert data['calibration']['status'] == ('artifact_verified' if calibrated else 'unavailable')
    if calibrated:
        assert data['probabilities_calibrated']['team_b'] == pytest.approx(.6)
    else:
        assert data['probabilities_calibrated'] is None
    assert data['selective_decision']['abstained'] is True
    assert data['selective_decision']['is_actionable'] is False
    assert data['selective_decision']['tier'] == tier
    winner = data['multi_output']['winner'] if endpoint.endswith('/custom') else data['winner']
    assert winner['predicted'] == 'abstain'
    if endpoint.endswith('/custom'):
        assert data['predicted_winner'] == 'ABSTAIN'



def test_predict_endpoints_share_calibrated_winner_transform(client, monkeypatch, tmp_path):
    from tests.serving_fixture import custom_match, install_serving_fixture
    install_serving_fixture(monkeypatch, _fake_preds(), tmp_path / 'temperature-two', temperature=2.0)
    a=client.post('/api/predict',json=_sample_match_payload()).get_json()
    b=client.post('/api/predict/custom',json=custom_match()).get_json()
    assert a['winner']['team_b']==b['red_win_probability']==pytest.approx(.5505,abs=.0001)
    assert a['heads_raw']['total_barons']==pytest.approx(_fake_preds()['total_barons'][0].tolist())
    assert a['probabilities_raw']['team_b']==pytest.approx(.6)
    assert a['probabilities_raw'] == b['probabilities_raw']
    assert a['probabilities_calibrated'] == b['probabilities_calibrated']
    assert a['calibration'] == b['calibration'] == {'status': 'artifact_verified', 'temperature': 2.0}
    assert a['model']['serving_validation']['status'] == 'validated'
    assert not a['selective_decision']['abstained']
    assert not b['selective_decision']['abstained']
