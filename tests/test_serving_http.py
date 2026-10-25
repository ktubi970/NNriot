import sqlite3
from types import SimpleNamespace

import pytest

from core.serving_contract import ContractError, FeatureUnavailableError
from core.serving_guards import ModelOutputError, ServiceLoading


def payload():
    roles = ['TOP','JUNGLE','MIDDLE','BOTTOM','UTILITY']
    return {'blue_team':[{'puuid':f'a{i}','championName':'Ahri','role':r} for i,r in enumerate(roles)],
            'red_team':[{'puuid':f'b{i}','championName':'Zed','role':r} for i,r in enumerate(roles)]}


def test_custom_prediction_uses_facade_without_legacy_trainer(monkeypatch):
    from app import app
    from app.routes import api
    from app import core
    captured = {}
    def predict(match):
        captured.update(match)
        return {'success':True,'winner':{'team_a':.4,'team_b':.6,'predicted':'team_b','confidence':.6},
                'selective_decision':{'is_actionable':False,'abstained':True},
                'probabilities_raw':{'team_a':.4,'team_b':.6}, 'probabilities_calibrated':None,
                'calibration':{'status':'unavailable'},'draft_advantage':None,'model':{'id':'reference'}}
    monkeypatch.setattr(core,'get_prediction_service',lambda **kwargs:SimpleNamespace(predict_match=predict))
    monkeypatch.setattr(core,'global_trainer',None)
    monkeypatch.setattr(core,'tf_available',False)
    monkeypatch.setattr(api.database,'get_player_stats',lambda *args,**kwargs:{})
    monkeypatch.setattr(api.database,'get_stale_puuids',lambda *args,**kwargs:[])
    response = app.test_client().post('/api/predict/custom',json=payload())
    assert response.status_code == 200
    data=response.get_json()
    assert data['probabilities_calibrated'] is None
    assert data['selective_decision']['abstained']
    assert captured['team_a'][0]['puuid']=='a0'
    assert 'career_win_rate' not in captured['team_a'][0]


def test_custom_model_contract_error_has_no_fallback(monkeypatch):
    from app import app, core
    def failed(**kwargs):raise ContractError('checkpoint SHA-256 mismatch')
    monkeypatch.setattr(core,'get_prediction_service',failed)
    response=app.test_client().post('/api/predict/custom',json=payload())
    assert response.status_code == 503
    assert response.get_json()['error_code']=='model_contract'


def test_status_reports_serving_failure_even_when_legacy_trainer_exists(monkeypatch):
    from app import app, core
    from app.routes import api
    def failed(**kwargs):
        raise ContractError('checkpoint SHA-256 mismatch')
    monkeypatch.setattr(core, 'get_prediction_service', failed)
    monkeypatch.setattr(core, 'global_trainer', object())
    monkeypatch.setattr(core, 'tf_available', True)
    data = app.test_client().get('/api/status').get_json()
    assert data['model_loaded'] is False
    assert data['trainer_loaded'] is True
    assert data['input_dimension'] is None
    assert data['error_code'] == 'model_contract'


def _endpoint_payload(endpoint):
    from tests.serving_fixture import full_match
    return full_match() if endpoint == '/api/predict' else payload()


def _install_predict(monkeypatch, predict):
    from app import core
    from app.routes import api
    monkeypatch.setattr(core, 'get_prediction_service', lambda **kwargs: SimpleNamespace(predict_match=predict))
    monkeypatch.setattr(api.database, 'get_stale_puuids', lambda *args, **kwargs: [])


@pytest.mark.parametrize('endpoint', ['/api/predict', '/api/predict/custom'])
@pytest.mark.parametrize('error,code,status', [
    (FeatureUnavailableError('18D timeline history source unavailable'), 'feature_unavailable', 503),
    (ModelOutputError('winner must contain normalized finite probabilities'), 'model_output', 503),
    (sqlite3.OperationalError('database is locked'), 'feature_unavailable', 503),
    (ContractError('team_a roles must contain each canonical role exactly once'), 'feature_contract', 400),
])
def test_prediction_endpoints_share_error_mapping(monkeypatch, endpoint, error, code, status):
    from app import app
    def predict(match):
        raise error
    _install_predict(monkeypatch, predict)
    response = app.test_client().post(endpoint, json=_endpoint_payload(endpoint))
    assert response.status_code == status
    data = response.get_json()
    assert data['error_code'] == code
    assert data['is_actionable'] is False


@pytest.mark.parametrize('endpoint', ['/api/predict', '/api/predict/custom'])
def test_missing_tensorflow_is_a_model_contract_failure(monkeypatch, endpoint):
    from app import app, core
    from app.routes import api
    def failed(**kwargs):
        raise ModuleNotFoundError("No module named 'tensorflow'")
    monkeypatch.setattr(core, 'get_prediction_service', failed)
    monkeypatch.setattr(api.database, 'get_stale_puuids', lambda *args, **kwargs: [])
    response = app.test_client().post(endpoint, json=_endpoint_payload(endpoint))
    assert response.status_code == 503
    assert response.get_json()['error_code'] == 'model_contract'


@pytest.mark.parametrize('mutation', ['team_id_only', 'missing', 'string'])
def test_raw_participants_require_integer_team_ids(monkeypatch, mutation):
    from app import app
    from tests.serving_fixture import full_match
    called = []
    _install_predict(monkeypatch, called.append)
    match = full_match()
    participant = match['participants'][1]
    if mutation == 'team_id_only':
        participant['team_id'] = participant.pop('teamId')
    elif mutation == 'missing':
        participant.pop('teamId')
    else:
        participant['teamId'] = '100'
    response = app.test_client().post('/api/predict', json=match)
    assert response.status_code == 400
    assert response.get_json()['error_code'] == 'feature_contract'
    assert not called


def test_raw_participant_identities_stay_aligned_with_champions(monkeypatch):
    from app import app
    from tests.serving_fixture import CHAMPIONS, full_match
    captured = {}
    def predict(match):
        captured.update(match)
        return {'success': True}
    _install_predict(monkeypatch, predict)
    assert app.test_client().post('/api/predict', json=full_match()).status_code == 200
    for key, team_id in (('team_a', 100), ('team_b', 200)):
        assert [p['puuid'] for p in captured[key]] == [f'{team_id}_{i}' for i in range(5)]
        assert [p['champion'] for p in captured[key]] == CHAMPIONS[team_id]


def test_status_reports_loading_instead_of_waiting(monkeypatch):
    from app import app, core
    def loading(**kwargs):
        assert kwargs['blocking'] is False
        raise ServiceLoading('Prediction model is loading')
    monkeypatch.setattr(core, 'get_prediction_service', loading)
    data = app.test_client().get('/api/status').get_json()
    assert data['model_loaded'] is False
    assert data['status'] == 'Model loading'
    assert data['error_code'] == 'model_loading'
