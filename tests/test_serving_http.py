from types import SimpleNamespace

import pytest

from core.serving_contract import ContractError


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
    monkeypatch.setattr(api,'get_prediction_service',lambda **kwargs:SimpleNamespace(predict_match=predict))
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
    from app import app
    from app.routes import api
    def failed(**kwargs):raise ContractError('checkpoint SHA-256 mismatch')
    monkeypatch.setattr(api,'get_prediction_service',failed)
    response=app.test_client().post('/api/predict/custom',json=payload())
    assert response.status_code == 503
    assert response.get_json()['error_code']=='model_contract'


def test_status_reports_serving_failure_even_when_legacy_trainer_exists(monkeypatch):
    from app import app, core
    from app.routes import api
    def failed(**kwargs):
        raise ContractError('checkpoint SHA-256 mismatch')
    monkeypatch.setattr(api, 'get_prediction_service', failed)
    monkeypatch.setattr(core, 'global_trainer', object())
    monkeypatch.setattr(core, 'tf_available', True)
    data = app.test_client().get('/api/status').get_json()
    assert data['model_loaded'] is False
    assert data['trainer_loaded'] is True
    assert data['input_dimension'] is None
    assert data['error_code'] == 'model_contract'
