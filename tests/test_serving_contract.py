import json
from types import SimpleNamespace

import numpy as np
import pytest

from core.serving_contract import ContractError, FeatureContract, load_model_manifest, validate_probabilities


def match():
    roles = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
    return {k: [{'champion': 'Ahri', 'role': r} for r in roles] for k in ('team_a', 'team_b')}


def test_contract_rejects_wrong_priors_and_duplicate_roles():
    c = FeatureContract('nnriot.players14.v1')
    m = match(); m['team_a'][0]['priors'] = [0.5] * 6
    with pytest.raises(ContractError, match='exactly 10'):
        c.validate_match(m)
    m = match(); m['team_a'][0]['role'] = 'JUNGLE'
    with pytest.raises(ContractError, match='roles'):
        c.validate_match(m)


def test_18d_requires_available_history():
    c = FeatureContract('nnriot.players18.v1')
    with pytest.raises(ContractError, match='history'):
        c.validate_match(match())
    m = match()
    m['priors_provenance'] = {'policy': 'result_end_strictly_before_start.v1', 'as_of_ms': 20, 'latest_result_end_ms': 20}
    for k in ('team_a', 'team_b'):
        for p in m[k]: p['priors'] = [0.5] * 14
    with pytest.raises(ContractError, match='strictly before'):
        c.validate_match(m)


def test_shape_and_probability_validation():
    c = FeatureContract('nnriot.players14.v1')
    model = SimpleNamespace(inputs=[SimpleNamespace(name=k, shape=(None, 5, 18)) for k in ('team_a', 'team_b')])
    with pytest.raises(ContractError, match='shape'):
        c.validate_model(model)
    with pytest.raises(ContractError, match='winner'):
        validate_probabilities(np.array([[0.9, 0.9]]), 'winner')


def test_manifest_rejects_checkpoint_tamper(tmp_path):
    ck = tmp_path/'model.keras'; ck.write_bytes(b'changed')
    p = tmp_path/'manifest.json'
    p.write_text(json.dumps({'schema_version': 1, 'model_id': 'x', 'checkpoint': {'path': str(ck), 'sha256': '0'*64}, 'feature_contract': 'nnriot.players14.v1', 'calibration': None}))
    with pytest.raises(ContractError, match='SHA-256'):
        load_model_manifest(p)


def test_undated_tracker_cannot_supply_live_priors():
    from core.features.pipeline import DraftFeaturePipeline
    from core.player_priors import PlayerPriorsTracker
    c = FeatureContract('nnriot.players14.v1')
    with pytest.raises(ContractError, match='undated'):
        c.extract(match(), DraftFeaturePipeline(player_tracker=PlayerPriorsTracker()), macro=False)


def test_duplicate_identity_is_not_a_valid_match():
    m = match()
    for key in ('team_a', 'team_b'):
        for p in m[key]: p['puuid'] = 'same-player'
    with pytest.raises(ContractError, match='identity'):
        FeatureContract('nnriot.players14.v1').validate_match(m)


def test_numpy_priors_are_rejected_instead_of_silently_ignored():
    m=match(); m['team_a'][0]['priors']=np.array([.9]*10)
    with pytest.raises(ContractError, match='list or tuple'):
        FeatureContract('nnriot.players14.v1').validate_match(m)


def test_18d_missing_timeline_source_is_rejected():
    from core.pre_match_history import ResultOrderedHistory
    m=ResultOrderedHistory().enrich(match(),include_timeline=True,as_of_ms=10)
    with pytest.raises(ContractError, match='source unavailable'):
        FeatureContract('nnriot.players18.v1').validate_match(m)


def test_legacy_encoder_config_keeps_original_architecture():
    from core.deepsets_layers import SiameseTeamEncoder
    encoder=SiameseTeamEncoder.from_config({'embed_dim':8,'latent_dim':16,'use_cross_attention':False})
    assert not encoder.use_set_transformer
    assert not encoder.use_sinkhorn_attention
    assert not encoder.use_periodic_embeddings
    with pytest.raises(ValueError,match='incomplete'):
        SiameseTeamEncoder.from_config({'use_set_transformer':True})


def test_calibration_with_failed_fit_is_rejected_even_when_hashes_match(tmp_path):
    from core.calibration import sha256_file
    from core.serving_contract import runtime_feature_sources
    ck=tmp_path/'model.keras';ck.write_bytes(b'checkpoint')
    split=ck.with_suffix('.split.json');split.write_text('{}')
    cal=ck.with_suffix('.calibration.json')
    cal.write_text(json.dumps({'schema_version':99,'method':'invented','input_dim':46,'feature_policy':'postgame',
                              'temperature':1.,'checkpoint_sha256':sha256_file(ck),
                              'split_manifest_sha256':sha256_file(split),'head':'winner','class_order':['A','B'],
                              'positive_class':1,'fit':{'partition':'validation','status':'failed','count':0}}))
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps({'schema_version':1,'model_id':'bad-calibration',
         'checkpoint':{'path':str(ck),'sha256':sha256_file(ck)},'feature_contract':'nnriot.players14.v1',
         'runtime_feature_sources':runtime_feature_sources(),
         'calibration':{'path':str(cal),'sha256':sha256_file(cal)}}))
    with pytest.raises(ContractError,match='verification'):
        load_model_manifest(mp)
