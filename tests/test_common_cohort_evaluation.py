import pytest
from core.common_cohort_evaluation import compare_on_intersection, paired_block_bootstrap


def test_failed_features_remain_in_initial_coverage():
    cohort = [{'match_id': str(i), 'winner_label': i%2} for i in range(6)]
    a = {str(i): .8 if i%2 else .2 for i in range(6)}
    b = {k:v for k,v in a.items() if k != '1'}
    report = compare_on_intersection(cohort, {'a':a, 'b':b, 'failed':{}}, {'b':{'missing_roles':1}, 'failed':{'contract':6}})
    assert report['intersection_count'] == 5
    assert report['coverage']['b']['inference_coverage'] == pytest.approx(5/6)
    assert report['coverage']['failed']['inference_coverage'] == 0
    assert report['metrics']['a']['accuracy'] == report['metrics']['b']['accuracy'] == 1
    assert report['metrics']['a']['risk_coverage'][0]['coverage_intersection']==1
    assert report['metrics']['a']['risk_coverage'][0]['coverage_initial_cohort']==pytest.approx(5/6)
    empty = report['metrics']['a']['risk_coverage'][-1]
    assert empty['count'] == 5


def test_paired_identical_predictions_have_zero_delta_and_interval():
    report = paired_block_bootstrap([0,1,0,1], [.2,.8,.4,.6], [.2,.8,.4,.6], repeats=20, block_size=2)
    for row in report['deltas'].values():
        assert row['estimate'] == 0
        assert row['ci95'] == [0,0]


def test_all_failed_candidates_have_no_common_evaluable_matches():
    report = compare_on_intersection([{'match_id': 'm', 'winner_label': 1}], {'failed': {}}, {'failed': {'contract': 1}})
    assert report['intersection_count'] == 0
    assert report['metrics'] == {}


def test_frozen_replay_rejects_changed_runtime_sources(tmp_path):
    import json
    from types import SimpleNamespace
    from scripts.compare_served_candidates import run
    from core.serving_contract import ContractError
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'protocol.json').write_text(json.dumps({'frozen_files': {}, 'runtime_sources_sha256': {'changed': 'hash'}}))
    args = SimpleNamespace(output_dir=tmp_path / 'replay', replay=source)
    with pytest.raises(ContractError, match='runtime source SHA-256 mismatch'):
        run(args)
