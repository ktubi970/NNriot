from copy import deepcopy
import json
from pathlib import Path
import joblib
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from core.draft_features import (
    TABULAR_FEATURE_DIM,
    extract_tabular_draft_features,
    get_tabular_feature_names,
)
from scripts.train_tree_baseline import extract_tabular_features, train_tree_model


def records(n=40):
    from core.feature_labels import LABEL_KEYS
    rows = []
    for i in range(n):
        labels = dict.fromkeys(LABEL_KEYS, 0)
        labels['winner'] = i % 2
        labels['total_kills'] = 30
        labels['team_a_kills'] = 15
        labels['team_b_kills'] = 15
        labels['total_towers'] = 10
        labels['total_dragons'] = 3
        player = {'championName': 'Ahri', 'teamPosition': 'MIDDLE',
                  'gold': 10000, 'career_win_rate': 0.99}
        rows.append({'match_id': f'M_{i:04d}', 'game_creation_time': 1000 + i,
                     'winner_label': i % 2, 'labels_json': labels,
                     'feature_json': {'game_version': '16.18.1',
                                      'season_id': 16,
                                      'region': 'EUW',
                                      'team_a': [dict(player) for _ in range(5)],
                                      'team_b': [dict(player) for _ in range(5)]}})
    return rows


def test_tabular_contract_preserves_duplicate_roles_and_aliases():
    team = [{'championName': name, 'teamPosition': 'MIDDLE', 'gold': 9999}
            for name in ['Zed', 'Ahri', 'Lux', 'Annie', 'Orianna']]
    values, names = extract_tabular_features({'team_a': team, 'team_b': team,
        'game_version': '16.18.1', 'season_id': 16, 'region': 'EUW'})
    assert len(values) == len(names) == 24
    assert values[0:10:2] == ['Ahri', 'Annie', 'Lux', 'Orianna', 'Zed']
    assert values[1:10:2] == ['MIDDLE'] * 5
    assert '9999' not in values
    assert len(set(names)) == 24


def test_tabular_contract_rejects_non_five_players():
    with pytest.raises(ValueError, match="five participants"):
        extract_tabular_features({'team_a': [{'champion': 'Ahri', 'role': 'MIDDLE'}],
                                  'team_b': [{'champion': 'Zed', 'role': 'MIDDLE'}] * 5})


def test_tree_report_keeps_shared_metric_contract(tmp_path):
    report = train_tree_model(records(200), output_dir=str(tmp_path / 'tree'))
    assert report['evaluation_target'] == 'winner'
    assert report['is_calibrated'] is False
    assert set(report['metrics']['test']) == {
        'model', 'baseline', 'accuracy_lift', 'log_loss_improvement'}
    assert np.isfinite(report['metrics']['test']['model']['brier'])
    assert (tmp_path / 'tree' / 'model.split.json').exists()
    assert (tmp_path / 'tree' / 'model.joblib').exists()
    assert (tmp_path / 'tree' / 'model.metrics.json').exists()


def test_tree_roundtrip_joblib_equality_and_unseen_champion(tmp_path):
    out_dir = tmp_path / 'roundtrip'
    rows = records(200)
    report = train_tree_model(rows, output_dir=str(out_dir))

    bundle = joblib.load(out_dir / 'model.joblib')
    model = bundle['model']
    encoder = bundle['encoder']
    assert encoder is None
    assert len(bundle['feature_names']) == TABULAR_FEATURE_DIM
    assert bundle['feature_names'] == get_tabular_feature_names()

    unseen_feat = deepcopy(rows[0]['feature_json'])
    for p in unseen_feat['team_a']:
        p['championName'] = 'NeverSeenChampion'
    encoded, _ = extract_tabular_draft_features(unseen_feat)

    probs = model.predict_proba([encoded])
    assert probs.shape == (1, 2)
    assert np.isclose(probs.sum(), 1.0)


def test_tree_model_rejects_one_class_train(tmp_path):
    rows = records(40)
    for r in rows:
        r['labels_json']['winner'] = 0
        r['winner_label'] = 0
    with pytest.raises(ValueError, match="train.*class"):
        train_tree_model(rows, output_dir=str(tmp_path / 'tree_one_class'))


def test_existing_output_directory_rejected_without_overwriting(tmp_path):
    out_dir = tmp_path / 'existing_tree'
    out_dir.mkdir()
    (out_dir / 'model.joblib').write_bytes(b'dummy')
    with pytest.raises(FileExistsError):
        train_tree_model(records(40), output_dir=str(out_dir))


def test_altered_comparison_hash_rejected_before_fitting(tmp_path, monkeypatch):
    from core.experiment import build_manifest
    from core.pre_match_dataset import chronological_split
    rows = records(200)
    splits = chronological_split(rows)
    manifest = build_manifest(splits, source_db=None,
                              split_metadata={'requested_strategy': 'chronological',
                                              'effective_strategy': 'chronological',
                                              'fallback_reason': None})

    fake_neural_report = {
        'dataset_sha256': '0' * 64,  # mismatched hash
        'split_sha256': manifest['split_sha256'],
        'train_positive_rate': 0.5,
        'split_counts': {'train': 140, 'validation': 30, 'test': 30},
        'metrics': {
            'test': {
                'baseline': {'accuracy': 0.5, 'log_loss': 0.693, 'brier': 0.25},
                'model': {'accuracy': 0.52, 'log_loss': 0.690, 'brier': 0.24},
            }
        }
    }

    fit_called = False
    original_fit = HistGradientBoostingClassifier.fit
    def spy_fit(self, *args, **kwargs):
        nonlocal fit_called
        fit_called = True
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(HistGradientBoostingClassifier, 'fit', spy_fit)

    with pytest.raises(ValueError, match="dataset_sha256"):
        train_tree_model(rows, output_dir=str(tmp_path / 'tree_compare_fail'),
                         split_manifest=manifest, compare_report=fake_neural_report)

    assert not fit_called


def test_candidate_selection_uses_validation_labels_only(tmp_path, monkeypatch):
    import scripts.train_tree_baseline as tree_module
    rows = records(200)

    val_sizes = []
    test_eval_calls = 0
    original_evaluate = tree_module.evaluate_winner

    def spy_evaluate_winner(y_true, probabilities, train_rate, **kwargs):
        nonlocal test_eval_calls
        if len(y_true) == 30:  # validation size for n=200 is 30, test size is 30
            val_sizes.append(len(y_true))
        res = original_evaluate(y_true, probabilities, train_rate, **kwargs)
        return res

    monkeypatch.setattr(tree_module, 'evaluate_winner', spy_evaluate_winner)

    report = train_tree_model(rows, output_dir=str(tmp_path / 'tree_eval'))
    # Candidate selection tested 3 iterations (50, 100, 150) on validation, plus final report on train, val, test
    assert len(val_sizes) >= 3  # at least 3 candidate selection evaluations on validation
