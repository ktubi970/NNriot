"""Score the locked baselines on the sealed test and compare every candidate (Windows, LightGBM).

Requires lock.json and predictions_deepsets.npz from infer_deepsets.py. Nothing is fitted here:
baselines load their validation-fitted artifacts, and the DeepSets ensemble is the mean of the
three calibrated seed probabilities.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT))
from infer_deepsets import (DATASET, DIMS, FAMILIES, SEEDS, TEST_COUNT, canonical, digest, publish_exclusive,
                            read_json, recheck, require)

BOOTSTRAP = dict(repeats=2000, seed=20261023, block_size=100)


def main():
    import numpy as np
    from core.aligned_artifacts import checked_path, validate_tensor_inputs
    from core.common_cohort_evaluation import paired_block_bootstrap
    from core.selective_classification import DEFAULT_CONVICTION_THRESHOLDS, evaluate_selective_coverage
    from core.tabular_baselines import load_baseline, predict_baseline
    from core.training_metrics import evaluate_winner
    lock = read_json(OUT / 'lock.json')
    bound, lock_sha = lock['bound'], digest(OUT / 'lock.json')
    recheck(bound)
    with np.load(OUT / 'predictions_deepsets.npz', allow_pickle=False) as archive:
        saved = {k: archive[k].copy() for k in archive.files}
    require(str(saved['lock_sha256'].item()) == lock_sha, 'DeepSets predictions belong to another lock')
    manifest = read_json(DATASET / 'dataset.json')
    records = read_json(checked_path(DATASET, manifest, 'sealed_test.json'))
    np.testing.assert_array_equal(saved['match_ids'], [r['match_id'] for r in records])
    np.testing.assert_array_equal(saved['labels'], [r['labels_json']['winner'] for r in records])
    y = saved['labels'].astype(int)
    p = {k: v for k, v in saved.items() if k.startswith('deepsets')}
    rates = set()
    for dim in DIMS:
        with np.load(checked_path(DATASET, manifest, f'sealed_test{dim}.npz'), allow_pickle=False) as archive:
            x = {k: archive[k].copy() for k in archive.files}
        validate_tensor_inputs(x, TEST_COUNT, dim, 'test')
        for family in FAMILIES:
            model = load_baseline(ROOT / bound['candidates'][f'{family}{dim}']['path'])
            require(model['calibration']['temperature'] == bound['candidates'][f'{family}{dim}']['temperature'],
                    'Baseline temperature differs from lock')
            rates.add(model['train_positive_rate'])
            p[f'{family}{dim}_raw'] = predict_baseline(model, x, calibrated=False)
            p[f'{family}{dim}_calibrated'] = predict_baseline(model, x)
        p[f'deepsets{dim}_ensemble_calibrated'] = np.mean([p[f'deepsets{dim}_s{s}_calibrated'] for s in SEEDS], axis=0)
    require(len(rates) == 1, 'Baselines disagree on the training base rate')
    train_rate = rates.pop()
    p['constant'] = np.full(TEST_COUNT, train_rate)
    recheck(bound)
    publish_exclusive(OUT / 'predictions_all.npz', lambda h: np.savez(h, labels=y, match_ids=saved['match_ids'],
                      game_creation_time=saved['game_creation_time'], lock_sha256=np.asarray(lock_sha), **p))

    metrics = {}
    for name, probabilities in p.items():
        score = evaluate_winner(y, probabilities, train_rate, diagnostics=True)
        score['selective_fixed_thresholds'] = [evaluate_selective_coverage(y, probabilities, threshold=t, confidence_level=.95)
                                               for t in DEFAULT_CONVICTION_THRESHOLDS]
        metrics[name] = score
    def contrast(a, b):
        return paired_block_bootstrap(y, p[a], p[b], **BOOTSTRAP)['deltas']
    primary = {f'deepsets18_ensemble - {f}18': contrast('deepsets18_ensemble_calibrated', f'{f}18_calibrated')
               for f in FAMILIES}
    secondary = {f'{name} - constant': contrast(name, 'constant') for name in p
                 if name.endswith('calibrated')}
    for family in ('deepsets', *FAMILIES):
        suffix = '_ensemble_calibrated' if family == 'deepsets' else '_calibrated'
        secondary[f'{family}18 - {family}14'] = contrast(f'{family}18{suffix}', f'{family}14{suffix}')
    for f in FAMILIES:
        secondary[f'deepsets14_ensemble - {f}14'] = contrast('deepsets14_ensemble_calibrated', f'{f}14_calibrated')
    secondary['lightgbm18 - logistic18'] = contrast('lightgbm18_calibrated', 'logistic18_calibrated')
    log_loss = lambda name: metrics[name]['model']['log_loss']
    seeds = {dim: dict(log_loss=[log_loss(f'deepsets{dim}_s{s}_calibrated') for s in SEEDS],
                       auc=[metrics[f'deepsets{dim}_s{s}_calibrated']['model']['roc_auc'] for s in SEEDS])
             for dim in DIMS}
    entirely_below = {k: v['log_loss_a_minus_b']['ci95'][1] < 0 for k, v in primary.items()}
    result = dict(schema_version=1, status='completed', lock_sha256=lock_sha,
                  predictions_sha256=digest(OUT / 'predictions_all.npz'), test_count=TEST_COUNT,
                  test_red_rate=float(y.mean()), train_red_rate=train_rate, bootstrap=BOOTSTRAP,
                  metrics=metrics, primary=primary, secondary=secondary, seed_spread=seeds,
                  invariants=json.loads(str(saved['invariants_json'].item())),
                  decision_rule=bound['protocol']['decision_rule'],
                  deepsets_added_value=all(entirely_below.values()), primary_entirely_below_zero=entirely_below,
                  production_promotion=False)
    publish_exclusive(OUT / 'comparison.json', lambda h: h.write(canonical(result)))
    print(json.dumps({k: (round(log_loss(k), 5), round(metrics[k]['model']['roc_auc'], 4),
                          round(metrics[k]['model']['accuracy'], 4)) for k in metrics}, indent=1))
    print(json.dumps(primary, indent=1))
    print('DeepSets added value under the preregistered rule:', result['deepsets_added_value'])


if __name__ == '__main__':
    main()
