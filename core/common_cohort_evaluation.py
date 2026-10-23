"""Common-cohort metrics, explicit feature failures, and paired uncertainty."""
from __future__ import annotations

import itertools
import numpy as np

from core.training_metrics import evaluate_winner
from core.selective_classification import evaluate_selective_coverage

THRESHOLDS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20)


def score_predictions(labels, probabilities) -> dict:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    metrics = evaluate_winner(y, p, 0.5, diagnostics=True)['model']
    selective = []
    for threshold in THRESHOLDS:
        row = evaluate_selective_coverage(y, p, threshold=threshold)
        count = row['count']
        selective.append({'threshold': threshold, 'count': count, 'coverage': row['coverage'],
                          'risk': 1-row['accuracy'] if count else None,
                          'risk_wilson95': [1-row['wilson_upper'], 1-row['wilson_lower']] if count else None})
    metrics['risk_coverage'] = selective
    return metrics


def paired_block_bootstrap(y, a, b, *, repeats=1000, seed=20261009, block_size=100) -> dict:
    """Moving block bootstrap of paired test observations, keeping chronology.

    Uses the same resampled indices for both models. Intervals are exploratory;
    player dependencies spanning blocks are not eliminated by this procedure.
    """
    y, a, b = (np.asarray(v) for v in (y, a, b))
    if not len(y) or a.shape != y.shape or b.shape != y.shape:
        raise ValueError('Paired predictions require identical nonempty match membership')
    eps = 1e-12
    pa, pb = np.clip(a, eps, 1-eps), np.clip(b, eps, 1-eps)
    differences = {
        'accuracy_a_minus_b': ((a >= .5) == y).astype(float) - ((b >= .5) == y).astype(float),
        'log_loss_a_minus_b': -(y*np.log(pa)+(1-y)*np.log1p(-pa)) + (y*np.log(pb)+(1-y)*np.log1p(-pb)),
        'brier_a_minus_b': (a-y)**2 - (b-y)**2,
    }
    block = min(block_size, max(1, len(y)//2))
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in differences}
    for _ in range(repeats):
        starts = rng.integers(0, len(y)-block+1, size=int(np.ceil(len(y)/block)))
        idx = np.concatenate([np.arange(s, s+block) for s in starts])[:len(y)]
        for k, d in differences.items():
            samples[k].append(float(d[idx].mean()))
    return {'method': 'paired moving chronological block bootstrap', 'block_size': block,
            'repeats': repeats, 'seed': seed, 'dependence_limit': 'Shared players can span blocks; no confirmatory significance claim',
            'deltas': {k: {'estimate': float(d.mean()), 'ci95': np.quantile(samples[k], [.025, .975]).tolist()} for k, d in differences.items()}}


def compare_on_intersection(test_cohort: list[dict], predictions: dict[str, dict[str, float]], failures: dict) -> dict:
    ids = [r['match_id'] for r in test_cohort]
    labels = {r['match_id']: r['winner_label'] for r in test_cohort}
    active = {k: p for k, p in predictions.items() if p}
    intersection = [mid for mid in ids if labels[mid] in (0, 1) and all(mid in p for p in active.values())] if active else []
    coverage = {k: {'initial_count': len(ids), 'inference_count': sum(mid in p for mid in ids),
                    'inference_coverage': sum(mid in p for mid in ids)/len(ids),
                    'exclusion_causes': failures.get(k, {})} for k, p in predictions.items()}
    if not intersection:
        return {'coverage': coverage, 'intersection_count': 0, 'metrics': {}, 'paired_uncertainty': {}}
    y = np.array([labels[mid] for mid in intersection])
    vectors = {k: np.array([p[mid] for mid in intersection]) for k, p in active.items()}
    scored = {k:score_predictions(y,p) for k,p in vectors.items()}
    for score in scored.values():
        for row in score['risk_coverage']:
            row['coverage_intersection'] = row['coverage']
            row['coverage_initial_cohort'] = row['count']/len(ids)
    return {'coverage': coverage, 'intersection_count': len(intersection), 'intersection_match_ids': intersection,
            'metrics': scored,
            'paired_uncertainty': {f'{a} vs {b}': paired_block_bootstrap(y, vectors[a], vectors[b])
                                   for a, b in itertools.combinations(vectors, 2)}}
