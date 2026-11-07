"""Score the frozen blend once on the official validation (5 000 matches, 16-17 Oct 2026).

Inputs: LR and LightGBM validation predictions refit on all train, the blend weight and the 50%-coverage
conviction threshold, both fixed on dev before this script runs. Also scores the preregistered LightGBM 18D
baseline on the same matches for a paired comparison.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, r'D:\projet\NNriot')
from core.aligned_artifacts import read_train_validation
from core.tabular_baselines import load_baseline, predict_baseline

lr_path, lgb_path, w, dev_lr, dev_lgb, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
a, b = np.load(lr_path), np.load(lgb_path)
assert (a['ids'] == b['ids']).all()
y, p = a['y'], w * a['p'] + (1 - w) * b['p']
da, db = np.load(dev_lr), np.load(dev_lgb)
pdev = w * da['p'] + (1 - w) * db['p']
thr50 = float(np.quantile(np.abs(pdev - 0.5), 0.5))


def wilson(k, n, z=1.96):
    c = (k + z * z / 2) / (n + z * z); h = z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return [round(float(c - h), 4), round(float(c + h), 4)]


def metrics(yy, pp):
    k = int(((pp > .5) == yy).sum()); n = len(yy)
    return dict(n=n, acc=round(k / n, 4), acc_wilson95=wilson(k, n), auc=round(float(roc_auc_score(yy, pp)), 4),
                logloss=round(float(log_loss(yy, pp)), 5))


x, rec, _ = read_train_validation(Path(r'D:\projet\NNriot\runs\aligned_dataset_20261023_prospective'), 'nnriot.players18.v1')
base = predict_baseline(load_baseline(r'D:\projet\NNriot\runs\prospective_20261023\baseline_lightgbm18.json'), x['validation'])
order = {r['match_id']: i for i, r in enumerate(rec['validation'])}
base = base[[order[i] for i in a['ids']]]
yb = np.array([rec['validation'][order[i]]['labels_json']['winner'] for i in a['ids']])
assert (yb == y).all()
cb, cn = (base > .5) == y, (p > .5) == y
rng = np.random.default_rng(20261024)
blocks = np.array_split(np.arange(len(y)), len(y) // 100)  # feature rows are written in creation order
deltas = []
for _ in range(2000):
    idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
    deltas.append(cn[idx].mean() - cb[idx].mean())
sel = np.abs(p - 0.5) >= thr50
res = dict(blend_weight_lr=w, new_model=metrics(y, p), lr_only=metrics(y, a['p']), lgb_only=metrics(y, b['p']),
           preregistered_lightgbm18=metrics(y, base),
           paired_delta_acc=round(float(cn.mean() - cb.mean()), 4),
           paired_delta_acc_block_bootstrap95=[round(float(q), 4) for q in np.quantile(deltas, [0.025, 0.975])],
           mcnemar_fixed_by_new=int((cn & ~cb).sum()), mcnemar_broken_by_new=int((~cn & cb).sum()),
           selective_50pct_threshold_from_dev=round(thr50, 5),
           selective=dict(coverage=round(float(sel.mean()), 4), **metrics(y[sel], p[sel])),
           majority_rate=round(float(max(y.mean(), 1 - y.mean())), 4))
Path(out).write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
