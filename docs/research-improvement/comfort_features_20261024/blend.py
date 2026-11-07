"""Convex blend of saved prediction files; weight chosen by log-loss on the first file set (dev)."""
import json
import sys

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
assert (a['ids'] == b['ids']).all()
y = a['y']
rows = []
for w in np.round(np.arange(0, 1.01, 0.1), 2):
    p = w * a['p'] + (1 - w) * b['p']
    rows.append((log_loss(y, p), w, ((p > .5) == y).mean(), roc_auc_score(y, p)))
for r in rows:
    print(f'w_first={r[1]:.1f} logloss={r[0]:.5f} acc={r[2]:.4f} auc={r[3]:.4f}')
best = min(rows)
print(json.dumps({'chosen_w_first': float(best[1]), 'logloss': round(best[0], 5), 'acc': round(float(best[2]), 4)}))
