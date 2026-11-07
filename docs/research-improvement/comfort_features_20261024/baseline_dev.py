"""Existing 18D tabular features on the same chronological dev split (last 12 000 train matches)."""
import sys, json
sys.path.insert(0, r'D:\projet\NNriot')
from pathlib import Path
import numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
from core.aligned_artifacts import read_train_validation
from core.tabular_baselines import tabular_features, LIGHTGBM_PARAMS
x, rec, _ = read_train_validation(Path(r'D:\projet\NNriot\runs\aligned_dataset_20261023_prospective'), 'nnriot.players18.v1')
X = tabular_features(x['train']); y = np.array([r['labels_json']['winner'] for r in rec['train']])
t = np.array([r['game_creation_time'] for r in rec['train']])
order = np.argsort(t, kind='stable'); fit, dev = order[:-12000], order[-12000:]
np.save(sys.argv[1], np.array([r['match_id'] for r in rec['train']])[dev])
b = lgb.train({**LIGHTGBM_PARAMS, 'seed': 42}, lgb.Dataset(X[fit], y[fit]), 3000,
              valid_sets=[lgb.Dataset(X[dev], y[dev])], callbacks=[lgb.early_stopping(100, verbose=False)])
p = b.predict(X[dev], num_iteration=b.best_iteration)
print(json.dumps(dict(tag='existing18_dev', n=len(dev), acc=float(((p > .5) == y[dev]).mean()), auc=float(roc_auc_score(y[dev], p)),
                      logloss=float(log_loss(y[dev], p)), majority=float(max(y[dev].mean(), 1 - y[dev].mean())), iters=b.best_iteration)))
