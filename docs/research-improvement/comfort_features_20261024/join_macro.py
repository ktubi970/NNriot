"""Append the project's 15-dim anti-symmetric macro clash vector (Red - Blue) to a feature table, by match_id."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r'D:\projet\NNriot')
from core.aligned_artifacts import read_train_validation

IN, OUT = sys.argv[1], sys.argv[2]
x, rec, _ = read_train_validation(Path(r'D:\projet\NNriot\runs\aligned_dataset_20261023_prospective'), 'nnriot.players14.v1')
macro = {}
for name in ('train', 'validation'):
    for r, v in zip(rec[name], x[name]['macro_clash']):
        macro[r['match_id']] = v
z = dict(np.load(IN, allow_pickle=False))
M = np.vstack([macro[i] for i in z['ids']]).astype(np.float32)
z['X'] = np.hstack([z['X'], M])
z['names'] = np.concatenate([z['names'], [f'macro{k}_diffrb' for k in range(M.shape[1])]])
np.savez_compressed(OUT, **z)
print('joined', z['X'].shape)
