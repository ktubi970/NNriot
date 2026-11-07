"""Online (point-in-time) Bradley-Terry ratings: player + player x champion + champion x role.

For each target, the logit is computed from parameters updated only by matches whose result ended
strictly before the target's creation. Prints the standalone dev AUC/acc for a small grid and writes
the chosen configurations' logits for every train/validation target.
"""
import heapq
import json
import math
import pickle
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

RAW, SPLIT, DEV_IDS, OUT = sys.argv[1:5]
ROLES = ('TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY')
raw = pickle.load(open(RAW, 'rb'))
F = {k: i for i, k in enumerate(raw['fields'])}
split = json.load(open(SPLIT))['splits']
targets = {e['match_id'] for name in ('train', 'validation') for e in split[name]}
test_start = min(e['game_creation_time'] for e in split['test'])
dev_ids = set(np.load(DEV_IDS).tolist())
matches = sorted(raw['matches'], key=lambda m: (m['creation'], m['match_id']))


def teams(m):
    t = {100: [None] * 5, 200: [None] * 5}
    for p in m['parts']:
        if p[F['teamPosition']] in ROLES and p[F['teamId']] in t:
            t[p[F['teamId']]][ROLES.index(p[F['teamPosition']])] = p
    return None if any(x is None for s in t.values() for x in s) else t


prepared = []
for m in matches:
    if m['creation'] >= test_start:
        break
    t = teams(m)
    if t is None:
        continue
    keys = {tid: [(p[F['puuid']], (p[F['puuid']], p[F['championName']]), (p[F['championName']], r))
                  for r, p in enumerate(side)] for tid, side in t.items()}
    remake = any(p[F['gameEndedInEarlySurrender']] for p in m['parts'])
    prepared.append((m['creation'], m['end'] if (m['end'] and m['end'] > m['creation']) else None, m['match_id'],
                     keys, int(bool(t[200][0][F['win']])), remake))


def run(lr_p, lr_pc, lr_c, decay):
    th = [defaultdict(float), defaultdict(float), defaultdict(float)]
    cnt = [defaultdict(int), defaultdict(int), defaultdict(int)]
    lrs = (lr_p, lr_pc, lr_c)
    bias = 0.0
    pending, out, serial = [], {}, 0

    def logit(keys):
        z = bias
        for sgn, tid in ((1, 200), (-1, 100)):
            for k3 in keys[tid]:
                z += sgn * (th[0][k3[0]] + th[1][k3[1]] + th[2][k3[2]])
        return z

    for creation, end, mid, keys, y, remake in prepared:
        while pending and pending[0][0] < creation:
            _, _, k, yy = heapq.heappop(pending)
            g = yy - 1 / (1 + math.exp(-logit(k)))
            bias += 0.001 * g
            for sgn, tid in ((1, 200), (-1, 100)):
                for k3 in k[tid]:
                    for j in range(3):
                        cnt[j][k3[j]] += 1
                        th[j][k3[j]] += sgn * lrs[j] * g / (1 + cnt[j][k3[j]]) ** decay
        if mid in targets:
            out[mid] = logit(keys)
        if end is not None and not remake:
            serial += 1
            heapq.heappush(pending, (end, serial, keys, y))
    return out


labels = {mid: y for _, _, mid, _, y, _ in prepared}
grid = [(0.2, 0.0, 0.0, 0.5), (0.4, 0.0, 0.0, 0.5), (0.2, 0.1, 0.0, 0.5), (0.2, 0.2, 0.02, 0.5),
        (0.3, 0.3, 0.02, 0.5), (0.2, 0.2, 0.02, 0.3), (0.1, 0.1, 0.01, 0.0), (0.4, 0.4, 0.03, 0.5)]
results = {}
for cfg in grid:
    out = run(*cfg)
    ids = [m for m in out if m in dev_ids]
    z = np.array([out[m] for m in ids]); yy = np.array([labels[m] for m in ids])
    print(cfg, 'dev auc', round(roc_auc_score(yy, z), 4), 'acc', round(float(((z > 0) == yy).mean()), 4), flush=True)
    results[str(cfg)] = out
pickle.dump(results, open(OUT, 'wb'))
