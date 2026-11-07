"""Dev evaluation with Red-Blue difference features and optional swap augmentation.

Fit on train[:-dev], early stop on the chronological dev tail. --final refits on all train and scores the
official validation once. The sealed test is never featurized.
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

ap = argparse.ArgumentParser()
ap.add_argument('features')
ap.add_argument('--dev', type=int, default=12000)
ap.add_argument('--drop', default='')
ap.add_argument('--swap', action='store_true')
ap.add_argument('--no-raw', action='store_true', help='only diffs/sums, no per-side raw columns')
ap.add_argument('--final', action='store_true')
ap.add_argument('--lr', type=float, default=0.02)
ap.add_argument('--leaves', type=int, default=31)
ap.add_argument('--min-leaf', type=int, default=200)
ap.add_argument('--ff', type=float, default=0.7)
ap.add_argument('--l2', type=float, default=5.0)
ap.add_argument('--seeds', type=int, default=1)
ap.add_argument('--importance', type=int, default=0)
ap.add_argument('--save-pred', default='')
ap.add_argument('--max-bin', type=int, default=255)
ap.add_argument('--es', type=int, default=300)
ap.add_argument('--threads', type=int, default=6)
ap.add_argument('--model', default='lgb', choices=('lgb', 'lr'))
ap.add_argument('--save-dev', default='')
ap.add_argument('--counts', action='store_true')
ap.add_argument('--hier', action='store_true')
ap.add_argument('--half-life', type=float, default=0.0)
args = ap.parse_args()

z = np.load(args.features, allow_pickle=False)
X0, y, split, t, names = z['X'], z['y'], z['split'], z['t'], list(z['names'])
P = list(z['pnames']); NP = len(P); ROLES = ('TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY')
drop = [d for d in args.drop.split(',') if d]
pkeep = [i for i, n in enumerate(P) if not any(d in n for d in drop)]
CAT = {'keystone', 'primary_style', 'sub_style', 'pref_status'}
tail_names = names[10 * NP:]


def build(X, swapped):
    blue = X[:, :5 * NP].reshape(-1, 5, NP); red = X[:, 5 * NP:10 * NP].reshape(-1, 5, NP)
    tail = X[:, 10 * NP:].copy()
    if swapped:
        blue, red = red, blue
        for i, n in enumerate(tail_names):
            if n.endswith('red_wr'):
                tail[:, i] = 1 - tail[:, i]
            elif n.endswith('_diffrb') or n == 'elo_diff':
                tail[:, i] = -tail[:, i]
    num = [i for i in pkeep if P[i] not in CAT]
    cols, out = [], []
    if not args.no_raw:
        out += [blue[:, :, pkeep].reshape(len(X), -1), red[:, :, pkeep].reshape(len(X), -1)]
        cols += [f'blue_{r}_{P[i]}' for r in ROLES for i in pkeep] + [f'red_{r}_{P[i]}' for r in ROLES for i in pkeep]
    diff = (red[:, :, num] - blue[:, :, num])
    out.append(diff.reshape(len(X), -1)); cols += [f'diff_{r}_{P[i]}' for r in ROLES for i in num]
    with np.errstate(all='ignore'):
        out.append(np.nanmean(red[:, :, num], 1) - np.nanmean(blue[:, :, num], 1)); cols += [f'teamdiff_{P[i]}' for i in num]
        out.append(np.nanmin(red[:, :, num], 1) - np.nanmin(blue[:, :, num], 1)); cols += [f'teammindiff_{P[i]}' for i in num]
    if 'pref_status' in [P[i] for i in pkeep]:
        k = P.index('pref_status')
        out.append(((red[:, :, k] == 2).sum(1) - (blue[:, :, k] == 2).sum(1))[:, None]); cols.append('autofill_count_diff')
    if args.counts:
        g = {n: P.index(n) for n in ('log_n', 'champ_log_n', 'role_recent_frac', 'log_level', 'pref_status',
                                     'spell_smite', 'games_24h', 'streak')}
        with np.errstate(invalid='ignore'):
            def cnt(side):
                known = side[:, :, g['log_n']] > 0
                return np.stack([
                    (known & (side[:, :, g['champ_log_n']] == 0)).sum(1),
                    (known & (side[:, :, g['champ_log_n']] <= np.log(4))).sum(1),
                    (known & (side[:, :, g['role_recent_frac']] < 0.25)).sum(1),
                    (side[:, :, g['log_level']] < np.log1p(60)).sum(1),
                    (side[:, :, g['pref_status']] == 1).sum(1),
                    (side[:, 1, g['spell_smite']] == 0).astype(int),
                    (side[:, :, g['games_24h']] >= 6).sum(1),
                    (side[:, :, g['streak']] <= -3).sum(1),
                    (side[:, :, g['streak']] >= 3).sum(1)], 1)
        out.append((cnt(red) - cnt(blue)).astype(np.float32))
        cols += [f'cnt_{k}' for k in ('first_time', 'low_champ', 'off_role', 'low_level', 'secondary',
                                      'jungle_no_smite', 'marathon', 'loss_streak3', 'win_streak3')]
    if args.hier:
        def hier(side):
            cn = np.expm1(side[:, :, P.index('champ_log_n')]); cw = side[:, :, P.index('champ_wr')] * (cn + 10) - 5
            n = np.expm1(side[:, :, P.index('log_n')]); w = side[:, :, P.index('wr')] * (n + 10) - 5
            pw = (w + 20 * 0.5) / (n + 20)
            ch = (cw + 5 * pw) / (cn + 5)
            return np.concatenate([ch, ch - pw, pw], 1)
        out.append((hier(red) - hier(blue)).astype(np.float32))
        cols += [f'hier_{k}_{r}' for k in ('champ_wr', 'champ_delta', 'player_wr') for r in ROLES]
    out.append(tail); cols += tail_names
    out.append(np.full((len(X), 1), float(swapped))); cols.append('swapped_side')
    return np.hstack(out).astype(np.float32), cols


Xo, cols = build(X0, False)
tr = np.where(split == 'train')[0]; tr = tr[np.argsort(t[tr], kind='stable')]
va = np.where(split == 'validation')[0]
fit_idx, dev_idx = tr[:-args.dev], tr[-args.dev:]
if args.swap:
    Xs, _ = build(X0, True)
cat_idx = [i for i, c in enumerate(cols) if any(c.endswith('_' + k) for k in CAT) and not c.startswith(('diff', 'team'))]


def train_set(idx):
    if args.swap:
        return np.vstack([Xo[idx], Xs[idx]]), np.concatenate([y[idx], 1 - y[idx]])
    return Xo[idx], y[idx]


def predict(bst, idx, it=None):
    p = bst.predict(Xo[idx], num_iteration=it)
    if args.swap:
        p = 0.5 * (p + 1 - bst.predict(Xs[idx], num_iteration=it))
    return p


def params(seed):
    return {'objective': 'binary', 'learning_rate': args.lr, 'num_leaves': args.leaves, 'min_data_in_leaf': args.min_leaf,
            'feature_fraction': args.ff, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': args.l2, 'verbose': -1,
            'seed': seed, 'num_threads': args.threads, 'max_bin': args.max_bin, 'deterministic': True, 'force_row_wise': True, 'cat_smooth': 50,
            'max_cat_to_onehot': 8}


def report(tag, yy, p):
    acc = ((p > 0.5) == yy).mean()
    out = dict(tag=tag, n=int(len(yy)), acc=round(float(acc), 4), acc_ci95=round(1.96 * float(np.sqrt(acc * (1 - acc) / len(yy))), 4),
               majority=round(float(max(yy.mean(), 1 - yy.mean())), 4), auc=round(float(roc_auc_score(yy, p)), 4),
               logloss=round(float(log_loss(yy, p)), 5))
    print(json.dumps(out), flush=True)


if args.model == 'lr':
    from sklearn.linear_model import LogisticRegression
    lin = [i for i, c in enumerate(cols) if not c.startswith(('blue_', 'red_')) and c != 'swapped_side']
    A = Xo[:, lin].astype(np.float64)
    mu = np.nanmean(A[fit_idx], 0); sd = np.nanstd(A[fit_idx], 0); sd[sd == 0] = 1
    A = np.nan_to_num((A - mu) / sd); A = np.clip(A, -6, 6)
    best = None
    for c in (0.0003, 0.001, 0.003):
        sw = None if not args.half_life else 0.5 ** ((t[fit_idx].max() - t[fit_idx]) / (args.half_life * 864e5))
        clf = LogisticRegression(C=c, max_iter=3000).fit(A[fit_idx], y[fit_idx], sample_weight=sw)
        p = clf.predict_proba(A[dev_idx])[:, 1]
        ll = log_loss(y[dev_idx], p)
        print('C', c, round(ll, 5), flush=True)
        if best is None or ll < best[0]:
            best = (ll, c, p)
    print('features', len(lin), 'C', best[1])
    report('dev', y[dev_idx], best[2])
    if args.save_dev:
        np.savez(args.save_dev, ids=z['ids'][dev_idx], p=best[2], y=y[dev_idx])
    if args.final:
        mu = np.nanmean(Xo[tr][:, lin], 0); sd = np.nanstd(Xo[tr][:, lin], 0); sd[sd == 0] = 1
        A = np.clip(np.nan_to_num((Xo[:, lin] - mu) / sd), -6, 6)
        pv = LogisticRegression(C=best[1], max_iter=3000).fit(A[tr], y[tr]).predict_proba(A[va])[:, 1]
        report('official_validation', y[va], pv)
        if args.save_pred:
            np.savez(args.save_pred, ids=z['ids'][va], p=pv, y=y[va])
    raise SystemExit(0)

xf, yf = train_set(fit_idx)
pd, its = [], []
for s in range(args.seeds):
    bst = lgb.train(params(42 + s), lgb.Dataset(xf, yf, categorical_feature=cat_idx, free_raw_data=False), num_boost_round=6000,
                    valid_sets=[lgb.Dataset(Xo[dev_idx], y[dev_idx], categorical_feature=cat_idx)],
                    callbacks=[lgb.early_stopping(args.es, verbose=False)])
    its.append(bst.best_iteration); pd.append(predict(bst, dev_idx, bst.best_iteration))
print('features', len(cols), 'best_iters', its)
report('dev', y[dev_idx], np.mean(pd, 0))
if args.save_dev:
    np.savez(args.save_dev, ids=z['ids'][dev_idx], p=np.mean(pd, 0), y=y[dev_idx], iters=np.array(its))
if args.importance:
    imp = bst.feature_importance('gain')
    for i in np.argsort(-imp)[:args.importance]:
        print(f'{imp[i]:10.0f} {cols[i]}')
if args.final:
    xa, ya = train_set(tr)
    pv = []
    for s, it in enumerate(its):
        full = lgb.train(params(42 + s), lgb.Dataset(xa, ya, categorical_feature=cat_idx), num_boost_round=int(it * len(tr) / len(fit_idx)))
        pv.append(predict(full, va))
    report('official_validation', y[va], np.mean(pv, 0))
    if args.save_pred:
        np.savez(args.save_pred, ids=z['ids'][va], p=np.mean(pv, 0), y=y[va])
if args.importance:
    pdev = np.mean(pd, 0)
    P_ = list(z['pnames']); k = P_.index('log_n')
    cold = (X0[dev_idx][:, [r * NP + k for r in range(10)]] == 0).sum(1)
    for c in range(11):
        msk = cold == c
        if msk.sum() > 200:
            print('cold', c, int(msk.sum()), round(float(((pdev[msk] > .5) == y[dev_idx][msk]).mean()), 4))
    conf = np.abs(pdev - 0.5)
    for q in (0.5, 0.25, 0.1):
        th = np.quantile(conf, 1 - q); msk = conf >= th
        print(f'top{int(q*100)}% conf acc', round(float(((pdev[msk] > .5) == y[dev_idx][msk]).mean()), 4))
