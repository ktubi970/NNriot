"""Training-time neutralisation of the crawler's selection artifact on the frozen comfort features.

Reproduces the configuration frozen on 24 Oct 2026 (runs/comfort_features_20261024/lock.json):
logistic regression C=0.001 on the difference columns, LightGBM "B" with swap augmentation and
symmetrised prediction, 0.4 LR + 0.6 LGB blend. Only the training matches (exclusion) or their
weights (inverse probability of selection) change. Dev = the last 12 000 train matches, never
modified; validation = the official 5 000 matches. The sealed test is never featurized.

    python neutralize.py FEAT_V4M_NPZ CRAWL_PROVENANCE_CSV OUT_DIR [--variants a,b] [--models lr,lgb]
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

FROZEN_FEATURES_SHA256 = "16190e876446a580b7bc4e40ce8bddeaf8abc76ca3a99a086f7b7951ce3e2b4f"
ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
CAT = {"keystone", "primary_style", "sub_style", "pref_status"}
DEV = 12000
LR_C = 0.001
BLEND_LR = 0.4
ES_ROUNDS = 300
LGB_PARAMS = {"objective": "binary", "learning_rate": 0.02, "num_leaves": 15, "min_data_in_leaf": 400,
              "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
              "verbose": -1, "seed": 42, "num_threads": 10, "max_bin": 63, "deterministic": True,
              "force_row_wise": True, "cat_smooth": 50, "max_cat_to_onehot": 8}
VARIANTS = ("baseline", "drop_seed", "drop_pre_entry_seed", "drop_pre_entry_anchor",
            "ipw_pre_entry_seed", "ipw_pre_entry_anchor", "ipw_anchor_and_crawled")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(X, P, tail_names, swapped):
    """evaluate.py of the comfort experiment with its frozen flags (no drop, raw columns kept)."""
    NP = len(P)
    blue = X[:, :5 * NP].reshape(-1, 5, NP)
    red = X[:, 5 * NP:10 * NP].reshape(-1, 5, NP)
    tail = X[:, 10 * NP:].copy()
    if swapped:
        blue, red = red, blue
        for i, n in enumerate(tail_names):
            if n.endswith("red_wr"):
                tail[:, i] = 1 - tail[:, i]
            elif n.endswith("_diffrb") or n == "elo_diff":
                tail[:, i] = -tail[:, i]
    keep = list(range(NP))
    num = [i for i in keep if P[i] not in CAT]
    out = [blue.reshape(len(X), -1), red.reshape(len(X), -1)]
    cols = [f"blue_{r}_{P[i]}" for r in ROLES for i in keep] + [f"red_{r}_{P[i]}" for r in ROLES for i in keep]
    out.append((red[:, :, num] - blue[:, :, num]).reshape(len(X), -1))
    cols += [f"diff_{r}_{P[i]}" for r in ROLES for i in num]
    with np.errstate(all="ignore"):
        out.append(np.nanmean(red[:, :, num], 1) - np.nanmean(blue[:, :, num], 1))
        cols += [f"teamdiff_{P[i]}" for i in num]
        out.append(np.nanmin(red[:, :, num], 1) - np.nanmin(blue[:, :, num], 1))
        cols += [f"teammindiff_{P[i]}" for i in num]
    k = P.index("pref_status")
    out.append(((red[:, :, k] == 2).sum(1) - (blue[:, :, k] == 2).sum(1))[:, None])
    cols.append("autofill_count_diff")
    out.append(tail)
    cols += tail_names
    out.append(np.full((len(X), 1), float(swapped)))
    cols.append("swapped_side")
    return np.hstack(out).astype(np.float32), cols


def load_provenance(path, ids):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = {r["match_id"]: r for r in csv.DictReader(fh)}
    missing = [i for i in ids if i not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} feature rows have no provenance, e.g. {missing[:3]}")

    def col(name):
        return np.array([int(rows[i][name]) for i in ids])

    pre_seed_a, pre_seed_b = col("pre_entry_seeds_a"), col("pre_entry_seeds_b")
    pre_anchor_a = pre_seed_a + col("pre_entry_anchors_a")
    pre_anchor_b = pre_seed_b + col("pre_entry_anchors_b")
    return {
        "has_ladder_seed": col("has_ladder_seed").astype(bool),
        "has_pre_entry_seed": col("has_pre_entry_seed").astype(bool),
        "has_pre_entry_anchor": col("has_pre_entry_anchor").astype(bool),
        "balanced": col("balanced").astype(bool),
        "crawled_imbalance": col("crawled_imbalance"),
        # +1: only team B (red) carries the artifact, -1: only team A (blue), 0: none or both.
        "pre_seed_side": (pre_seed_b > 0).astype(int) - (pre_seed_a > 0).astype(int),
        "pre_anchor_side": (pre_anchor_b > 0).astype(int) - (pre_anchor_a > 0).astype(int),
    }


def ipw(weights, idx, stratum, y):
    """Weight each stratum so that its artifact side wins exactly half of it.

    ``stratum`` is signed: its sign says which side carries the artifact (+ = team B, red),
    its value groups matches with the same imbalance; 0 leaves the match untouched.
    The stratum keeps its total weight.
    """
    info = {}
    for s in np.unique(stratum[idx]):
        if s == 0:
            continue
        rows = idx[stratum[idx] == s]
        won = (y[rows] == 1) if s > 0 else (y[rows] == 0)
        p = float(won.mean())
        weights[rows] = np.where(won, 0.5 / p, 0.5 / (1 - p))
        info[f"{int(s):+d}"] = {"n": int(len(rows)), "artifact_side_win_rate": round(p, 4)}
    return info


def training_set(variant, idx, prov, y):
    """Rows and weights used to fit, for a chronological index set (fit or full train)."""
    if variant == "baseline":
        return idx, np.ones(len(y)), {}
    if variant.startswith("drop_"):
        flag = {"drop_seed": "has_ladder_seed", "drop_pre_entry_seed": "has_pre_entry_seed",
                "drop_pre_entry_anchor": "has_pre_entry_anchor"}[variant]
        kept = idx[~prov[flag][idx]]
        return kept, np.ones(len(y)), {"dropped": int(len(idx) - len(kept))}
    weights = np.ones(len(y))
    if variant == "ipw_pre_entry_seed":
        return idx, weights, {"pre_entry_seed": ipw(weights, idx, prov["pre_seed_side"], y)}
    info = {"pre_entry_anchor": ipw(weights, idx, prov["pre_anchor_side"], y)}
    if variant == "ipw_anchor_and_crawled":
        crawled = np.clip(prov["crawled_imbalance"], -3, 3) * ~prov["has_pre_entry_anchor"]
        info["crawled"] = ipw(weights, idx, crawled, y)
    return idx, weights, info


def metrics(y, p):
    if len(y) < 50:
        return {"n": int(len(y))}
    acc = float(((p > 0.5) == y).mean())
    return {"n": int(len(y)), "acc": round(acc, 4), "auc": round(float(roc_auc_score(y, p)), 4),
            "logloss": round(float(log_loss(y, p, labels=[0, 1])), 5)}


def subsets(prov, idx):
    strict = ~prov["has_pre_entry_anchor"] & (prov["crawled_imbalance"] == 0)
    return {"all": np.ones(len(idx), bool), "balanced": prov["balanced"][idx], "strict_balanced": strict[idx],
            "imbalanced": ~prov["balanced"][idx], "pre_entry_anchor": prov["has_pre_entry_anchor"][idx]}


def tilt(prov, idx, y, p, key="pre_anchor_side"):
    """Predicted vs observed win rate of the artifact side where only one side carries it."""
    side = np.sign(prov[key][idx])
    m = side != 0
    if not m.any():
        return {}
    pa = np.where(side[m] == 1, p[m], 1 - p[m])
    ya = np.where(side[m] == 1, y[idx][m], 1 - y[idx][m])
    return {"n": int(m.sum()), "predicted": round(float(pa.mean()), 4), "observed": round(float(ya.mean()), 4)}


def block_bootstrap(y, p_new, p_ref, mask, reps=2000, block=100, seed=20261025):
    """Paired difference (new - reference) of accuracy and log-loss, moving blocks in time order."""
    rng = np.random.default_rng(seed)
    n = len(y)
    starts = np.arange(0, n - block + 1)
    n_blocks = int(np.ceil(n / block))
    eps = 1e-15
    acc_d = ((p_new > 0.5) == y).astype(float) - ((p_ref > 0.5) == y).astype(float)
    ll = lambda p: -(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1)))
    ll_d = ll(p_new) - ll(p_ref)
    out_acc, out_ll = [], []
    for _ in range(reps):
        rows = (rng.choice(starts, n_blocks)[:, None] + np.arange(block)).ravel()
        rows = rows[mask[rows]]
        out_acc.append(acc_d[rows].mean())
        out_ll.append(ll_d[rows].mean())
    point_acc, point_ll = float(acc_d[mask].mean()), float(ll_d[mask].mean())
    return {"n": int(mask.sum()),
            "delta_acc": round(point_acc, 4), "delta_acc_ci95": [round(float(q), 4) for q in np.quantile(out_acc, [.025, .975])],
            "delta_logloss": round(point_ll, 5), "delta_logloss_ci95": [round(float(q), 5) for q in np.quantile(out_ll, [.025, .975])]}


def fit_lr(A, rows, weights, y):
    clf = LogisticRegression(C=LR_C, max_iter=3000)
    return clf.fit(A[rows], y[rows], sample_weight=weights[rows])


def standardized(Xo, lin, rows):
    A = Xo[:, lin].astype(np.float64)
    mu = np.nanmean(A[rows], 0)
    sd = np.nanstd(A[rows], 0)
    sd[sd == 0] = 1
    return np.clip(np.nan_to_num((A - mu) / sd), -6, 6)


def run_lr(Xo, cols, y, fit, dev, tr, va, variant, prov):
    lin = [i for i, c in enumerate(cols) if not c.startswith(("blue_", "red_")) and c != "swapped_side"]
    rows, w, info_fit = training_set(variant, fit, prov, y)
    A = standardized(Xo, lin, rows)
    p_dev = fit_lr(A, rows, w, y).predict_proba(A[dev])[:, 1]
    rows, w, info_train = training_set(variant, tr, prov, y)
    A = standardized(Xo, lin, rows)
    p_val = fit_lr(A, rows, w, y).predict_proba(A[va])[:, 1]
    return p_dev, p_val, {"fit": info_fit, "train": info_train}


def run_lgb(Xo, Xs, cols, y, fit, dev, tr, va, variant, prov):
    cat_idx = [i for i, c in enumerate(cols) if any(c.endswith("_" + k) for k in CAT) and not c.startswith(("diff", "team"))]

    def dataset(rows, w):
        return lgb.Dataset(np.vstack([Xo[rows], Xs[rows]]), np.concatenate([y[rows], 1 - y[rows]]),
                           weight=np.concatenate([w[rows], w[rows]]), categorical_feature=cat_idx, free_raw_data=True)

    def predict(bst, rows, it=None):
        return 0.5 * (bst.predict(Xo[rows], num_iteration=it) + 1 - bst.predict(Xs[rows], num_iteration=it))

    rows, w, info_fit = training_set(variant, fit, prov, y)
    bst = lgb.train(LGB_PARAMS, dataset(rows, w), num_boost_round=6000,
                    valid_sets=[lgb.Dataset(Xo[dev], y[dev], categorical_feature=cat_idx)],
                    callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
    best, n_fit = bst.best_iteration, len(rows)
    p_dev = predict(bst, dev, best)
    rows, w, info_train = training_set(variant, tr, prov, y)
    full = lgb.train(LGB_PARAMS, dataset(rows, w), num_boost_round=int(best * len(rows) / n_fit))
    return p_dev, predict(full, va), {"fit": info_fit, "train": info_train, "best_iteration": best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("provenance")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--models", default="lr,lgb")
    args = ap.parse_args()
    if sha256_file(args.features) != FROZEN_FEATURES_SHA256:
        raise SystemExit("feature table is not the frozen feat_v4m.npz")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "inputs.json").write_text(json.dumps({
        "features": {"path": str(args.features), "sha256": FROZEN_FEATURES_SHA256},
        "provenance": {"path": str(args.provenance), "sha256": sha256_file(args.provenance)},
        "script_sha256": sha256_file(__file__)}, indent=2))

    z = np.load(args.features, allow_pickle=False)
    X0, y, split, t, ids = z["X"], z["y"], z["split"], z["t"], list(z["ids"])
    P, names = list(z["pnames"]), list(z["names"])
    tail_names = names[10 * len(P):]
    prov = load_provenance(args.provenance, ids)
    Xo, cols = build(X0, P, tail_names, False)
    tr = np.where(split == "train")[0]
    tr = tr[np.argsort(t[tr], kind="stable")]
    va = np.where(split == "validation")[0]
    va = va[np.argsort(t[va], kind="stable")]
    fit, dev = tr[:-DEV], tr[-DEV:]
    models = args.models.split(",")
    Xs = build(X0, P, tail_names, True)[0] if "lgb" in models else None

    results_path = args.out_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for variant in args.variants.split(","):
        entry = results.setdefault(variant, {})
        preds = {}
        for model in models:
            t0 = time.time()
            if model == "lr":
                p_dev, p_val, info = run_lr(Xo, cols, y, fit, dev, tr, va, variant, prov)
            else:
                p_dev, p_val, info = run_lgb(Xo, Xs, cols, y, fit, dev, tr, va, variant, prov)
            preds[model] = (p_dev, p_val)
            np.savez(args.out_dir / f"pred_{variant}_{model}.npz", dev_ids=np.array(ids)[dev], p_dev=p_dev,
                     val_ids=np.array(ids)[va], p_val=p_val)
            entry[model] = {"info": info, "seconds": round(time.time() - t0, 1)}
            print(variant, model, round(time.time() - t0, 1), "s", flush=True)
        for model in list(entry):
            path = args.out_dir / f"pred_{variant}_{model}.npz"
            if model not in preds and path.exists():
                saved = np.load(path)
                preds[model] = (saved["p_dev"], saved["p_val"])
        if {"lr", "lgb"} <= set(preds):
            preds["blend"] = tuple(BLEND_LR * preds["lr"][i] + (1 - BLEND_LR) * preds["lgb"][i] for i in (0, 1))
            entry.setdefault("blend", {})
        for model, (p_dev, p_val) in preds.items():
            scores = entry[model].setdefault("scores", {})
            for name, idx, p in (("dev", dev, p_dev), ("validation", va, p_val)):
                scores[name] = {s: metrics(y[idx][m], p[m]) for s, m in subsets(prov, idx).items()}
                scores[name]["artifact_tilt"] = tilt(prov, idx, y, p)
                scores[name]["crawled_tilt"] = tilt(prov, idx, y, p, "crawled_imbalance")
        results_path.write_text(json.dumps(results, indent=2))
        print(json.dumps({m: entry[m]["scores"] for m in entry if "scores" in entry[m]})[:1500], flush=True)

    if "baseline" in results:
        for variant in results:
            for model in ("lr", "lgb", "blend"):
                ref = args.out_dir / f"pred_baseline_{model}.npz"
                new = args.out_dir / f"pred_{variant}_{model}.npz"
                if variant == "baseline" or model not in results[variant]:
                    continue
                if model == "blend":
                    def blend(v):
                        a, b = np.load(args.out_dir / f"pred_{v}_lr.npz"), np.load(args.out_dir / f"pred_{v}_lgb.npz")
                        return {k: BLEND_LR * a[k] + (1 - BLEND_LR) * b[k] for k in ("p_dev", "p_val")}
                    if not all((args.out_dir / f"pred_{v}_{m}.npz").exists() for v in (variant, "baseline") for m in ("lr", "lgb")):
                        continue
                    pn, pr = blend(variant), blend("baseline")
                elif ref.exists() and new.exists():
                    pn, pr = np.load(new), np.load(ref)
                else:
                    continue
                cmp = {}
                for name, idx, key in (("dev", dev, "p_dev"), ("validation", va, "p_val")):
                    cmp[name] = {s: block_bootstrap(y[idx], pn[key], pr[key], m)
                                 for s, m in subsets(prov, idx).items() if s in ("all", "balanced", "strict_balanced")}
                results[variant][model]["vs_baseline"] = cmp
        results_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
