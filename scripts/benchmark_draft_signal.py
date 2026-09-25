"""Automated benchmark testing whether LoL draft alone has statistically significant predictive signal."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any
import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import config
from core.pre_match_dataset import load_records, split_records
from core.draft_features import batch_extract_draft_features
from core.training_metrics import evaluate_winner

logger = logging.getLogger("benchmark_draft_signal")


def run_draft_signal_benchmark(
    records: list[dict[str, Any]],
    min_samples: int = 100,
    n_splits: int = 3,
) -> dict[str, Any]:
    """Evaluate whether draft features achieve statistically significant lift over random chance."""
    if len(records) < min_samples:
        raise ValueError(f"Need at least {min_samples} records, got {len(records)}")

    splits, _ = split_records(records, strategy="chronological")
    x_train = batch_extract_draft_features([r["feature_json"] for r in splits["train"]], mode="tabular")
    x_val = batch_extract_draft_features([r["feature_json"] for r in splits["validation"]], mode="tabular")
    x_test = batch_extract_draft_features([r["feature_json"] for r in splits["test"]], mode="tabular")

    y_train = np.asarray([r["labels_json"]["winner"] for r in splits["train"]], dtype=int)
    y_val = np.asarray([r["labels_json"]["winner"] for r in splits["validation"]], dtype=int)
    y_test = np.asarray([r["labels_json"]["winner"] for r in splits["test"]], dtype=int)

    train_rate = float(y_train.mean())

    # Fit tabular gradient booster with early stopping on validation
    clf = HistGradientBoostingClassifier(
        categorical_features=list(range(10)),
        random_state=42,
        max_iter=100,
        early_stopping=True,
    )
    clf.fit(x_train, y_train)

    val_probs = clf.predict_proba(x_val)[:, 1]
    test_probs = clf.predict_proba(x_test)[:, 1]

    val_metrics = evaluate_winner(y_val, val_probs, train_rate)
    test_metrics = evaluate_winner(y_test, test_probs, train_rate)

    raw_test_auc = test_metrics["model"]["roc_auc"]
    test_auc = float(raw_test_auc) if raw_test_auc is not None else 0.5
    test_acc = float(test_metrics["model"]["accuracy"])
    base_acc = float(test_metrics["baseline"]["accuracy"])
    acc_lift = test_acc - base_acc

    # Standard error for AUC via Hanley & McNeil approximation
    n_pos = int(np.sum(y_test == 1))
    n_neg = int(np.sum(y_test == 0))
    if n_pos == 0 or n_neg == 0:
        auc_se = 1.0
        p_value = 1.0
    else:
        q1 = test_auc / (2.0 - test_auc) if (2.0 - test_auc) != 0 else 0.0
        q2 = 2.0 * test_auc**2 / (1.0 + test_auc) if (1.0 + test_auc) != 0 else 0.0
        denom = n_pos * n_neg
        auc_var = (
            test_auc * (1.0 - test_auc)
            + (n_pos - 1) * (q1 - test_auc**2)
            + (n_neg - 1) * (q2 - test_auc**2)
        ) / denom
        auc_se = float(np.sqrt(max(1e-8, auc_var)))
        z_score = (test_auc - 0.50) / auc_se if auc_se > 0 else 0.0
        p_value = float(1.0 - stats.norm.cdf(z_score))

    if test_auc >= 0.52 and p_value < 0.05:
        verdict = "SIGNAL_CONFIRMED"
    elif test_auc > 0.50 and p_value < 0.10:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "NO_SIGNAL"

    ci_lower = float(np.clip(test_auc - 1.96 * auc_se, 0.0, 1.0))
    ci_upper = float(np.clip(test_auc + 1.96 * auc_se, 0.0, 1.0))

    return {
        "verdict": verdict,
        "sample_counts": {k: len(v) for k, v in splits.items()},
        "test_auc": test_auc,
        "auc_standard_error": auc_se,
        "auc_ci_95": [ci_lower, ci_upper],
        "p_value_vs_null": p_value,
        "test_accuracy": test_acc,
        "baseline_accuracy": base_acc,
        "accuracy_lift": acc_lift,
        "val_log_loss": val_metrics["model"]["log_loss"],
        "test_log_loss": test_metrics["model"]["log_loss"],
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark draft signal on pre-match matches.")
    parser.add_argument("--db", type=str, required=True, help="Path to SQLite database.")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples to evaluate.")
    parser.add_argument("--output", type=str, default="runs/draft_signal_report.json", help="Report output path.")
    args = parser.parse_args()

    records = load_records(args.db, limit=args.samples)
    report = run_draft_signal_benchmark(records)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Benchmark finished: verdict={report['verdict']} test_auc={report['test_auc']:.4f} p={report['p_value_vs_null']:.4e}")


if __name__ == "__main__":
    main()
