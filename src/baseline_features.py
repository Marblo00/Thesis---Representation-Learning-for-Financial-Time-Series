# src/baseline_features.py
"""
Hand-crafted feature baseline for volatility regime classification.

Pipeline:
1) Load train/val/test windows via datasets.create_dataset()
2) Extract simple statistical features per window
3) Train + tune Logistic Regression via train_classifier.train_and_evaluate_logreg()
4) Return metrics (and optionally write to results/tables/metrics.csv)

Notes:
- No shuffling (time-based splits already handled in datasets.py)
- All standardization and C-tuning handled in train_classifier.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config import METRICS_PATH, ensure_directories
from data_processing import ProcessingConfig
from datasets import SplitConfig, create_dataset
from train_classifier import ClassifierConfig, train_and_evaluate_logreg


PathLike = Union[str, Path]


@dataclass(frozen=True)
class BaselineConfig:
    """
    Configuration for baseline feature extraction + classifier training.
    """
    include_higher_moments: bool = True  # skew/kurtosis 
    tune_metric: str = "f1"
    standardize: bool = True


def _skew(x: np.ndarray) -> float:
    """
    Population skewness (no scipy dependency).
    Returns 0 if variance is ~0.
    """
    x = x.astype(float)
    m = x.mean()
    s = x.std()
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis_excess(x: np.ndarray) -> float:
    """
    Population excess kurtosis (kurtosis - 3), no scipy dependency.
    Returns 0 if variance is ~0.
    """
    x = x.astype(float)
    m = x.mean()
    s = x.std()
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def extract_baseline_features(X: np.ndarray, include_higher_moments: bool = True) -> np.ndarray:
    """
    Convert windows (N, 60, 2) into handcrafted features (N, d_feat).

    Assumes channel 0 = log_return, channel 1 = abs_log_return.

    Features (minimum):
      - mean_r
      - std_r
      - mean_abs_r
      - std_abs_r
      - sum_r (cumulative log return)
      - last_r
      - max_abs_r
      - min_r
      - max_r

    Optional:
      - skew_r
      - kurtosis_r (excess)
    """
    if X.ndim != 3 or X.shape[2] < 2:
        raise ValueError(f"Expected X shape (N, L, 2+). Got {X.shape}")

    r = X[:, :, 0].astype(float)
    ar = X[:, :, 1].astype(float)

    mean_r = r.mean(axis=1)
    std_r = r.std(axis=1)
    mean_ar = ar.mean(axis=1)
    std_ar = ar.std(axis=1)

    sum_r = r.sum(axis=1)
    last_r = r[:, -1]
    max_ar = ar.max(axis=1)

    min_r = r.min(axis=1)
    max_r = r.max(axis=1)

    feats = [
        mean_r,
        std_r,
        mean_ar,
        std_ar,
        sum_r,
        last_r,
        max_ar,
        min_r,
        max_r,
    ]

    if include_higher_moments:
        skew_r = np.array([_skew(row) for row in r], dtype=float)
        kurt_r = np.array([_kurtosis_excess(row) for row in r], dtype=float)
        feats.extend([skew_r, kurt_r])

    X_feat = np.column_stack(feats)
    if not np.isfinite(X_feat).all():
        raise ValueError("Non-finite values in extracted features. Check input windows.")
    return X_feat


def append_metrics_row(path: Path, row: Dict[str, object]) -> None:
    """
    Append one experiment row to metrics.csv, creating the file with header if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if path.exists():
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)


def run_baseline(
    data_path: PathLike,
    market_name: str,
    processing_cfg: ProcessingConfig = ProcessingConfig(),
    split_cfg: SplitConfig = SplitConfig(),
    baseline_cfg: BaselineConfig = BaselineConfig(),
    clf_cfg: ClassifierConfig = ClassifierConfig(),
    save_metrics: bool = True,
) -> Dict[str, object]:
    """
    Run handcrafted-feature baseline on one market file.
    """
    ensure_directories()

    ds = create_dataset(
        data_path=Path(data_path),
        processing_cfg=processing_cfg,
        split_cfg=split_cfg,
        label_threshold=None,  # compute threshold on train only
    )

    X_train, y_train = ds["X_train"], ds["y_train"]
    X_val, y_val = ds["X_val"], ds["y_val"]
    X_test, y_test = ds["X_test"], ds["y_test"]

    # Feature extraction
    F_train = extract_baseline_features(X_train, include_higher_moments=baseline_cfg.include_higher_moments)
    F_val = extract_baseline_features(X_val, include_higher_moments=baseline_cfg.include_higher_moments)
    F_test = extract_baseline_features(X_test, include_higher_moments=baseline_cfg.include_higher_moments)

    # Train/evaluate
    result = train_and_evaluate_logreg(
        X_train=F_train,
        y_train=y_train,
        X_val=F_val,
        y_val=y_val,
        X_test=F_test,
        y_test=y_test,
        cfg=clf_cfg,
        tune_metric=baseline_cfg.tune_metric,
        standardize=baseline_cfg.standardize,
    )

    out = {
        "market": market_name,
        "method": "baseline_handcrafted",
        "n_features": int(F_train.shape[1]),
        "threshold": float(ds["threshold"]),
        "best_C": float(result["best_C"]),
        "val_metrics_best": result["val_metrics_best"],
        "test_metrics": result["test_metrics"],
        "dataset_stats": ds["stats"],
    }

    if save_metrics:
        row = {
            "market": market_name,
            "method": "baseline_handcrafted",
            "embedding_dim": "",
            "n_features": int(F_train.shape[1]),
            "threshold": float(ds["threshold"]),
            "best_C": float(result["best_C"]),
            "val_accuracy": float(result["val_metrics_best"]["accuracy"]),
            "val_f1": float(result["val_metrics_best"]["f1"]),
            "val_balanced_accuracy": float(result["val_metrics_best"]["balanced_accuracy"]),
            "test_accuracy": float(result["test_metrics"]["accuracy"]),
            "test_f1": float(result["test_metrics"]["f1"]),
            "test_balanced_accuracy": float(result["test_metrics"]["balanced_accuracy"]),
        }
        append_metrics_row(METRICS_PATH, row)

    return out


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Handcrafted baseline features + logistic regression.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw data file, e.g. data/omxs.txt")
    parser.add_argument("--market", type=str, required=True, help="Market name tag, e.g. OMXS or SPX")
    parser.add_argument("--no-save", action="store_true", help="Do not append results to metrics.csv")

    parser.add_argument("--window-len", type=int, default=60)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--train-split", type=float, default=0.70)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)

    parser.add_argument("--no-higher-moments", action="store_true", help="Disable skew/kurtosis features.")
    parser.add_argument("--tune-metric", type=str, default="f1", choices=["accuracy", "f1", "balanced_accuracy"])
    parser.add_argument("--no-standardize", action="store_true", help="Disable StandardScaler.")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    processing_cfg = ProcessingConfig(window_len=args.window_len, vol_window=args.vol_window)
    split_cfg = SplitConfig(train_split=args.train_split, val_split=args.val_split, test_split=args.test_split)
    baseline_cfg = BaselineConfig(
        include_higher_moments=not args.no_higher_moments,
        tune_metric=args.tune_metric,
        standardize=not args.no_standardize,
    )
    clf_cfg = ClassifierConfig(max_iter=args.max_iter, random_state=args.random_state)

    res = run_baseline(
        data_path=args.input,
        market_name=args.market,
        processing_cfg=processing_cfg,
        split_cfg=split_cfg,
        baseline_cfg=baseline_cfg,
        clf_cfg=clf_cfg,
        save_metrics=not args.no_save,
    )

    print(json.dumps(res, indent=2))