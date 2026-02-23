# src/datasets.py
"""
Time-based dataset splitting for financial time series windows.

Handles train/val/test splits (70/15/15) with proper threshold computation
on train set only to avoid data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from data_processing import (
    ProcessingConfig,
    add_returns_and_volatility,
    build_windows,
    load_raw_market_file,
)


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for time-based dataset splits."""
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15

    def __post_init__(self):
        total = self.train_split + self.val_split + self.test_split
        if not np.isclose(total, 1.0):
            raise ValueError(f"Splits must sum to 1.0, got {total}")


def split_windows_chronological(
    X: np.ndarray,
    y: np.ndarray,
    end_dates: np.ndarray,
    cfg: SplitConfig = SplitConfig(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split windows chronologically into train/val/test sets.
    
    Args:
        X: Window features of shape (n_windows, window_len, n_features)
        y: Binary labels of shape (n_windows,)
        end_dates: End dates for each window
        cfg: Split configuration
        
    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    n_total = len(X)
    
    # Compute split indices
    n_train = int(n_total * cfg.train_split)
    n_val = int(n_total * cfg.val_split)
    # n_test = n_total - n_train - n_val (remaining)
    
    # Split chronologically (no shuffling)
    X_train = X[:n_train]
    y_train = y[:n_train]
    
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_threshold_on_train(
    df: pd.DataFrame,
    end_dates_train: np.ndarray,
    cfg: ProcessingConfig,
) -> float:
    """
    Compute volatility threshold using only train set windows.
    
    Args:
        df: DataFrame with 'rolling_vol' column
        end_dates_train: End dates of train windows
        cfg: Processing config with label_quantile
        
    Returns:
        Threshold value for binary labeling
    """
    # Get continuous volatility values at train window end dates
    df_dates = pd.to_datetime(df["date"].values)
    train_indices = []
    
    for end_date in end_dates_train:
        # Find row index matching this end date
        mask = df_dates == pd.to_datetime(end_date)
        if mask.any():
            idx = np.where(mask)[0][0]
            train_indices.append(idx)
    
    if not train_indices:
        raise ValueError("No matching dates found for train windows")
    
    train_vol = df.iloc[train_indices]["rolling_vol"].values
    train_vol = train_vol[np.isfinite(train_vol)]
    
    if len(train_vol) == 0:
        raise ValueError("No valid volatility values in train set")
    
    threshold = float(np.quantile(train_vol, cfg.label_quantile))
    return threshold


def create_dataset(
    data_path: Path,
    processing_cfg: ProcessingConfig = ProcessingConfig(),
    split_cfg: SplitConfig = SplitConfig(),
    label_threshold: float | None = None,
) -> dict:
    """
    Create train/val/test datasets from raw market data file.
    
    This function:
    1. Loads and preprocesses data
    2. Builds windows
    3. Computes threshold on train set only (if not provided)
    4. Applies threshold to create binary labels
    5. Splits chronologically
    
    Args:
        data_path: Path to raw data file (e.g., data/omxs.txt)
        processing_cfg: Configuration for windowing and preprocessing
        split_cfg: Configuration for train/val/test splits
        label_threshold: Optional pre-computed threshold. If None, computed on train only.
        
    Returns:
        Dictionary with:
        - 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'
        - 'threshold': threshold used for labeling
        - 'stats': dictionary with sizes and class balance info
    """
    # Load and preprocess
    df = load_raw_market_file(data_path)
    df = add_returns_and_volatility(df, vol_window=processing_cfg.vol_window)
    
    # Build windows (without thresholding if we want train-only threshold)
    # We'll build windows first, then recompute labels with train-only threshold
    if label_threshold is None:
        # Build windows to get structure, but we'll recompute labels
        # Temporarily use a dummy threshold, then recompute
        X_temp, _, end_dates, _ = build_windows(
            df, cfg=processing_cfg, label_threshold=0.0
        )
        
        # Split to get train indices
        n_total = len(X_temp)
        n_train = int(n_total * split_cfg.train_split)
        end_dates_train = end_dates[:n_train]
        
        # Compute threshold on train only
        threshold = compute_threshold_on_train(df, end_dates_train, processing_cfg)
        
        # Rebuild windows with correct threshold
        X, y, end_dates, threshold_used = build_windows(
            df, cfg=processing_cfg, label_threshold=threshold
        )
    else:
        # Use provided threshold
        X, y, end_dates, threshold_used = build_windows(
            df, cfg=processing_cfg, label_threshold=label_threshold
        )
        threshold = label_threshold
    
    # Split chronologically
    X_train, y_train, X_val, y_val, X_test, y_test = split_windows_chronological(
        X, y, end_dates, cfg=split_cfg
    )
    
    # Compute statistics
    stats = {
        "n_total": len(X),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "train_class_balance": {
            "class_0": int(np.sum(y_train == 0)),
            "class_1": int(np.sum(y_train == 1)),
            "class_0_pct": float(np.mean(y_train == 0)),
            "class_1_pct": float(np.mean(y_train == 1)),
        },
        "val_class_balance": {
            "class_0": int(np.sum(y_val == 0)),
            "class_1": int(np.sum(y_val == 1)),
            "class_0_pct": float(np.mean(y_val == 0)),
            "class_1_pct": float(np.mean(y_val == 1)),
        },
        "test_class_balance": {
            "class_0": int(np.sum(y_test == 0)),
            "class_1": int(np.sum(y_test == 1)),
            "class_0_pct": float(np.mean(y_test == 0)),
            "class_1_pct": float(np.mean(y_test == 1)),
        },
        "threshold": float(threshold_used),
    }
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "threshold": threshold_used,
        "stats": stats,
    }


def log_dataset_stats(stats: dict, dataset_name: str = "") -> None:
    """
    Log dataset statistics to console.
    
    Args:
        stats: Statistics dictionary from create_dataset()
        dataset_name: Optional name prefix for logging
    """
    prefix = f"[{dataset_name}] " if dataset_name else ""
    
    print(f"{prefix}Dataset Statistics:")
    print(f"  Total windows: {stats['n_total']}")
    print(f"  Train: {stats['n_train']} ({stats['n_train']/stats['n_total']*100:.1f}%)")
    print(f"  Validation: {stats['n_val']} ({stats['n_val']/stats['n_total']*100:.1f}%)")
    print(f"  Test: {stats['n_test']} ({stats['n_test']/stats['n_total']*100:.1f}%)")
    print(f"  Threshold: {stats['threshold']:.6f}")
    print(f"\n  Train class balance:")
    print(f"    Class 0 (low vol): {stats['train_class_balance']['class_0']} ({stats['train_class_balance']['class_0_pct']*100:.1f}%)")
    print(f"    Class 1 (high vol): {stats['train_class_balance']['class_1']} ({stats['train_class_balance']['class_1_pct']*100:.1f}%)")
    print(f"  Val class balance:")
    print(f"    Class 0: {stats['val_class_balance']['class_0']} ({stats['val_class_balance']['class_0_pct']*100:.1f}%)")
    print(f"    Class 1: {stats['val_class_balance']['class_1']} ({stats['val_class_balance']['class_1_pct']*100:.1f}%)")
    print(f"  Test class balance:")
    print(f"    Class 0: {stats['test_class_balance']['class_0']} ({stats['test_class_balance']['class_0_pct']*100:.1f}%)")
    print(f"    Class 1: {stats['test_class_balance']['class_1']} ({stats['test_class_balance']['class_1_pct']*100:.1f}%)")


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    REPO_ROOT = Path(__file__).resolve().parents[1]
    data_path = REPO_ROOT / "data" / "omxs.txt"
    
    dataset = create_dataset(data_path)
    log_dataset_stats(dataset["stats"], dataset_name="OMXS")
    
    print(f"\nShapes:")
    print(f"  X_train: {dataset['X_train'].shape}")
    print(f"  X_val: {dataset['X_val'].shape}")
    print(f"  X_test: {dataset['X_test'].shape}")