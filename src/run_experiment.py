
"""
Orchestrates the complete representation learning pipeline for a market.

Runs in sequence:
1. Baseline (handcrafted features)
2. PCA representations
3. Autoencoder (AE) representations
4. Variational Autoencoder (VAE) representations

All methods use the same dataset splits and evaluation harness for fair comparison.
Results are appended to results/tables/metrics.csv.

Usage:
    python src/run_experiment.py --market OMXS
    python src/run_experiment.py --market SPX
    python src/run_experiment.py --market OMXS --skip-ae --skip-vae  # baseline + PCA only
    python src/run_experiment.py --market OMXS > results/omxs_experiment.json
    python src/run_experiment.py --market SPX  > results/spx_experiment.json

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from config import (
    DATA_DIR,
    RANDOM_SEED,
    ensure_directories,
    set_global_seed,
)
from data_processing import ProcessingConfig
from datasets import SplitConfig, create_dataset, log_dataset_stats
from baseline_features import BaselineConfig, run_baseline
from repr_pca import PCAConfig, run_pca
from repr_ae import AEConfig, run_ae
from repr_vae import VAETrainConfig, run_vae
from train_classifier import ClassifierConfig


def run_full_pipeline(
    market_name: str,
    data_path: Optional[Path] = None,
    processing_cfg: ProcessingConfig = ProcessingConfig(),
    split_cfg: SplitConfig = SplitConfig(),
    baseline_cfg: BaselineConfig = BaselineConfig(),
    pca_cfg: PCAConfig = PCAConfig(),
    ae_cfg: AEConfig = AEConfig(),
    vae_cfg: VAETrainConfig = VAETrainConfig(),
    clf_cfg: ClassifierConfig = ClassifierConfig(),
    skip_baseline: bool = False,
    skip_pca: bool = False,
    skip_ae: bool = False,
    skip_vae: bool = False,
    save_metrics: bool = True,
    save_models: bool = True,
) -> dict:
    """
    Run complete pipeline: baseline → PCA → AE → VAE for one market.

    Args:
        market_name: Market identifier (e.g., "OMXS", "SPX")
        data_path: Path to raw data file. If None, inferred from market_name.
        skip_*: Flags to skip specific methods
        save_metrics: Append results to metrics.csv
        save_models: Save trained model weights

    Returns:
        Dict with results from all methods and dataset stats.
    """
    ensure_directories()
    set_global_seed(RANDOM_SEED)

    # Infer data path if not provided
    if data_path is None:
        data_path = DATA_DIR / f"{market_name.lower()}.txt"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Could not find data file for market '{market_name}'. "
                f"Expected: {data_path}"
            )

    print(f"\n{'='*70}")
    print(f"Running full pipeline for market: {market_name}")
    print(f"Data file: {data_path}")
    print(f"{'='*70}\n")

    # Load dataset once (all methods use same splits)
    print("Loading and preprocessing data...")
    dataset = create_dataset(
        data_path=data_path,
        processing_cfg=processing_cfg,
        split_cfg=split_cfg,
        label_threshold=None,  # compute threshold on train only
    )

    # Log dataset statistics
    log_dataset_stats(dataset["stats"], dataset_name=market_name)

    results = {
        "market": market_name,
        "data_path": str(data_path),
        "dataset_stats": dataset["stats"],
        "methods": {},
    }

    # 1. Baseline
    if not skip_baseline:
        print(f"\n{'─'*70}")
        print("1. Running baseline (handcrafted features)...")
        print(f"{'─'*70}")
        baseline_result = run_baseline(
            data_path=data_path,
            market_name=market_name,
            processing_cfg=processing_cfg,
            split_cfg=split_cfg,
            baseline_cfg=baseline_cfg,
            clf_cfg=clf_cfg,
            save_metrics=save_metrics,
        )
        results["methods"]["baseline"] = baseline_result
        print("✓ Baseline complete")

    # 2. PCA
    if not skip_pca:
        print(f"\n{'─'*70}")
        print("2. Running PCA representations...")
        print(f"{'─'*70}")
        pca_result = run_pca(
            data_path=data_path,
            market_name=market_name,
            processing_cfg=processing_cfg,
            split_cfg=split_cfg,
            pca_cfg=pca_cfg,
            clf_cfg=clf_cfg,
            save_metrics=save_metrics,
        )
        results["methods"]["pca"] = pca_result
        print("✓ PCA complete")

    # 3. Autoencoder
    if not skip_ae:
        print(f"\n{'─'*70}")
        print("3. Running Autoencoder (AE) representations...")
        print(f"{'─'*70}")
        print("(This may take a while - training multiple seeds per embedding dim)")
        ae_result = run_ae(
            data_path=data_path,
            market_name=market_name,
            processing_cfg=processing_cfg,
            split_cfg=split_cfg,
            ae_cfg=ae_cfg,
            clf_cfg=clf_cfg,
            save_metrics=save_metrics,
            save_models=save_models,
        )
        results["methods"]["ae"] = ae_result
        print("✓ AE complete")

    # 4. Variational Autoencoder
    if not skip_vae:
        print(f"\n{'─'*70}")
        print("4. Running Variational Autoencoder (VAE) representations...")
        print(f"{'─'*70}")
        print("(This may take a while - training multiple seeds per embedding dim)")
        vae_result = run_vae(
            data_path=data_path,
            market_name=market_name,
            processing_cfg=processing_cfg,
            split_cfg=split_cfg,
            vae_cfg=vae_cfg,
            clf_cfg=clf_cfg,
            save_metrics=save_metrics,
        )
        results["methods"]["vae"] = vae_result
        print("✓ VAE complete")

    print(f"\n{'='*70}")
    print(f"Pipeline complete for {market_name}")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run complete representation learning pipeline (baseline + PCA + AE + VAE)."
    )
    parser.add_argument(
        "--market",
        type=str,
        required=True,
        choices=["OMXS", "SPX"],
        help="Market to run experiments on",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to raw data file (default: data/{market}.txt)",
    )

    # Skip flags
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline method")
    parser.add_argument("--skip-pca", action="store_true", help="Skip PCA method")
    parser.add_argument("--skip-ae", action="store_true", help="Skip Autoencoder method")
    parser.add_argument("--skip-vae", action="store_true", help="Skip VAE method")

    # Output flags
    parser.add_argument("--no-save", action="store_true", help="Do not append to metrics.csv")
    parser.add_argument("--no-save-models", action="store_true", help="Do not save model weights")

    # Config overrides (optional - most use defaults)
    parser.add_argument("--window-len", type=int, default=60)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED)

    args = parser.parse_args()

    # Build configs
    processing_cfg = ProcessingConfig(
        window_len=args.window_len,
        vol_window=args.vol_window,
    )
    split_cfg = SplitConfig()  # 70/15/15 default
    baseline_cfg = BaselineConfig()
    pca_cfg = PCAConfig()
    ae_cfg = AEConfig()
    vae_cfg = VAETrainConfig()
    clf_cfg = ClassifierConfig(random_state=args.random_state)

    data_path = Path(args.input) if args.input else None

    # Run pipeline
    results = run_full_pipeline(
        market_name=args.market,
        data_path=data_path,
        processing_cfg=processing_cfg,
        split_cfg=split_cfg,
        baseline_cfg=baseline_cfg,
        pca_cfg=pca_cfg,
        ae_cfg=ae_cfg,
        vae_cfg=vae_cfg,
        clf_cfg=clf_cfg,
        skip_baseline=args.skip_baseline,
        skip_pca=args.skip_pca,
        skip_ae=args.skip_ae,
        skip_vae=args.skip_vae,
        save_metrics=not args.no_save,
        save_models=not args.no_save_models,
    )

    # Print summary JSON
    print("\nSummary:")
    print(json.dumps(results, indent=2, default=str))