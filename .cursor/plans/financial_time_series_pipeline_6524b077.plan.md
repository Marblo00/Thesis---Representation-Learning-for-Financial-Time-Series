---
name: Financial Time Series Pipeline
overview: Build a reproducible end-to-end pipeline for representation learning on financial time series, implementing baseline features, PCA, AE, and VAE representations for volatility regime classification on OMXS and SPX datasets.
todos:
  - id: setup_structure
    content: "Create repository structure: config.py, all src/ modules, results/ subdirectories (tables/, models/, plots/)"
    status: pending
  - id: datasets
    content: "Implement datasets.py: use data_processing.py functions to load/preprocess, add time-based train/val/test splits (70/15/15), create split dataset interface"
    status: pending
    dependencies:
      - setup_structure
  - id: classifier_utils
    content: "Implement train_classifier.py: logistic regression training, hyperparameter tuning (C), evaluation metrics (Accuracy, F1, Balanced Accuracy)"
    status: pending
    dependencies:
      - datasets
  - id: baseline
    content: "Implement baseline_features.py: handcrafted features (mean, std, abs mean, momentum), train/evaluate classifier, save metrics to CSV"
    status: pending
    dependencies:
      - classifier_utils
  - id: pca
    content: "Implement repr_pca.py: flatten windows, standardize (train stats only), fit PCA on train, transform all splits, evaluate with classifier, save metrics"
    status: pending
    dependencies:
      - baseline
  - id: ae
    content: "Implement repr_ae.py: MLP autoencoder (120→64→d encoder, d→64→120 decoder), train on train only, early stopping, extract embeddings, evaluate with multiple seeds, save models and metrics"
    status: pending
    dependencies:
      - pca
  - id: vae
    content: "Implement repr_vae.py: VAE with mu/logvar encoder, use mu as embedding, train with ELBO loss (recon + β*KL), early stopping, multiple seeds, save models and metrics"
    status: pending
    dependencies:
      - ae
  - id: orchestration
    content: "Implement run_experiment.py: single command runs full pipeline (baseline→PCA→AE→VAE), deterministic, logs sizes/balance, saves all artifacts"
    status: pending
    dependencies:
      - vae
  - id: multi_market
    content: Extend run_experiment.py to support SPX dataset, run identical pipeline, compare results across markets
    status: pending
    dependencies:
      - orchestration
---

# Financial Time Series Representation Learning Pipeline

## Overview

Implement a complete, reproducible pipeline for learning representations from financial time series and evaluating them on volatility regime classification. The pipeline processes daily returns, creates rolling windows, learns representations via PCA/AE/VAE, and evaluates using logistic regression.

## Repository Structure

```
data/
  omxs.txt (existing)
  spx.txt (existing)

src/
  config.py              # Configuration parameters
  data_processing.py     # EXISTING: Load, preprocess, windowing (load_raw_market_file, add_returns_and_volatility, build_windows)
  datasets.py            # Time-based splits and dataset interface (uses data_processing.py)
  baseline_features.py   # Handcrafted features + classifier
  repr_pca.py            # PCA representation + classifier
  repr_ae.py             # Autoencoder training + embeddings
  repr_vae.py            # VAE training + embeddings
  train_classifier.py    # Logistic regression utilities
  run_experiment.py      # Main orchestration script

results/
  tables/
    metrics.csv          # Evaluation metrics
  models/                # Saved AE/VAE weights
  plots/                 # Optional visualizations
```

## Implementation Order

### Phase 1: Foundation (Steps 0-3)

**0. Repository structure**

- Create all directory structure
- Set up `config.py` with hyperparameters (window_length=60, vol_window=20, embedding_dims=[8,16,32], train/val/test splits)

**1. Data loading and preprocessing** ✅ **ALREADY IMPLEMENTED** in [data_processing.py](src/data_processing.py)

- `load_raw_market_file()`: Parses CSV, extracts DATE and CLOSE, sorts chronologically
- `add_returns_and_volatility()`: Computes log returns, absolute returns, rolling volatility
- `build_windows()`: Creates rolling windows (L=60), aligns labels to window end dates, thresholds volatility
- All critical time alignment rules are already implemented

**2. Time-based splitting and dataset interface** ([datasets.py](src/datasets.py))

- Import and use functions from `data_processing.py`:
  - `process_market()` or direct calls to `load_raw_market_file()`, `add_returns_and_volatility()`, `build_windows()`
- Add time-based splitting function:
  - Takes X, y, end_dates from `build_windows()`
  - Train: first 70% of windows (chronological)
  - Validation: next 15%
  - Test: last 15%
  - No random splits (prevents distribution leakage)
- Create dataset interface function that returns:
  - `(X_train, y_train)`, `(X_val, y_val)`, `(X_test, y_test)`
  - Logs sizes and class balance

**4. Baseline features** ([baseline_features.py](src/baseline_features.py))

- Per window (60 returns):
  - Mean of returns
  - Std of returns
  - Mean of abs returns
  - Last-1 cumulative return (momentum)
  - Optional: skew, kurtosis
- Train logistic regression on train features
- Tune C parameter on validation set
- Evaluate on test: Accuracy, F1, Balanced Accuracy
- Save metrics to `results/tables/metrics.csv`

### Phase 2: Representation Learning (Steps 4-6)

**5. PCA representation** ([repr_pca.py](src/repr_pca.py))

- Flatten each `(60,2)` window to `(120,)`
- Standardize using train-set mean/std only
- Fit PCA on train windows only
- Transform train/val/test to embeddings of size d ∈ {8,16,32}
- Run logistic regression on embeddings (same evaluation as baseline)
- Append metrics to `results/tables/metrics.csv`

**6. Autoencoder** ([repr_ae.py](src/repr_ae.py))

- Architecture: MLP AE (start simple)
  - Encoder: 120 → 64 → d
  - Decoder: d → 64 → 120
  - Loss: MSE reconstruction
- Training:
  - Train only on train windows (unsupervised)
  - Early stopping on validation reconstruction loss
  - Save best model to `results/models/`
- Embeddings:
  - Extract by running encoder on train/val/test
  - Use encoder output (not decoder input)
- Evaluation:
  - Logistic regression on embeddings
  - Run with multiple seeds (≥3), report mean/std
  - Append metrics to CSV

**7. Variational Autoencoder** ([repr_vae.py](src/repr_vae.py))

- Encoder outputs `(mu, logvar)`
- Sample z from `N(mu, exp(logvar))`
- For classification: use `mu` as embedding (stable)
- Loss: `recon_loss (MSE) + β * KL` (start β=1.0, try 0.1 if needed)
- Training:
  - Early stopping on validation ELBO
  - Train only on train windows
- Embeddings and evaluation: same as AE
- Multiple seeds, report mean/std

### Phase 3: Orchestration and Multi-Market (Steps 7-8)

**8. Main experiment runner** ([run_experiment.py](src/run_experiment.py))

- Single command runs entire pipeline:

  1. Use `data_processing.py` to load and preprocess data
  2. Use `datasets.py` to create windows and splits
  3. Baseline features → metrics
  4. PCA → metrics
  5. AE → metrics
  6. VAE → metrics

- Deterministic (set random seeds)
- Log train/val/test sizes, class balance
- Save all artifacts to `results/`

**9. Second market** (extend [run_experiment.py](src/run_experiment.py))

- Add SPX dataset support
- Run identical pipeline
- Compare within-market performance
- Optional: cross-market robustness (train on one, test on other)

## Critical Implementation Rules

1. **No data leakage**:

   - **Note**: `build_windows()` in `data_processing.py` computes threshold on all windows. For proper train-only threshold, either:
     - Modify `build_windows()` to accept pre-computed threshold, OR
     - Compute threshold on train windows only in `datasets.py` before calling `build_windows()` with `label_threshold` parameter
   - Standardization statistics from train only
   - PCA fit on train only
   - AE/VAE trained on train only

2. **Time alignment**:

   - Window ending at t uses returns `[t-59 ... t]`
   - Label at t uses volatility from `[t-19 ... t]`
   - No future information in labels

3. **Determinism**:

   - Set all random seeds (numpy, random, torch if used)
   - Re-running produces identical baseline/PCA results
   - AE/VAE: report mean/std across seeds

4. **Evaluation consistency**:

   - Same logistic regression setup for all methods
   - Same train/val/test splits
   - Same metrics (Accuracy, F1, Balanced Accuracy)

## Milestone Definition

**Milestone 1 complete when**:

- Single command produces metrics table with baseline + PCA for OMXS
- Train/val/test sizes logged
- Class balance logged
- Artifacts saved in `results/`
- Re-running gives identical results for baseline/PCA
- **Only then proceed to AE/VAE**

## Technical Stack

- Python 3.x
- pandas: data loading and manipulation
- numpy: numerical operations
- scikit-learn: PCA, logistic regression, metrics
- PyTorch (or TensorFlow): AE/VAE implementation
- Optional: matplotlib for plots

## Configuration Parameters

Store in [config.py](src/config.py):

- `WINDOW_LENGTH = 60`
- `VOL_WINDOW = 20`
- `EMBEDDING_DIMS = [8, 16, 32]`
- `TRAIN_SPLIT = 0.70`
- `VAL_SPLIT = 0.15`
- `TEST_SPLIT = 0.15`
- `VOL_THRESHOLD_METHOD = 'median'` (or 'quantile')
- `RANDOM_SEED = 42`
- `AE_SEEDS = [42, 123, 456]` (for multiple runs)
- `BETA_VAE = 1.0` (KL weight)