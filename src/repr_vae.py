# src/repr_vae.py
"""
Variational Autoencoder (VAE) representation learning for financial time series windows,
followed by downstream volatility regime classification using logistic regression.

Pipeline:
1) Load train/val/test windows via datasets.create_dataset()
2) Flatten windows: (N, 60, 2) -> (N, 120)
3) Standardize using train-only statistics (no leakage)
4) Train VAE (unsupervised) on train only, early-stop on val ELBO
5) Extract embeddings as mu (latent mean) for train/val/test
6) Evaluate embeddings with train_classifier.train_and_evaluate_logreg()
7) Append results to results/tables/metrics.csv (one row per seed & embedding_dim, plus optional mean/std)

Run (repo root):
  python src/repr_vae.py --input data/omxs.txt --market OMXS
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import METRICS_PATH, MODELS_DIR, ensure_directories, set_global_seed
from data_processing import ProcessingConfig
from datasets import SplitConfig, create_dataset
from train_classifier import ClassifierConfig, train_and_evaluate_logreg

PathLike = Union[str, Path]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class VAETrainConfig:
    embedding_dims: Tuple[int, ...] = (8, 16, 32)
    hidden_dim: int = 64
    beta: float = 1.0
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    patience: int = 10
    tune_metric: str = "f1"  # for downstream logreg tuning on val
    seeds: Tuple[int, ...] = (42, 123, 456)
    save_mean_std_row: bool = True


class MLPVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # z = mu + std * eps
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Reconstruction loss (MSE). Mean over batch.
    recon = F.mse_loss(x_hat, x, reduction="mean")
    # KL divergence between N(mu, sigma^2) and N(0,1). Mean over batch.
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    total = recon + beta * kl
    return total, recon, kl


def append_metrics_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if path.exists():
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)


def _flatten_windows(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1).astype(np.float32)


def _make_loaders(
    X_train: np.ndarray,
    X_val: np.ndarray,
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    # Deterministic shuffling for train loader
    g = torch.Generator()
    g.manual_seed(seed)

    train_ds = TensorDataset(torch.from_numpy(X_train))
    val_ds = TensorDataset(torch.from_numpy(X_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_vae_one_seed(
    X_train: np.ndarray,
    X_val: np.ndarray,
    input_dim: int,
    latent_dim: int,
    cfg: VAETrainConfig,
    seed: int,
    device: torch.device,
) -> Tuple[MLPVAE, Dict[str, float]]:
    set_global_seed(seed)

    model = MLPVAE(input_dim=input_dim, hidden_dim=cfg.hidden_dim, latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train_loader, val_loader = _make_loaders(X_train, X_val, cfg.batch_size, seed=seed)

    best_val = float("inf")
    best_state = None
    patience_left = cfg.patience

    last_logged = {"train_total": np.nan, "val_total": np.nan, "val_recon": np.nan, "val_kl": np.nan}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []

        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad(set_to_none=True)
            x_hat, mu, logvar = model(xb)
            total, recon, kl = vae_loss(xb, x_hat, mu, logvar, beta=cfg.beta)
            total.backward()
            opt.step()
            train_losses.append(float(total.detach().cpu().item()))

        # Validation
        model.eval()
        val_totals, val_recons, val_kls = [], [], []
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                x_hat, mu, logvar = model(xb)
                total, recon, kl = vae_loss(xb, x_hat, mu, logvar, beta=cfg.beta)
                val_totals.append(float(total.cpu().item()))
                val_recons.append(float(recon.cpu().item()))
                val_kls.append(float(kl.cpu().item()))

        train_total = float(np.mean(train_losses)) if train_losses else float("inf")
        val_total = float(np.mean(val_totals)) if val_totals else float("inf")
        val_recon = float(np.mean(val_recons)) if val_recons else float("inf")
        val_kl = float(np.mean(val_kls)) if val_kls else float("inf")

        last_logged = {
            "train_total": train_total,
            "val_total": val_total,
            "val_recon": val_recon,
            "val_kl": val_kl,
        }

        # Early stopping
        if val_total < best_val - 1e-8:
            best_val = val_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_val_total": best_val, **last_logged}


@torch.no_grad()
def extract_mu_embeddings(model: MLPVAE, X: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    ds = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    mus = []
    for (xb,) in loader:
        xb = xb.to(device)
        mu, _ = model.encode(xb)
        mus.append(mu.detach().cpu().numpy())
    return np.concatenate(mus, axis=0)


def run_vae(
    data_path: PathLike,
    market_name: str,
    processing_cfg: ProcessingConfig = ProcessingConfig(),
    split_cfg: SplitConfig = SplitConfig(),
    vae_cfg: VAETrainConfig = VAETrainConfig(),
    clf_cfg: ClassifierConfig = ClassifierConfig(),
    save_metrics: bool = True,
) -> Dict[str, object]:
    ensure_directories()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    ds = create_dataset(
        data_path=Path(data_path),
        processing_cfg=processing_cfg,
        split_cfg=split_cfg,
        label_threshold=None,  # train-only threshold
    )

    X_train_w, y_train = ds["X_train"], ds["y_train"]
    X_val_w, y_val = ds["X_val"], ds["y_val"]
    X_test_w, y_test = ds["X_test"], ds["y_test"]

    # Flatten windows
    X_train = _flatten_windows(X_train_w)
    X_val = _flatten_windows(X_val_w)
    X_test = _flatten_windows(X_test_w)

    # Standardize using train only (no leakage)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    input_dim = X_train_s.shape[1]
    all_rows = []
    summary: Dict[str, object] = {
        "market": market_name,
        "method": "vae",
        "device": str(device),
        "threshold": float(ds["threshold"]),
        "runs": [],
    }

    for d in vae_cfg.embedding_dims:
        per_seed_test = []
        per_seed_val = []

        for seed in vae_cfg.seeds:
            # Train
            model, train_log = train_vae_one_seed(
                X_train=X_train_s,
                X_val=X_val_s,
                input_dim=input_dim,
                latent_dim=int(d),
                cfg=vae_cfg,
                seed=int(seed),
                device=device,
            )

            # Save model
            model_path = MODELS_DIR / f"vae_{market_name.lower()}_d{d}_seed{seed}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dim": vae_cfg.hidden_dim,
                    "latent_dim": int(d),
                    "beta": float(vae_cfg.beta),
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_,
                },
                model_path,
            )

            # Embeddings (use mu)
            Z_train = extract_mu_embeddings(model, X_train_s, device=device)
            Z_val = extract_mu_embeddings(model, X_val_s, device=device)
            Z_test = extract_mu_embeddings(model, X_test_s, device=device)

            # Downstream classifier
            clf_result = train_and_evaluate_logreg(
                X_train=Z_train,
                y_train=y_train,
                X_val=Z_val,
                y_val=y_val,
                X_test=Z_test,
                y_test=y_test,
                cfg=clf_cfg,
                tune_metric=vae_cfg.tune_metric,
                standardize=True,  # standardize embeddings (train-only inside util)
            )

            run_info = {
                "embedding_dim": int(d),
                "seed": int(seed),
                "best_C": float(clf_result["best_C"]),
                "val_metrics_best": clf_result["val_metrics_best"],
                "test_metrics": clf_result["test_metrics"],
                "train_log": train_log,
                "model_path": str(model_path),
            }
            summary["runs"].append(run_info)

            per_seed_val.append(clf_result["val_metrics_best"])
            per_seed_test.append(clf_result["test_metrics"])

            if save_metrics:
                row = {
                    "market": market_name,
                    "method": "vae",
                    "embedding_dim": int(d),
                    "n_features": int(d),
                    "seed": int(seed),
                    "beta": float(vae_cfg.beta),
                    "threshold": float(ds["threshold"]),
                    "best_C": float(clf_result["best_C"]),
                    "val_accuracy": float(clf_result["val_metrics_best"]["accuracy"]),
                    "val_f1": float(clf_result["val_metrics_best"]["f1"]),
                    "val_balanced_accuracy": float(clf_result["val_metrics_best"]["balanced_accuracy"]),
                    "test_accuracy": float(clf_result["test_metrics"]["accuracy"]),
                    "test_f1": float(clf_result["test_metrics"]["f1"]),
                    "test_balanced_accuracy": float(clf_result["test_metrics"]["balanced_accuracy"]),
                    "best_val_total": float(train_log["best_val_total"]),
                    "val_recon": float(train_log["val_recon"]),
                    "val_kl": float(train_log["val_kl"]),
                    "model_path": str(model_path),
                }
                append_metrics_row(METRICS_PATH, row)
                all_rows.append(row)

        # Optional mean/std row across seeds 
        if vae_cfg.save_mean_std_row and save_metrics and per_seed_test:
            def mean_std(metric_name: str) -> Tuple[float, float]:
                vals = np.array([m[metric_name] for m in per_seed_test], dtype=float)
                return float(vals.mean()), float(vals.std(ddof=0))

            acc_m, acc_s = mean_std("accuracy")
            f1_m, f1_s = mean_std("f1")
            bal_m, bal_s = mean_std("balanced_accuracy")

            row_mean = {
                "market": market_name,
                "method": "vae_meanstd",
                "embedding_dim": int(d),
                "n_features": int(d),
                "seed": "mean±std",
                "beta": float(vae_cfg.beta),
                "threshold": float(ds["threshold"]),
                "best_C": "",
                "val_accuracy": "",
                "val_f1": "",
                "val_balanced_accuracy": "",
                "test_accuracy": f"{acc_m:.6f}±{acc_s:.6f}",
                "test_f1": f"{f1_m:.6f}±{f1_s:.6f}",
                "test_balanced_accuracy": f"{bal_m:.6f}±{bal_s:.6f}",
                "best_val_total": "",
                "val_recon": "",
                "val_kl": "",
                "model_path": "",
            }
            append_metrics_row(METRICS_PATH, row_mean)
            all_rows.append(row_mean)

    summary["metrics_rows_written"] = len(all_rows)
    return summary


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Train VAE representations and evaluate with logistic regression.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw data file, e.g. data/omxs.txt")
    parser.add_argument("--market", type=str, required=True, help="Market name tag, e.g. OMXS or SPX")

    parser.add_argument("--dims", type=int, nargs="*", default=[8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 123, 456])
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--tune-metric", type=str, default="f1", choices=["accuracy", "f1", "balanced_accuracy"])
    parser.add_argument("--no-save", action="store_true", help="Do not append results to metrics.csv")
    parser.add_argument("--no-meanstd", action="store_true", help="Do not write mean±std summary rows")

    args = parser.parse_args()

    vae_cfg = VAETrainConfig(
        embedding_dims=tuple(args.dims),
        hidden_dim=args.hidden_dim,
        beta=args.beta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        tune_metric=args.tune_metric,
        seeds=tuple(args.seeds),
        save_mean_std_row=not args.no_meanstd,
    )

    summary = run_vae(
        data_path=args.input,
        market_name=args.market,
        vae_cfg=vae_cfg,
        save_metrics=not args.no_save,
    )

    print(json.dumps(summary, indent=2))