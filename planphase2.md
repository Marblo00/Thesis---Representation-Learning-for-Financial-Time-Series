## Next Phase: Analysis & Thesis Work

After the OMXS pipeline (baseline, PCA, AE, VAE) is implemented and verified, the next steps are:

1. Run full pipeline on SPX  
   - `python src/run_experiment.py --market SPX`  
   - Confirm `results/tables/metrics.csv` contains rows for both `OMXS` and `SPX` and all methods (`baseline_handcrafted`, `pca`, `ae_mlp`, `vae`).

2. Aggregate metrics for tables  
   - Load `results/tables/metrics.csv` in a notebook or analysis script.  
   - For AE and VAE: group by `(market, method, embedding_dim)` and aggregate over seeds (mean and std of test accuracy, F1, balanced accuracy).  
   - For PCA: aggregate over `(market, method, embedding_dim)` if multiple dims.  
   - For baseline: one row per market.  
   - Produce summary tables:
     - Per market: baseline vs best PCA vs best AE vs best VAE.
     - For AE/VAE: performance as a function of embedding dimension.

3. Core plots for the paper  
   - **Per-market method comparison**: bar plots of test metrics (accuracy, F1, balanced accuracy) for `{baseline, PCA, AE, VAE}`.  
   - **Performance vs latent dimensionality**:
     - For each of PCA, AE, VAE: line plots of test metrics vs `embedding_dim` (8, 16, 32).
     - Add error bars for AE/VAE using std across seeds.

4. Diagnostics (optional but useful)  
   - Confusion matrices on test for the best model of each method/market.  
   - Probability histograms for predicted high-vol class to visualize separation and calibration.

5. Latent space inspection (qualitative)  
   - Pick best AE and best VAE (e.g. `embedding_dim=16` or `32`).  
   - Extract embeddings (AE encoder output, VAE μ) for train/val/test.  
   - Run PCA/t-SNE/UMAP on embeddings; plot with points colored by volatility regime (low/high), and optionally by time/crisis periods.  
   - Goal: visually check whether regimes separate better in learned spaces than in raw windows/PCA.

6. Robustness checks (if time)  
   - Change volatility threshold (e.g. different quantile on train) and rerun at least baseline + PCA; check if the relative ranking of methods is stable.  
   - Optionally test a different window length (e.g. 40 days) for one method (PCA) to see sensitivity.

7. Cross-market comparison  
   - Compare best configurations across OMXS and SPX in a single table.  
   - Optional transfer experiment:
     - Train representation (PCA/AE/VAE) on OMXS, train classifier on OMXS, test on SPX (and/or reverse) using same embeddings.
     - This probes cross-market robustness of representations.

8. Documentation & reproducibility  
   - In `README`, add a short “Reproducing results” section:
     - Create venv, `pip install -r requirements.txt`.
     - Run `run_experiment.py` for both markets.
     - Point to the analysis notebook/script that turns `metrics.csv` into tables and plots used in the thesis.  
   - Keep `requirements.txt` in sync with the environment used for final results.

These steps move the project from “pipeline implemented” to “results analyzed and ready to write up” without further core code changes.