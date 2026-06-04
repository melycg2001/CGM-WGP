"""
gblup_flowering.py

True GBLUP prediction of flowering DAP using the genomic relationship matrix.
Model: DAP = planting_mean + genomic_breeding_value + error
       u ~ N(0, G * sigma_g²)

Uses eigenvalue decomposition (EVD) of G-matrix for efficient computation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def find_rscript() -> Optional[str]:
    """Locate Rscript binary."""
    p = shutil.which("Rscript")
    if p:
        return p
    for c in ["/opt/homebrew/bin/Rscript", "/usr/local/bin/Rscript", "/usr/bin/Rscript"]:
        if Path(c).is_file():
            return c
    return None


def load_evd(evd_rda_path: str, rscript_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load EVD.rda via a subprocess R call, return (V, d, genotype_ids).

    V: eigenvectors matrix (n x n)
    d: eigenvalues vector (n,)
    genotype_ids: list of genotype IDs from rownames
    """
    if rscript_path is None:
        rscript_path = find_rscript()
    if rscript_path is None:
        raise RuntimeError("Rscript not found on PATH or common locations")

    evd_rda_path = str(Path(evd_rda_path).resolve())
    if not Path(evd_rda_path).exists():
        raise FileNotFoundError(f"EVD file not found: {evd_rda_path}")

    # Create temp files for export
    tmp_dir = tempfile.mkdtemp(prefix="gblup_evd_")
    vectors_csv = os.path.join(tmp_dir, "vectors.csv")
    values_csv = os.path.join(tmp_dir, "values.csv")

    r_code = f"""
load("{evd_rda_path}")
write.csv(EVD$vectors, "{vectors_csv}", row.names=TRUE)
ids <- rownames(EVD$vectors)
if (is.null(ids)) ids <- paste0("G", seq_len(nrow(EVD$vectors)))
write.csv(data.frame(id=ids, eigenvalue=EVD$values), "{values_csv}", row.names=FALSE)
"""
    try:
        result = subprocess.run(
            [rscript_path, "-e", r_code],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"R export failed (exit {result.returncode}): {result.stderr}")

        # Read vectors (first column is row names = genotype IDs)
        v_df = pd.read_csv(vectors_csv, index_col=0)
        V = v_df.values.astype(np.float64)
        genotype_ids = [str(x).strip().strip('"') for x in v_df.index]

        # Read eigenvalues
        d_df = pd.read_csv(values_csv)
        d = d_df["eigenvalue"].values.astype(np.float64)

        return V, d, genotype_ids
    finally:
        # Clean up temp files
        for f in [vectors_csv, values_csv]:
            if os.path.exists(f):
                os.unlink(f)
        if os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)


def estimate_lambda_reml(d: np.ndarray, z: np.ndarray) -> float:
    """
    REML estimate of lambda = sigma_e² / sigma_g² from spectral decomposition.

    d: eigenvalues of G-matrix
    z: rotated phenotypes (V' @ y_adjusted)

    Maximizes: L(λ) = -0.5 * Σ[log(d_i + λ) + z_i²/(d_i + λ)]
    """
    # Only use components where eigenvalue > 0 (skip zero eigenvalues)
    mask = d > 1e-10
    d_pos = d[mask]
    z_pos = z[mask]

    if len(d_pos) == 0:
        return 1.0  # fallback

    def neg_log_lik(log_lambda):
        lam = np.exp(log_lambda)
        denom = d_pos + lam
        return 0.5 * np.sum(np.log(denom) + z_pos ** 2 / denom)

    result = minimize_scalar(neg_log_lik, bounds=(-14, 14), method="bounded")
    return np.exp(result.x)


def gblup_predict_dap(
    evd_path: str,
    pheno_df: pd.DataFrame,
    train_plantings: List[str],
    predict_plantings: List[str],
    genotype_col: str = "id",
    planting_col: str = "Planting",
    ft_col: str = "ft",
    censored_col: str = "censored",
    rscript_path: Optional[str] = None,
    pred_planting_means: Optional[Dict[str, float]] = None,
    test_pheno_df: Optional[pd.DataFrame] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Predict flowering DAP using GBLUP directly from the G-matrix.

    When test_pheno_df is provided (balanced_training_subset CV scheme), the
    observed DAP lookup for validation is built from test_pheno_df instead of
    from pheno_df at the predict plantings. pheno_df still drives REML fitting
    and breeding value estimation; test_pheno_df is read-only for metric rows.

    Returns:
        (prediction_rows, info_dict)
        prediction_rows: list of {id, planting, observed_dap, predicted_dap, error, breeding_value}
        info_dict: {lambda_est, n_gmatrix, n_matched, train_planting_means, pred_planting_means}
    """
    # Load EVD
    V, d, gmatrix_ids = load_evd(evd_path, rscript_path)
    n_gmatrix = len(gmatrix_ids)

    # Build genotype ID index (case-insensitive matching)
    gmatrix_id_upper = [gid.upper() for gid in gmatrix_ids]
    gmatrix_idx = {gid_upper: i for i, gid_upper in enumerate(gmatrix_id_upper)}

    # Filter phenotype data: uncensored, valid ft, in training plantings
    df = pheno_df.copy()
    if censored_col in df.columns:
        df = df[df[censored_col] == 0]
    df = df.dropna(subset=[ft_col])
    df[ft_col] = df[ft_col].astype(float)
    df[genotype_col] = df[genotype_col].astype(str).str.strip()

    train_df = df[df[planting_col].isin(train_plantings)]

    # Step 1: Compute planting means from training data
    train_planting_means = train_df.groupby(planting_col)[ft_col].mean().to_dict()

    # Step 2: Adjusted phenotypes (DAP - planting mean)
    train_df = train_df.copy()
    train_df["planting_mean"] = train_df[planting_col].map(train_planting_means)
    train_df["y_adj"] = train_df[ft_col] - train_df["planting_mean"]

    # Step 3: Per-genotype mean adjusted phenotype
    genotype_means = train_df.groupby(genotype_col)["y_adj"].mean()

    # Step 4: Map to G-matrix order
    y_vec = np.zeros(n_gmatrix, dtype=np.float64)
    observed_mask = np.zeros(n_gmatrix, dtype=bool)
    n_matched = 0
    for gid, y_adj_mean in genotype_means.items():
        idx = gmatrix_idx.get(str(gid).strip().upper())
        if idx is not None:
            y_vec[idx] = y_adj_mean
            observed_mask[idx] = True
            n_matched += 1

    if n_matched == 0:
        print("Warning: no genotypes matched between phenotype data and G-matrix", file=sys.stderr)
        return [], {"lambda_est": None, "n_gmatrix": n_gmatrix, "n_matched": 0,
                     "train_planting_means": train_planting_means, "pred_planting_means": {}}

    # Step 5: REML to estimate lambda
    z = V.T @ y_vec  # rotated phenotypes
    lambda_est = estimate_lambda_reml(d, z)

    # Step 6: GBLUP breeding values
    shrinkage = d / (d + lambda_est)
    u_hat = V @ (shrinkage * z)  # û = V @ diag(d/(d+λ)) @ V' @ y

    # Step 6b: Per-genotype prediction error variance (PEV)
    # PEV_i = sigma_g^2 * sum_k(v_ik^2 * lambda/(d_k + lambda))
    # sigma_g^2 estimated from REML spectral decomposition
    mask_pos = d > 1e-10
    d_pos = d[mask_pos]
    z_pos = z[mask_pos]
    n_eff = int(np.sum(mask_pos))
    sigma_g_sq = float(np.sum(z_pos ** 2 * d_pos / (d_pos + lambda_est) ** 2) / max(n_eff, 1))
    sigma_e_sq = lambda_est * sigma_g_sq

    pev_weights = lambda_est / (d + lambda_est)  # lambda/(d_k + lambda) per component
    relative_pev = np.sum(V ** 2 * pev_weights[np.newaxis, :], axis=1)  # shape (n_gmatrix,)
    pev_array = sigma_g_sq * relative_pev
    pred_var_array = sigma_e_sq + pev_array  # total prediction variance per genotype

    # Step 7: Prediction planting means
    overall_train_mean = train_df[ft_col].mean() if len(train_df) > 0 else 0.0
    if pred_planting_means is not None:
        # Use externally provided means (e.g. from mechanistic model predictions)
        for pp in predict_plantings:
            if pp not in pred_planting_means:
                pred_planting_means[pp] = overall_train_mean
                print(f"Warning: no mechanistic mean for {pp}, using overall training mean {overall_train_mean:.1f}", file=sys.stderr)
    else:
        # Fallback: compute per-env means from TRAINING data only. Must NOT
        # use the raw `df` here — it may contain rows for the predict env
        # (e.g. leave-one-env-out CV where the held-out env's rows are still
        # in pheno_df, but excluded from train_plantings). Using observed
        # DAPs from the predict env would be a data leak.
        pred_df = train_df[train_df[planting_col].isin(predict_plantings)]
        pred_planting_means = pred_df.groupby(planting_col)[ft_col].mean().to_dict()
        for pp in predict_plantings:
            if pp not in pred_planting_means:
                pred_planting_means[pp] = overall_train_mean
                print(f"Warning: no training DAP for {pp}, using overall training mean {overall_train_mean:.1f}", file=sys.stderr)

    # Step 8: Generate predictions
    # Build observed DAP lookup for evaluation. When test_pheno_df is provided,
    # use held-out cells for the obs lookup (pheno_df only has training rows
    # in balanced_training_subset mode); otherwise use pheno_df at predict envs.
    obs_lookup = {}
    if test_pheno_df is not None:
        _tdf = test_pheno_df.copy()
        if censored_col in _tdf.columns:
            _tdf = _tdf[_tdf[censored_col] == 0]
        _tdf = _tdf.dropna(subset=[ft_col])
        _tdf[genotype_col] = _tdf[genotype_col].astype(str).str.strip()
        _tdf = _tdf[_tdf[planting_col].isin(predict_plantings)]
        for _, row in _tdf.iterrows():
            key = (str(row[genotype_col]).strip(), row[planting_col])
            obs_lookup[key] = int(round(float(row[ft_col])))
    else:
        pred_df = df[df[planting_col].isin(predict_plantings)]
        for _, row in pred_df.iterrows():
            key = (str(row[genotype_col]).strip(), row[planting_col])
            obs_lookup[key] = int(row[ft_col])

    prediction_rows = []
    # Predict for ALL genotypes in G-matrix (including unseen via genomic relationships)
    for i, gid in enumerate(gmatrix_ids):
        breeding_val = float(u_hat[i])

        for pred_e in predict_plantings:
            pred_mean = pred_planting_means[pred_e]
            predicted_dap = round(pred_mean + breeding_val)

            observed_dap = obs_lookup.get((gid, pred_e))
            # Also try uppercase matching
            if observed_dap is None:
                observed_dap = obs_lookup.get((gid.upper(), pred_e))

            error = (predicted_dap - observed_dap) if observed_dap is not None else None

            prediction_rows.append({
                "id": gid,
                "planting": pred_e,
                "observed_dap": observed_dap if observed_dap is not None else "NA",
                "predicted_dap": predicted_dap,
                "error": f"{error:.1f}" if error is not None else "NA",
                "breeding_value": f"{breeding_val:.4f}",
                "pred_var": f"{pred_var_array[i]:.4f}",
            })

    info = {
        "lambda_est": lambda_est,
        "n_gmatrix": n_gmatrix,
        "n_matched": n_matched,
        "train_planting_means": train_planting_means,
        "pred_planting_means": pred_planting_means,
        "sigma_g_sq": sigma_g_sq,
        "sigma_e_sq": sigma_e_sq,
        "pev_array": pev_array,
        "pred_var_array": pred_var_array,
        "gmatrix_ids": gmatrix_ids,
    }
    return prediction_rows, info


def gblup_proxy_mech(
    evd_path: str,
    mech_predictions: Dict[str, float],
    rscript_path: Optional[str] = None,
) -> Dict[str, float]:
    """GBLUP-predict Smoothed Mechanistic DAP for genotypes not in mech_predictions.

    Uses genomic relationships to estimate what the mechanistic model would predict
    for unseen genotypes. Takes fitted genotypes' Smoothed Mechanistic predicted DAPs
    as a "phenotype" and runs GBLUP to predict for all G-matrix genotypes.

    Returns {genotype_id: estimated_mech_DAP} for ALL G-matrix genotypes.
    Fitted genotypes get their original mech_predictions values back.
    """
    V, d, gmatrix_ids = load_evd(evd_path, rscript_path)
    n = len(gmatrix_ids)
    gmatrix_idx = {gid.upper(): i for i, gid in enumerate(gmatrix_ids)}

    # Build y-vector: mech predictions for fitted genotypes, 0 for unseen
    y_vec = np.zeros(n)
    observed_mask = np.zeros(n, dtype=bool)
    for gid, dap in mech_predictions.items():
        idx = gmatrix_idx.get(str(gid).strip().upper())
        if idx is not None:
            y_vec[idx] = dap
            observed_mask[idx] = True

    n_obs = int(observed_mask.sum())
    if n_obs < 2:
        return {}

    # Center around mean of observed
    obs_mean = y_vec[observed_mask].mean()
    y_centered = np.where(observed_mask, y_vec - obs_mean, 0.0)

    # GBLUP via spectral decomposition
    z = V.T @ y_centered
    lambda_est = estimate_lambda_reml(d, z)
    u_hat = V @ ((d / (d + lambda_est)) * z)

    # Return predictions for ALL genotypes
    result = {}
    for i, gid in enumerate(gmatrix_ids):
        if observed_mask[i]:
            # Fitted genotypes: return original mech prediction
            result[gid] = mech_predictions.get(gid, round(obs_mean + float(u_hat[i])))
        else:
            # Unseen genotypes: GBLUP-predicted mech DAP
            result[gid] = round(obs_mean + float(u_hat[i]))

    n_unseen = n - n_obs
    print(f"  G-matrix proxy mech: {n_obs} fitted -> {n_unseen} unseen genotypes predicted "
          f"(lambda={lambda_est:.2f}, mean={obs_mean:.1f})")
    return result


def _loo_gblup_spectral(
    V: np.ndarray, d: np.ndarray,
    y_alpha: np.ndarray, y_beta: np.ndarray,
    obs_mask: np.ndarray, gmatrix_ids: List[str],
    y_B: Optional[np.ndarray] = None,
    y_Pc: Optional[np.ndarray] = None,
) -> Dict[str, Tuple]:
    """LOO-GBLUP: for each fitted genotype, predict using the other N-1.

    For each observed genotype i, removes it from training, re-estimates lambda
    via REML on N-1 genotypes, and predicts i from the remaining genotypes'
    genomic relationships. This provides genuinely independent targets for the
    iterative penalty loop.

    Returns {genotype_id: (alpha_pred, beta_pred, [B_pred, Pc_pred,]
                           alpha_pev, beta_pev, [B_pev, Pc_pev])}.
    """
    obs_indices = np.where(obs_mask)[0]
    n_obs = len(obs_indices)
    has_photo = y_B is not None and y_Pc is not None
    loo_preds = {}

    for loo_idx in obs_indices:
        loo_gid = gmatrix_ids[loo_idx]

        # LOO mask: remove this genotype
        loo_mask = obs_mask.copy()
        loo_mask[loo_idx] = False

        # LOO GBLUP for alpha
        mean_a = y_alpha[loo_mask].mean()
        y_a_c = np.where(loo_mask, y_alpha - mean_a, 0.0)
        z_a = V.T @ y_a_c
        lam_a = estimate_lambda_reml(d, z_a)
        u_a = V @ ((d / (d + lam_a)) * z_a)

        # LOO GBLUP for beta
        mean_b = y_beta[loo_mask].mean()
        y_b_c = np.where(loo_mask, y_beta - mean_b, 0.0)
        z_b = V.T @ y_b_c
        lam_b = estimate_lambda_reml(d, z_b)
        u_b = V @ ((d / (d + lam_b)) * z_b)

        pred_a = mean_a + float(u_a[loo_idx])
        pred_b = mean_b + float(u_b[loo_idx])

        # PEV for held-out genotype
        shrink_a = d / (d + lam_a)
        shrink_b = d / (d + lam_b)
        var_a = np.var(y_alpha[loo_mask])
        var_b = np.var(y_beta[loo_mask])
        sg2_a = var_a / (1.0 + lam_a) if lam_a > 0 else var_a
        sg2_b = var_b / (1.0 + lam_b) if lam_b > 0 else var_b
        rel_a = float(np.sum(V[loo_idx, :] ** 2 * shrink_a))
        rel_b = float(np.sum(V[loo_idx, :] ** 2 * shrink_b))
        pev_a = max(sg2_a * (1.0 - rel_a), 1e-6)
        pev_b = max(sg2_b * (1.0 - rel_b), 1e-6)

        if has_photo:
            # LOO GBLUP for B
            mean_B_ = y_B[loo_mask].mean()
            y_B_c = np.where(loo_mask, y_B - mean_B_, 0.0)
            z_B_ = V.T @ y_B_c
            lam_B_ = estimate_lambda_reml(d, z_B_)
            u_B_ = V @ ((d / (d + lam_B_)) * z_B_)
            pred_B_ = mean_B_ + float(u_B_[loo_idx])

            # LOO GBLUP for Pc
            mean_Pc_ = y_Pc[loo_mask].mean()
            y_Pc_c = np.where(loo_mask, y_Pc - mean_Pc_, 0.0)
            z_Pc_ = V.T @ y_Pc_c
            lam_Pc_ = estimate_lambda_reml(d, z_Pc_)
            u_Pc_ = V @ ((d / (d + lam_Pc_)) * z_Pc_)
            pred_Pc_ = mean_Pc_ + float(u_Pc_[loo_idx])

            # PEV for B and Pc
            shrink_B_ = d / (d + lam_B_)
            shrink_Pc_ = d / (d + lam_Pc_)
            var_B_ = np.var(y_B[loo_mask])
            var_Pc_ = np.var(y_Pc[loo_mask])
            sg2_B_ = var_B_ / (1.0 + lam_B_) if lam_B_ > 0 else var_B_
            sg2_Pc_ = var_Pc_ / (1.0 + lam_Pc_) if lam_Pc_ > 0 else var_Pc_
            rel_B_ = float(np.sum(V[loo_idx, :] ** 2 * shrink_B_))
            rel_Pc_ = float(np.sum(V[loo_idx, :] ** 2 * shrink_Pc_))
            pev_B_ = max(sg2_B_ * (1.0 - rel_B_), 1e-6)
            pev_Pc_ = max(sg2_Pc_ * (1.0 - rel_Pc_), 1e-6)

            loo_preds[loo_gid] = (pred_a, pred_b, pred_B_, pred_Pc_,
                                  pev_a, pev_b, pev_B_, pev_Pc_)
        else:
            loo_preds[loo_gid] = (pred_a, pred_b, pev_a, pev_b)

    return loo_preds


def gmatrix_neighbor_targets(
    evd_path: str,
    fitted_params: Dict[str, Tuple],
    rscript_path: Optional[str] = None,
    positive_only: bool = True,
) -> Dict[str, Tuple]:
    """G-matrix weighted neighbor targets for iterative penalty.

    For each genotype, computes the kinship-weighted average of other
    genotypes' fitted parameters.  This replaces LOO-GBLUP which
    over-shrinks toward the grand mean and destroys cluster structure.

    The penalty ``Σ_j G⁺(i,j) × (α_i − α_j)²`` is mathematically
    equivalent to a quadratic centered at the G-weighted average with
    per-genotype strength proportional to total positive kinship.

    Args:
        evd_path: Path to EVD.rda file.
        fitted_params: ``{genotype_id: (alpha, beta)}`` or
            ``{genotype_id: (alpha, beta, topt)}`` from current round.
        rscript_path: Optional path to Rscript binary.
        positive_only: If True (default), only use positive G entries
            (``max(G_ij, 0)``).  Avoids extrapolation from negative kinship.

    Returns:
        When fitted_params has 2-tuples:
            ``{gid: (alpha_target, beta_target, w_norm, w_norm)}``
        When fitted_params has 3-tuples:
            ``{gid: (alpha_target, beta_target, topt_target, w_norm, w_norm)}``
        ``w_norm`` is the normalised kinship weight (mean = 1.0) used for
        per-genotype lambda scaling.
    """
    V, d, gmatrix_ids = load_evd(evd_path, rscript_path)
    n = len(gmatrix_ids)

    # Reconstruct G-matrix from EVD: G = V diag(d) V'
    G = (V * d[None, :]) @ V.T

    # Detect whether fitted_params includes topt (3-tuples)
    has_topt = any(len(v) >= 3 for v in fitted_params.values())

    # Map fitted genotypes to G-matrix indices
    gmatrix_idx = {gid.upper(): i for i, gid in enumerate(gmatrix_ids)}
    fitted_mask = np.zeros(n, dtype=bool)
    alpha_vals = np.zeros(n)
    beta_vals = np.zeros(n)
    topt_vals = np.zeros(n)
    for gid, params in fitted_params.items():
        a, b = params[0], params[1]
        t = params[2] if len(params) > 2 else float("nan")
        idx = gmatrix_idx.get(str(gid).strip().upper())
        if idx is not None and np.isfinite(a) and np.isfinite(b):
            fitted_mask[idx] = True
            alpha_vals[idx] = a
            beta_vals[idx] = b
            if has_topt and np.isfinite(t):
                topt_vals[idx] = t

    fitted_idx = np.where(fitted_mask)[0]
    n_fitted = len(fitted_idx)
    if n_fitted < 2:
        return {}

    # Sub-matrix of G for fitted genotypes  (n_fitted × n_fitted)
    G_sub = G[np.ix_(fitted_idx, fitted_idx)].copy()
    np.fill_diagonal(G_sub, 0.0)           # exclude self-kinship
    if positive_only:
        G_sub = np.maximum(G_sub, 0.0)

    alpha_fitted = alpha_vals[fitted_idx]
    beta_fitted = beta_vals[fitted_idx]
    topt_fitted = topt_vals[fitted_idx] if has_topt else None

    # Row sums = total positive kinship per genotype
    w = G_sub.sum(axis=1)                   # (n_fitted,)
    safe_w = np.maximum(w, 1e-10)

    # G-weighted average of neighbours
    target_alpha = (G_sub @ alpha_fitted) / safe_w
    target_beta = (G_sub @ beta_fitted) / safe_w
    target_topt = (G_sub @ topt_fitted) / safe_w if has_topt else None

    # Normalise weights so mean = 1.0
    w_mean = w.mean() if n_fitted > 0 else 1.0
    w_norm = w / max(w_mean, 1e-10)

    # Build result for fitted genotypes
    result: Dict[str, Tuple] = {}
    for k, idx in enumerate(fitted_idx):
        gid = gmatrix_ids[idx]
        if has_topt:
            result[gid] = (float(target_alpha[k]), float(target_beta[k]),
                           float(target_topt[k]),
                           float(w_norm[k]), float(w_norm[k]))
        else:
            result[gid] = (float(target_alpha[k]), float(target_beta[k]),
                           float(w_norm[k]), float(w_norm[k]))

    # Unseen genotypes: weighted average from ALL fitted (no self-exclusion)
    for i, gid in enumerate(gmatrix_ids):
        if not fitted_mask[i]:
            g_row = G[i, fitted_idx].copy()
            if positive_only:
                g_row = np.maximum(g_row, 0.0)
            w_i = g_row.sum()
            if w_i > 1e-10:
                ta = float((g_row * alpha_fitted).sum() / w_i)
                tb = float((g_row * beta_fitted).sum() / w_i)
                tt = float((g_row * topt_fitted).sum() / w_i) if has_topt else None
            else:
                ta = float(alpha_fitted.mean())
                tb = float(beta_fitted.mean())
                tt = float(topt_fitted.mean()) if has_topt else None
            wn = float(w_i / max(w_mean, 1e-10))
            if has_topt:
                result[gid] = (ta, tb, tt, wn, wn)
            else:
                result[gid] = (ta, tb, wn, wn)

    topt_msg = ""
    if has_topt and target_topt is not None:
        topt_msg = f", target topt range [{target_topt.min():.2f}, {target_topt.max():.2f}]"
    print(f"  G-matrix neighbor targets: {n_fitted} fitted genotypes, "
          f"w_norm range [{w_norm.min():.2f}, {w_norm.max():.2f}], "
          f"target alpha range [{target_alpha.min():.2f}, {target_alpha.max():.2f}]"
          f"{topt_msg}")
    return result


def gblup_predict_params(
    evd_path: str,
    fitted_params: Dict[str, Tuple[float, float]],
    rscript_path: Optional[str] = None,
    loo: bool = False,
    has_photo: bool = False,
) -> Dict[str, Tuple]:
    """GBLUP-predict (alpha, beta[, B, Pc]) for all G-matrix genotypes from fitted genotypes.

    Uses genomic relationships to predict mechanistic shape parameters for unseen
    genotypes. Alpha and beta are genotype-specific traits independent of environment,
    so this works even when training and prediction genotype panels are disjoint.

    Args:
        evd_path: Path to EVD.rda file
        fitted_params: {genotype_id: (alpha, beta[, B, Pc])} from mechanistic fitting
        rscript_path: Optional path to Rscript binary
        loo: If True, use LOO-GBLUP for fitted genotypes (each predicted from
             the other N-1). Essential for iterative penalty targets to avoid
             circular predictions that cause instant convergence.
        has_photo: If True, fitted_params tuples contain B, Pc as last 2 elements.

    Returns:
        Without has_photo: {gid: (alpha_pred, beta_pred, alpha_pev, beta_pev)}
        With has_photo: {gid: (alpha_pred, beta_pred, B_pred, Pc_pred,
                               alpha_pev, beta_pev, B_pev, Pc_pev)}
        For ALL G-matrix genotypes.
    """
    V, d, gmatrix_ids = load_evd(evd_path, rscript_path)
    n = len(gmatrix_ids)
    gmatrix_idx = {gid.upper(): i for i, gid in enumerate(gmatrix_ids)}

    # Build y-vectors for alpha and beta (and optionally B, Pc)
    y_alpha = np.zeros(n)
    y_beta = np.zeros(n)
    y_B = np.zeros(n) if has_photo else None
    y_Pc = np.zeros(n) if has_photo else None
    obs_mask = np.zeros(n, dtype=bool)
    for gid, params in fitted_params.items():
        alpha, beta = params[0], params[1]
        idx = gmatrix_idx.get(str(gid).strip().upper())
        if idx is not None and np.isfinite(alpha) and np.isfinite(beta):
            y_alpha[idx] = alpha
            y_beta[idx] = beta
            if has_photo:
                y_B[idx] = params[-2]
                y_Pc[idx] = params[-1]
            obs_mask[idx] = True

    n_obs = int(obs_mask.sum())
    if n_obs < 2:
        return {}

    # Full-data GBLUP (used for unseen genotypes and when loo=False)
    mean_alpha = y_alpha[obs_mask].mean()
    y_alpha_centered = np.where(obs_mask, y_alpha - mean_alpha, 0.0)
    z_alpha = V.T @ y_alpha_centered
    lambda_alpha = estimate_lambda_reml(d, z_alpha)
    u_alpha = V @ ((d / (d + lambda_alpha)) * z_alpha)

    mean_beta = y_beta[obs_mask].mean()
    y_beta_centered = np.where(obs_mask, y_beta - mean_beta, 0.0)
    z_beta = V.T @ y_beta_centered
    lambda_beta = estimate_lambda_reml(d, z_beta)
    u_beta = V @ ((d / (d + lambda_beta)) * z_beta)

    # GBLUP for photoperiod B, Pc (when fitting per-genotype)
    if has_photo:
        mean_B = y_B[obs_mask].mean()
        y_B_centered = np.where(obs_mask, y_B - mean_B, 0.0)
        z_B = V.T @ y_B_centered
        lambda_B = estimate_lambda_reml(d, z_B)
        u_B = V @ ((d / (d + lambda_B)) * z_B)

        mean_Pc = y_Pc[obs_mask].mean()
        y_Pc_centered = np.where(obs_mask, y_Pc - mean_Pc, 0.0)
        z_Pc = V.T @ y_Pc_centered
        lambda_Pc = estimate_lambda_reml(d, z_Pc)
        u_Pc = V @ ((d / (d + lambda_Pc)) * z_Pc)

    # Compute per-genotype PEV for unseen genotypes
    var_alpha = np.var(y_alpha[obs_mask])
    sg2_alpha = var_alpha / (1.0 + lambda_alpha) if lambda_alpha > 0 else var_alpha
    shrink_alpha = d / (d + lambda_alpha)

    var_beta = np.var(y_beta[obs_mask])
    sg2_beta = var_beta / (1.0 + lambda_beta) if lambda_beta > 0 else var_beta
    shrink_beta = d / (d + lambda_beta)

    if has_photo:
        var_B = np.var(y_B[obs_mask])
        sg2_B = var_B / (1.0 + lambda_B) if lambda_B > 0 else var_B
        shrink_B = d / (d + lambda_B)

        var_Pc = np.var(y_Pc[obs_mask])
        sg2_Pc = var_Pc / (1.0 + lambda_Pc) if lambda_Pc > 0 else var_Pc
        shrink_Pc = d / (d + lambda_Pc)

    # LOO-GBLUP for fitted genotypes (when requested)
    loo_preds = {}
    if loo and n_obs >= 3:
        loo_preds = _loo_gblup_spectral(
            V, d, y_alpha, y_beta, obs_mask, gmatrix_ids,
            y_B=y_B if has_photo else None,
            y_Pc=y_Pc if has_photo else None,
        )

    result = {}
    for i, gid in enumerate(gmatrix_ids):
        if obs_mask[i]:
            if loo and gid in loo_preds:
                # LOO prediction: predicted from N-1 genotypes (independent)
                result[gid] = loo_preds[gid]
            else:
                # Non-LOO: return original params, PEV=0
                orig = fitted_params.get(gid)
                if orig is None:
                    for fgid, fparams in fitted_params.items():
                        if str(fgid).strip().upper() == gid.upper():
                            orig = fparams
                            break
                if orig:
                    if has_photo:
                        result[gid] = (orig[0], orig[1], orig[-2], orig[-1],
                                       0.0, 0.0, 0.0, 0.0)
                    else:
                        result[gid] = (orig[0], orig[1], 0.0, 0.0)
                else:
                    if has_photo:
                        result[gid] = (mean_alpha + float(u_alpha[i]),
                                       mean_beta + float(u_beta[i]),
                                       mean_B + float(u_B[i]),
                                       mean_Pc + float(u_Pc[i]),
                                       0.0, 0.0, 0.0, 0.0)
                    else:
                        result[gid] = (mean_alpha + float(u_alpha[i]),
                                       mean_beta + float(u_beta[i]), 0.0, 0.0)
        else:
            # Unseen genotype: full-data GBLUP prediction with PEV
            reliability_alpha = float(np.sum(V[i, :] ** 2 * shrink_alpha))
            reliability_beta = float(np.sum(V[i, :] ** 2 * shrink_beta))
            pev_alpha = max(sg2_alpha * (1.0 - reliability_alpha), 1e-6)
            pev_beta = max(sg2_beta * (1.0 - reliability_beta), 1e-6)
            if has_photo:
                reliability_B = float(np.sum(V[i, :] ** 2 * shrink_B))
                reliability_Pc = float(np.sum(V[i, :] ** 2 * shrink_Pc))
                pev_B = max(sg2_B * (1.0 - reliability_B), 1e-6)
                pev_Pc = max(sg2_Pc * (1.0 - reliability_Pc), 1e-6)
                result[gid] = (
                    mean_alpha + float(u_alpha[i]),
                    mean_beta + float(u_beta[i]),
                    mean_B + float(u_B[i]),
                    mean_Pc + float(u_Pc[i]),
                    pev_alpha, pev_beta, pev_B, pev_Pc,
                )
            else:
                result[gid] = (
                    mean_alpha + float(u_alpha[i]),
                    mean_beta + float(u_beta[i]),
                    pev_alpha,
                    pev_beta,
                )

    n_unseen = n - n_obs
    loo_label = " (LOO)" if loo else ""
    photo_label = ""
    if has_photo:
        photo_label = f", lambda_B={lambda_B:.2f}, lambda_Pc={lambda_Pc:.2f}"
    print(f"  GBLUP param prediction{loo_label}: {n_obs} fitted -> {n_unseen} unseen genotypes "
          f"(lambda_alpha={lambda_alpha:.2f}, lambda_beta={lambda_beta:.2f}, "
          f"mean_alpha={mean_alpha:.4f}, mean_beta={mean_beta:.4f}{photo_label})")
    return result


def gblup_predict_photo3(
    evd_path: str,
    fitted_photo: Dict[str, Tuple[float, float, float]],
    rscript_path: Optional[str] = None,
    loo: bool = False,
) -> Dict[str, Tuple]:
    """GBLUP-predict logistic3 photoperiod params (a, b, d) from fitted genotypes.

    Same LOO-GBLUP spectral math as gblup_predict_params(), but for 3 photoperiod
    params instead of (alpha, beta).

    Args:
        evd_path: Path to EVD.rda file
        fitted_photo: {genotype_id: (a, b, d)} from photoperiod fitting
        loo: If True, use LOO-GBLUP (predict each from N-1)

    Returns:
        {gid: (a_pred, b_pred, d_pred, a_pev, b_pev, d_pev)} for all G-matrix genotypes.
    """
    V, d_eig, gmatrix_ids = load_evd(evd_path, rscript_path)
    n = len(gmatrix_ids)
    gmatrix_idx = {gid.upper(): i for i, gid in enumerate(gmatrix_ids)}

    # Build y-vectors for a, b, d
    y_a = np.zeros(n)
    y_b = np.zeros(n)
    y_d = np.zeros(n)
    obs_mask = np.zeros(n, dtype=bool)
    for gid, params in fitted_photo.items():
        idx = gmatrix_idx.get(str(gid).strip().upper())
        if idx is not None and all(np.isfinite(p) for p in params[:3]):
            y_a[idx] = params[0]
            y_b[idx] = params[1]
            y_d[idx] = params[2]
            obs_mask[idx] = True

    n_obs = int(obs_mask.sum())
    if n_obs < 2:
        return {}

    # Full-data GBLUP per param
    params_data = []
    for y_vec, name in [(y_a, "a"), (y_b, "b"), (y_d, "d")]:
        mean_val = y_vec[obs_mask].mean()
        y_centered = np.where(obs_mask, y_vec - mean_val, 0.0)
        z = V.T @ y_centered
        lam = estimate_lambda_reml(d_eig, z)
        u = V @ ((d_eig / (d_eig + lam)) * z)
        var_val = np.var(y_vec[obs_mask])
        sg2 = var_val / (1.0 + lam) if lam > 0 else var_val
        shrink = d_eig / (d_eig + lam)
        params_data.append((mean_val, u, lam, sg2, shrink, name))

    # LOO-GBLUP for fitted genotypes
    loo_preds = {}
    if loo and n_obs >= 3:
        # Run LOO for each param independently using _loo_gblup_spectral-like logic
        # We need per-genotype LOO predictions for a, b, d
        obs_indices = np.where(obs_mask)[0]
        obs_gids = [gmatrix_ids[i] for i in obs_indices]

        for i_obs, (gidx, gid) in enumerate(zip(obs_indices, obs_gids)):
            loo_mask = obs_mask.copy()
            loo_mask[gidx] = False
            loo_vals = []
            loo_pevs = []
            for y_vec, name in [(y_a, "a"), (y_b, "b"), (y_d, "d")]:
                mean_loo = y_vec[loo_mask].mean()
                y_loo = np.where(loo_mask, y_vec - mean_loo, 0.0)
                z_loo = V.T @ y_loo
                lam_loo = estimate_lambda_reml(d_eig, z_loo)
                u_loo = V @ ((d_eig / (d_eig + lam_loo)) * z_loo)
                pred_val = mean_loo + float(u_loo[gidx])
                # PEV
                shrink_loo = d_eig / (d_eig + lam_loo)
                var_loo = np.var(y_vec[loo_mask])
                sg2_loo = var_loo / (1.0 + lam_loo) if lam_loo > 0 else var_loo
                rel = float(np.sum(V[gidx, :] ** 2 * shrink_loo))
                pev = max(sg2_loo * (1.0 - rel), 1e-6)
                loo_vals.append(pred_val)
                loo_pevs.append(pev)
            loo_preds[gid] = tuple(loo_vals + loo_pevs)  # (a, b, d, pev_a, pev_b, pev_d)

    result = {}
    for i, gid in enumerate(gmatrix_ids):
        if obs_mask[i]:
            if loo and gid in loo_preds:
                result[gid] = loo_preds[gid]
            else:
                orig = fitted_photo.get(gid)
                if orig is None:
                    for fgid, fparams in fitted_photo.items():
                        if str(fgid).strip().upper() == gid.upper():
                            orig = fparams
                            break
                if orig:
                    result[gid] = (orig[0], orig[1], orig[2], 0.0, 0.0, 0.0)
                else:
                    result[gid] = tuple(
                        [pd[0] + float(pd[1][i]) for pd in params_data] +
                        [0.0, 0.0, 0.0]
                    )
        else:
            # Unseen genotype: full-data prediction with PEV
            vals = []
            pevs = []
            for mean_val, u, lam, sg2, shrink, name in params_data:
                vals.append(mean_val + float(u[i]))
                rel = float(np.sum(V[i, :] ** 2 * shrink))
                pevs.append(max(sg2 * (1.0 - rel), 1e-6))
            result[gid] = tuple(vals + pevs)

    n_unseen = n - n_obs
    loo_label = " (LOO)" if loo else ""
    lambdas = ", ".join(f"lambda_{pd[5]}={pd[2]:.2f}" for pd in params_data)
    means = ", ".join(f"mean_{pd[5]}={pd[0]:.4f}" for pd in params_data)
    print(f"  GBLUP photo3 prediction{loo_label}: {n_obs} fitted -> {n_unseen} unseen "
          f"({lambdas}, {means})")
    return result
