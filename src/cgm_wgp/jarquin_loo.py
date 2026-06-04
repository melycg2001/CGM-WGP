#!/usr/bin/env python3
"""
jarquin_loo.py

Jarquín et al. (2014) reaction norm model — leave-one-environment-out
cross-validation for comparison with EnvCov GBLUP.

Reference:
    Jarquín D, Crossa J, Lacaze X, et al. (2014) A reaction norm model for
    genomic selection using high-dimensional genomic and environmental data.
    Theor Appl Genet 127:595–607.

Model:
    y_ij = μ + g_i + e_j + ge_ij + ε_ij

    g   ~ MVN(0, G   · σ²_g)
    e   ~ MVN(0, E_K · σ²_e)
    ge  ~ MVN(0, G⊙E · σ²_ge)   [element-wise Hadamard in observation-space]
    ε   ~ N(0, I · σ²_ε)

    Fixed effects: overall intercept μ (environment means absorbed by e-random).

Environmental kernel E_K:
    - Per-planting weather covariates: Tmean, Tmin, Tmax, Trange
      computed over the observed growing season (Start_Date..End_Date).
    - W_std = standardized (col-wise, zero mean/unit sd across environments)
    - E_K = W_std @ W_std.T / n_cov   (positive semi-definite, 6×6)

Usage:
    python3.11 scripts/jarquin_loo.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

from cgm_wgp.gblup import load_evd

# ---------- paths ----------
PHENO_PATH = PROJECT_ROOT / "pipeline" / "1_data" / "Broccoli" / "phenotypes_dated.csv"
WEATHER_PATH = PROJECT_ROOT / "pipeline" / "1_data" / "weather" / "FL_Hasting" / "weather.csv"
EVD_PATH = str(PROJECT_ROOT / "pipeline" / "2_gmatrix" / "Broccoli" / "EVD.rda")
LOO_PRED_PATH = (
    PROJECT_ROOT
    / "pipeline"
    / "output"
    / "Broccoli"
    / "FL_Hasting"
    / "loo_cv_2026-02-27_22-14-02"
    / "loo_cv_predictions.csv"
)
EXCLUDE_GENOTYPES = {"FA306", "FA125"}

PLANTINGS = ["Planting 1", "Planting 2", "Planting 3", "Planting 4", "Planting 5", "Planting 6"]
PLANTING_COLORS = {
    "Planting 1": "#e41a1c",
    "Planting 2": "#ff7f00",
    "Planting 3": "#4daf4a",
    "Planting 4": "#377eb8",
    "Planting 5": "#984ea3",
    "Planting 6": "#8c4b00",
}


# ──────────────────────────────────────────────────────────────────────────────
# Environmental kernel
# ──────────────────────────────────────────────────────────────────────────────

def compute_env_kernel(
    pheno_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    env_order: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build E_K (n_env × n_env) from weather covariates per planting.

    Covariates per planting: Tmean, Tmin, Tmax, Trange (over Start_Date..End_Date).
    E_K = W_std @ W_std.T / n_cov
    """
    wdf = weather_df.copy()
    wdf["Period"] = pd.to_datetime(wdf["Period"])
    wdf["tmean"] = (wdf["tmx"] + wdf["tmn"]) / 2.0

    rows = []
    for env in env_order:
        env_data = pheno_df[pheno_df["Planting"] == env]
        start = pd.to_datetime(env_data["Start_Date"].min())
        end   = pd.to_datetime(env_data["End_Date"].max())

        mask = (wdf["Period"] >= start) & (wdf["Period"] <= end)
        w = wdf[mask]
        if len(w) == 0:
            raise ValueError(f"No weather data for {env} ({start} → {end})")

        rows.append({
            "env":    env,
            "Tmean":  float(w["tmean"].mean()),
            "Tmin":   float(w["tmn"].mean()),
            "Tmax":   float(w["tmx"].mean()),
            "Trange": float((w["tmx"] - w["tmn"]).mean()),
        })

    ec_df = pd.DataFrame(rows).set_index("env")
    W = ec_df[["Tmean", "Tmin", "Tmax", "Trange"]].values.astype(np.float64)
    W_std = (W - W.mean(axis=0)) / (W.std(axis=0) + 1e-12)

    n_cov = W_std.shape[1]
    E_K = (W_std @ W_std.T) / n_cov

    # Ensure numerical PSD (small ridge for near-zero eigenvalues)
    eigvals = np.linalg.eigvalsh(E_K)
    if eigvals.min() < -1e-10:
        print(f"  [warn] E_K has negative eigenvalue ({eigvals.min():.2e}); adding ridge.", file=sys.stderr)
    E_K += 1e-6 * np.eye(len(env_order))

    return E_K, ec_df


# ──────────────────────────────────────────────────────────────────────────────
# Photoperiod kernel (Option A: raw daylength stats per environment)
# ──────────────────────────────────────────────────────────────────────────────

def compute_photo_kernel(
    pheno_df: pd.DataFrame,
    env_order: list[str],
    latitude_map: dict[str, float],
    weather_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build P_K (n_env × n_env) from per-environment photoperiod summary stats.

    Covariates per planting (over Start_Date..End_Date):
        Mean_DL, Min_DL, Max_DL, Mean_Night

    Daylength source:
        - If weather_df is provided AND has a 'DAYLhr' column for the env, use it.
        - Otherwise, compute from latitude_map[env] + DOY via Spencer (1971).

    P_K = W_p_std @ W_p_std.T / n_cov     (PSD, mirrors E_K construction)
    """
    from cgm_wgp.mech_fit import compute_daylength_series

    rows = []
    for env in env_order:
        env_data = pheno_df[pheno_df["Planting"] == env]
        if env_data.empty:
            raise ValueError(f"No phenotype rows for env '{env}' when building P_K")
        start = pd.to_datetime(env_data["Start_Date"].min())
        end   = pd.to_datetime(env_data["End_Date"].max())

        dl_arr = None
        if weather_df is not None and "DAYLhr" in weather_df.columns and "SITE" in weather_df.columns:
            wdf_env = weather_df[weather_df["SITE"] == env]
            if not wdf_env.empty:
                wdf_env = wdf_env.copy()
                wdf_env["Period"] = pd.to_datetime(wdf_env["Period"])
                mask = (wdf_env["Period"] >= start) & (wdf_env["Period"] <= end)
                sub = wdf_env[mask]
                if not sub.empty:
                    dl_arr = sub["DAYLhr"].values.astype(float)

        if dl_arr is None:
            if env not in latitude_map:
                raise ValueError(
                    f"Cannot build P_K for env '{env}': no DAYLhr in weather "
                    f"and no latitude in latitude_map."
                )
            doys = pd.date_range(start, end).dayofyear
            dl_arr = compute_daylength_series(list(doys), latitude_map[env])

        rows.append({
            "env":         env,
            "Mean_DL":     float(np.mean(dl_arr)),
            "Min_DL":      float(np.min(dl_arr)),
            "Max_DL":      float(np.max(dl_arr)),
            "Mean_Night":  float(24.0 - np.mean(dl_arr)),
        })

    pc_df = pd.DataFrame(rows).set_index("env")
    Wp = pc_df[["Mean_DL", "Min_DL", "Max_DL", "Mean_Night"]].values.astype(np.float64)
    Wp_std = (Wp - Wp.mean(axis=0)) / (Wp.std(axis=0) + 1e-12)

    n_cov = Wp_std.shape[1]
    P_K = (Wp_std @ Wp_std.T) / n_cov

    eigvals = np.linalg.eigvalsh(P_K)
    if eigvals.min() < -1e-10:
        print(f"  [warn] P_K has negative eigenvalue ({eigvals.min():.2e}); adding ridge.", file=sys.stderr)
    P_K += 1e-6 * np.eye(len(env_order))

    return P_K, pc_df


# ──────────────────────────────────────────────────────────────────────────────
# REML fitting
# ──────────────────────────────────────────────────────────────────────────────

def _build_obs_kernels(
    gids: np.ndarray,   # (n_obs,) int indices into G rows/cols
    eids: np.ndarray,   # (n_obs,) int indices into E_K rows/cols
    G: np.ndarray,
    E_K: np.ndarray,
    P_K: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """
    Return observation-space kernels.

    If P_K is None: returns (K_G, K_E, K_GE) — original 4-VC behaviour.
    If P_K provided: returns (K_G, K_E, K_P, K_GE, K_GP) — 6-VC mode.
    """
    K_G  = G[np.ix_(gids, gids)]
    K_E  = E_K[np.ix_(eids, eids)]
    K_GE = K_G * K_E   # element-wise Hadamard (Jarquín GxE kernel)
    if P_K is None:
        return K_G, K_E, K_GE
    K_P  = P_K[np.ix_(eids, eids)]
    K_GP = K_G * K_P
    return K_G, K_E, K_P, K_GE, K_GP


def _reml_loglik(
    log_vc: np.ndarray,
    y: np.ndarray,
    K_G: np.ndarray,
    K_E: np.ndarray,
    K_GE: np.ndarray,
    K_P: np.ndarray | None = None,
    K_GP: np.ndarray | None = None,
) -> float:
    """
    Negative REML log-likelihood.

    4 VC mode (K_P/K_GP=None):  log_vc = [σ²_g, σ²_e, σ²_ge, σ²_ε]
    6 VC mode (K_P/K_GP given): log_vc = [σ²_g, σ²_e, σ²_p, σ²_ge, σ²_gp, σ²_ε]

    Fixed effect: overall intercept (X = column of ones).
    logL_R = -0.5 [log|V| + log|X'V⁻¹X| + (y - Xβ̂)'V⁻¹(y - Xβ̂)]
    """
    vc = np.exp(log_vc)
    n = len(y)

    if K_P is None:
        sg2, se2, sge2, sr2 = vc
        V = sg2 * K_G + se2 * K_E + sge2 * K_GE + sr2 * np.eye(n)
    else:
        sg2, se2, sp2, sge2, sgp2, sr2 = vc
        V = (sg2 * K_G + se2 * K_E + sp2 * K_P
             + sge2 * K_GE + sgp2 * K_GP + sr2 * np.eye(n))

    try:
        cf = cho_factor(V, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return 1e10

    L_diag = cf[0][np.arange(n), np.arange(n)]
    log_det_V = 2.0 * np.sum(np.log(np.abs(L_diag) + 1e-30))

    Vy   = cho_solve(cf, y,             check_finite=False)
    ones = np.ones(n)
    VX   = cho_solve(cf, ones,          check_finite=False)

    XtVX = float(ones @ VX)
    if XtVX <= 0:
        return 1e10

    mu_hat = float(ones @ Vy) / XtVX
    r      = y - mu_hat
    ytPy   = float(r @ cho_solve(cf, r, check_finite=False))

    return 0.5 * (log_det_V + np.log(XtVX) + ytPy)


def estimate_vc(
    y: np.ndarray,
    K_G: np.ndarray,
    K_E: np.ndarray,
    K_GE: np.ndarray,
    K_P: np.ndarray | None = None,
    K_GP: np.ndarray | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Estimate variance components via REML.

    4 VC mode (default):       returns [σ²_g, σ²_e,        σ²_ge,         σ²_ε]
    6 VC mode (K_P/K_GP given): returns [σ²_g, σ²_e, σ²_p, σ²_ge, σ²_gp, σ²_ε]

    Optimized in log-space with Nelder-Mead. Starting values: equal shares of
    total phenotypic variance across all components.
    """
    use_photo = K_P is not None
    n_vc = 6 if use_photo else 4

    var_y = float(np.var(y, ddof=1))
    vc0_each = max(var_y / n_vc, 1.0)
    log_vc0 = np.log([vc0_each] * n_vc)

    res = minimize(
        _reml_loglik,
        log_vc0,
        args=(y, K_G, K_E, K_GE, K_P, K_GP),
        method="Nelder-Mead",
        options={"maxiter": 4000 if use_photo else 2000,
                 "xatol": 1e-5, "fatol": 1e-5, "adaptive": True},
    )
    vc_hat = np.exp(res.x)
    if verbose:
        if use_photo:
            sg2, se2, sp2, sge2, sgp2, sr2 = vc_hat
            total = sg2 + se2 + sp2 + sge2 + sgp2 + sr2
            print(
                f"    VC: σ²_g={sg2:.2f}  σ²_e={se2:.2f}  σ²_p={sp2:.2f}"
                f"  σ²_ge={sge2:.2f}  σ²_gp={sgp2:.2f}  σ²_ε={sr2:.2f}"
                f"  (total={total:.2f})"
            )
        else:
            sg2, se2, sge2, sr2 = vc_hat
            total = sg2 + se2 + sge2 + sr2
            print(
                f"    VC: σ²_g={sg2:.2f}  σ²_e={se2:.2f}  σ²_ge={sge2:.2f}  σ²_ε={sr2:.2f}"
                f"  (total={total:.2f})"
            )
    return vc_hat


# ──────────────────────────────────────────────────────────────────────────────
# LOO-CV prediction
# ──────────────────────────────────────────────────────────────────────────────

def predict_held_out(
    y_train: np.ndarray,
    gids_train: np.ndarray,
    eids_train: np.ndarray,
    gids_predict: np.ndarray,
    eid_predict: int,
    G: np.ndarray,
    E_K: np.ndarray,
    vc: np.ndarray,
    P_K: np.ndarray | None = None,
) -> tuple:
    """
    BLUP predictions for all genotypes in the held-out environment.

    4 VC mode (P_K=None): vc = [σ²_g, σ²_e, σ²_ge, σ²_ε]
    6 VC mode (P_K given): vc = [σ²_g, σ²_e, σ²_p, σ²_ge, σ²_gp, σ²_ε]

    Returns (predictions, pred_vars).
    """
    use_photo = P_K is not None
    n_train = len(y_train)

    # Training kernel matrices
    K_G_tr  = G[np.ix_(gids_train, gids_train)]
    K_E_tr  = E_K[np.ix_(eids_train, eids_train)]
    K_GE_tr = K_G_tr * K_E_tr

    if use_photo:
        sg2, se2, sp2, sge2, sgp2, sr2 = vc
        K_P_tr  = P_K[np.ix_(eids_train, eids_train)]
        K_GP_tr = K_G_tr * K_P_tr
        V_train = (sg2 * K_G_tr + se2 * K_E_tr + sp2 * K_P_tr
                   + sge2 * K_GE_tr + sgp2 * K_GP_tr + sr2 * np.eye(n_train))
    else:
        sg2, se2, sge2, sr2 = vc
        V_train = sg2 * K_G_tr + se2 * K_E_tr + sge2 * K_GE_tr + sr2 * np.eye(n_train)

    cf_train = cho_factor(V_train, lower=True, check_finite=False)

    # Overall intercept (GLS)
    ones = np.ones(n_train)
    Vy_tr = cho_solve(cf_train, y_train, check_finite=False)
    VX_tr = cho_solve(cf_train, ones,    check_finite=False)
    mu_hat = float(ones @ Vy_tr) / float(ones @ VX_tr)

    # Residuals & V⁻¹r
    r = y_train - mu_hat
    Vr = cho_solve(cf_train, r, check_finite=False)

    # Predict for each genotype in held-out environment
    # Cross-covariance: k_cross[obs] = σ²_g G[g,obs] + σ²_e E_K[j*,obs_env]
    #                                  + σ²_ge G[g,obs]*E_K[j*,obs_env]
    #   (+ photo terms when use_photo)
    e_row = E_K[eid_predict, eids_train]  # shape (n_train,)
    e_self = float(E_K[eid_predict, eid_predict])
    if use_photo:
        p_row = P_K[eid_predict, eids_train]
        p_self = float(P_K[eid_predict, eid_predict])

    predictions = np.empty(len(gids_predict))
    pred_vars = np.empty(len(gids_predict))
    for i, g_idx in enumerate(gids_predict):
        g_row = G[g_idx, gids_train]                             # (n_train,)
        if use_photo:
            k_cross = (sg2 * g_row + se2 * e_row + sp2 * p_row
                       + sge2 * g_row * e_row + sgp2 * g_row * p_row)
        else:
            k_cross = sg2 * g_row + se2 * e_row + sge2 * g_row * e_row
        predictions[i] = mu_hat + float(k_cross @ Vr)

        # Per-genotype prediction error variance (conditional variance)
        g_self = float(G[g_idx, g_idx])
        if use_photo:
            var_new = (sg2 * g_self + se2 * e_self + sp2 * p_self
                       + sge2 * g_self * e_self + sgp2 * g_self * p_self + sr2)
        else:
            var_new = sg2 * g_self + se2 * e_self + sge2 * g_self * e_self + sr2
        Vk = cho_solve(cf_train, k_cross, check_finite=False)
        pred_vars[i] = max(var_new - float(k_cross @ Vk), 1e-6)

    return predictions, pred_vars


def run_loo_cv(
    pheno_df: pd.DataFrame,
    G: np.ndarray,
    gmatrix_ids: list[str],
    E_K: np.ndarray,
    env_order: list[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Leave-one-environment-out CV with the Jarquín model.

    Returns DataFrame with columns:
        gid, planting, observed, predicted
    """
    # Map genotype ID → row index in G
    gid_to_idx = {gid.upper().strip(): i for i, gid in enumerate(gmatrix_ids)}

    # Map environment → index in env_order
    env_to_eidx = {e: i for i, e in enumerate(env_order)}

    # Clean phenotype data: no excluded genotypes, non-NaN ft
    df = pheno_df.copy()
    df = df[~df["id"].isin(EXCLUDE_GENOTYPES)]
    df = df.dropna(subset=["ft"])
    df["id"] = df["id"].astype(str).str.strip()
    df["ft"] = df["ft"].astype(float)

    # Map each observation to G-matrix index (keep only those in G)
    df["g_idx"] = df["id"].str.upper().map(gid_to_idx)
    df = df.dropna(subset=["g_idx"])
    df["g_idx"] = df["g_idx"].astype(int)
    df["e_idx"] = df["Planting"].map(env_to_eidx).astype(int)

    all_rows = []

    for hold_idx, held_env in enumerate(env_order):
        print(f"  LOO fold {hold_idx+1}/{len(env_order)}: hold out {held_env}")

        train_mask = df["Planting"] != held_env
        df_train   = df[train_mask]
        df_test    = df[df["Planting"] == held_env]

        y_train    = df_train["ft"].values
        gids_train = df_train["g_idx"].values
        eids_train = df_train["e_idx"].values

        # Estimate variance components on training data
        K_G_tr, K_E_tr, K_GE_tr = _build_obs_kernels(gids_train, eids_train, G, E_K)
        vc = estimate_vc(y_train, K_G_tr, K_E_tr, K_GE_tr, verbose=verbose)

        # Prediction genotypes: all G-matrix genotypes that have observations
        # in the held-out environment
        gids_pred_unique = np.array(sorted(df_test["g_idx"].unique()))
        eid_pred = env_to_eidx[held_env]

        preds = predict_held_out(
            y_train, gids_train, eids_train,
            gids_pred_unique, eid_pred,
            G, E_K, vc,
        )

        # Map predictions back to rows
        pred_map = {g: float(p) for g, p in zip(gids_pred_unique, preds)}
        for _, row in df_test.iterrows():
            all_rows.append({
                "gid":       row["id"],
                "planting":  held_env,
                "observed":  int(row["ft"]),
                "predicted": round(pred_map[row["g_idx"]]),
            })

    return pd.DataFrame(all_rows)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline API: single-fold prediction for use in main_fit_alpha_beta.py
# ──────────────────────────────────────────────────────────────────────────────

def jarquin_predict_dap(
    pheno_df: pd.DataFrame,
    weather_source,
    evd_path: str,
    train_plantings: list,
    predict_plantings: list,
    use_photo: bool = False,
    latitude_map: dict | None = None,
    test_pheno_df: pd.DataFrame | None = None,
) -> tuple:
    """
    Single-fold Jarquín / RN-GBLUP prediction for pipeline use.

    Estimates variance components on training plantings and predicts DAP
    for all genotypes in predict_plantings observed in pheno_df.

    Parameters
    ----------
    pheno_df          DataFrame with: id, Planting, ft, Start_Date, End_Date
                      (already filtered to uncensored rows with ft present).
                      This is the TRAINING set — used for VC estimation and
                      to build K_G/K_E kernels over observed cells.
    weather_source    Path to a weather CSV file, or a directory containing
                      {planting}/weather.csv files (multi-location mode).
    evd_path          Path to G-matrix EVD .rda file.
    train_plantings   List of training planting names.
    predict_plantings List of planting names to predict.
    use_photo         If True, fit the extended model
                      y = µ + g + e + p + ge + gp + ε  (6 variance components),
                      adding a photoperiod environmental kernel P_K and the
                      G ⊙ P interaction kernel. Default False preserves the
                      original 4-VC Jarquín model byte-identically.
    latitude_map      Required when use_photo=True (and weather has no DAYLhr
                      column). Maps planting name → latitude, used by
                      Spencer (1971) to compute daylengths.
    test_pheno_df     Optional DataFrame of HELD-OUT (gid, env) cells for
                      validation. When provided, prediction_rows contain one
                      row per test cell (looked up from test_pheno_df for
                      observed DAPs) instead of iterating rows of pheno_df
                      at predict_plantings. Used by the balanced_training_subset
                      CV scheme to implement per-environment cell hold-out.
                      Same column schema as pheno_df.

    Returns
    -------
    (prediction_rows, info_dict)
    prediction_rows  list of dicts: id, planting, observed_dap, predicted_dap, error
    info_dict        {'vc': ...} — 4-vector or 6-vector depending on use_photo.
    """
    from pathlib import Path as _Path

    weather_path = _Path(weather_source)

    # ── Load G-matrix ──────────────────────────────────────────────────────────
    V, d, gmatrix_ids = load_evd(str(evd_path))
    d_clipped = np.maximum(d, 0.0)
    G = (V * d_clipped) @ V.T

    # ── Detect CV00 scenario (new genotypes AND new environment) ──────────────
    _train_gids = set(
        pheno_df[pheno_df["Planting"].isin(train_plantings)]["id"]
        .astype(str).str.strip().str.upper().unique()
    )
    # When held-out test cells are provided (balanced_training_subset CV
    # scheme), compute the overlap against the ACTUAL test genotypes instead
    # of inferring them from pheno_df. This keeps CV00 detection meaningful
    # when pheno_df contains only training rows.
    if test_pheno_df is not None:
        _predict_gids = set(
            test_pheno_df[test_pheno_df["Planting"].isin(predict_plantings)]["id"]
            .astype(str).str.strip().str.upper().unique()
        )
    else:
        _predict_gids = set(
            pheno_df[pheno_df["Planting"].isin(predict_plantings)]["id"]
            .astype(str).str.strip().str.upper().unique()
        )
    _overlap_ratio = (
        len(_train_gids & _predict_gids) / len(_predict_gids)
        if _predict_gids else 1.0
    )
    _CV00_THRESHOLD = 0.30
    # CV00 requires BOTH low genotype overlap AND low environment overlap.
    # new_varieties has zero genotype overlap but 100% env overlap (predict
    # envs == train envs) — that's CV1 (new genotypes), not CV00. Without
    # this check, _vc_pool_envs would be empty and REML would crash.
    _env_overlap = (
        len(set(train_plantings) & set(predict_plantings))
        / max(1, len(set(predict_plantings)))
    )
    _use_pooled_vc = (_overlap_ratio < _CV00_THRESHOLD
                      and _env_overlap < 0.50)

    _all_envs = list(pheno_df["Planting"].unique())
    if _use_pooled_vc:
        _vc_pool_envs = [e for e in _all_envs if e not in predict_plantings]
        print(
            f"  [CV00] Genotype overlap {_overlap_ratio:.1%}"
            f" < {_CV00_THRESHOLD:.0%} threshold,"
            f" env overlap {_env_overlap:.1%}"
        )
        print(
            f"  Pooling {len(_vc_pool_envs)} environments for VC estimation"
            f" (excluding predict)"
        )
    else:
        _vc_pool_envs = None

    # ── Build env_order ─────────────────────────────────────────────────────────
    seen: dict = {}
    if _use_pooled_vc:
        # Include ALL environments so E_K covers pooled + predict
        for pl in _all_envs:
            if pl not in seen:
                seen[pl] = None
        # Ensure predict plantings are in env_order even when absent from
        # pheno_df (cv00_double_novelty: predict env is fully held out)
        for pl in predict_plantings:
            if pl not in seen:
                seen[pl] = None
    else:
        for pl in list(train_plantings) + list(predict_plantings):
            if pl not in seen:
                seen[pl] = None
    env_order = list(seen.keys())

    # ── Build weather DataFrame ────────────────────────────────────────────────
    if weather_path.is_dir():
        frames = []
        _available_envs = []
        for pl in env_order:
            loc_file = weather_path / pl / "weather.csv"
            if not loc_file.exists():
                if (_use_pooled_vc
                        and pl not in train_plantings
                        and pl not in predict_plantings):
                    continue  # skip pooled env without weather
                raise FileNotFoundError(
                    f"Jarquín: weather file not found for planting '{pl}': {loc_file}"
                )
            frames.append(pd.read_csv(loc_file))
            _available_envs.append(pl)
        weather_df = pd.concat(frames, ignore_index=True)
        if _use_pooled_vc:
            env_order = _available_envs
            _vc_pool_envs = [e for e in _vc_pool_envs if e in _available_envs]
    else:
        weather_df = pd.read_csv(weather_path)

    # ── Environmental kernel ───────────────────────────────────────────────────
    # When predict environments are absent from pheno_df (cv00_double_novelty),
    # augment with date info from test_pheno_df so E_K/P_K construction can
    # determine the growing season date range for those environments.
    pheno_for_kernel = pheno_df
    if test_pheno_df is not None:
        _missing_envs = set(predict_plantings) - set(pheno_df["Planting"].unique())
        if _missing_envs:
            _date_rows = test_pheno_df[test_pheno_df["Planting"].isin(_missing_envs)]
            if not _date_rows.empty:
                _augment = _date_rows.drop_duplicates(subset=["Planting"]).copy()
                for c in pheno_df.columns:
                    if c not in _augment.columns:
                        _augment[c] = np.nan
                pheno_for_kernel = pd.concat(
                    [pheno_df, _augment[pheno_df.columns]], ignore_index=True
                )
    E_K, _ = compute_env_kernel(pheno_for_kernel, weather_df, env_order)

    # ── Photoperiod kernel (Option A: raw daylength stats per env) ────────────
    P_K = None
    if use_photo:
        if latitude_map is None and "DAYLhr" not in weather_df.columns:
            raise ValueError(
                "use_photo=True requires latitude_map (or DAYLhr in weather)."
            )
        P_K, _pc_df = compute_photo_kernel(
            pheno_for_kernel, env_order, latitude_map or {}, weather_df=weather_df
        )
        print(
            f"  RN-GBLUP photoperiod kernel built for {len(env_order)} envs"
            f" (Mean DL range: {_pc_df['Mean_DL'].min():.2f}–{_pc_df['Mean_DL'].max():.2f}h)"
        )

    # ── Map genotype → G-matrix index ─────────────────────────────────────────
    gid_to_idx = {gid.upper().strip(): i for i, gid in enumerate(gmatrix_ids)}
    env_to_eidx = {e: i for i, e in enumerate(env_order)}

    # ── Prepare phenotype data ─────────────────────────────────────────────────
    df = pheno_df.copy()
    df = df.dropna(subset=["ft"])
    df["id"] = df["id"].astype(str).str.strip()
    df["ft"] = df["ft"].astype(float)
    df["g_idx"] = df["id"].str.upper().map(gid_to_idx)
    df = df.dropna(subset=["g_idx"])
    df["g_idx"] = df["g_idx"].astype(int)
    df["e_idx"] = df["Planting"].map(env_to_eidx)

    # ── Training data ──────────────────────────────────────────────────────────
    df_train = df[df["Planting"].isin(train_plantings)].dropna(subset=["e_idx"]).copy()
    df_train["e_idx"] = df_train["e_idx"].astype(int)
    y_train = df_train["ft"].values
    gids_train = df_train["g_idx"].values
    eids_train = df_train["e_idx"].values

    if len(y_train) < 3:
        raise ValueError(
            f"Too few training observations ({len(y_train)}) for Jarquín model."
        )

    # ── Estimate variance components ───────────────────────────────────────────
    if _use_pooled_vc:
        # CV00: pool all environments (except predict) for VC estimation
        df_vc = df[df["Planting"].isin(_vc_pool_envs)].dropna(subset=["e_idx"]).copy()
        df_vc["e_idx"] = df_vc["e_idx"].astype(int)
        y_vc = df_vc["ft"].values
        gids_vc = df_vc["g_idx"].values
        eids_vc = df_vc["e_idx"].values
        print(
            f"  Pooled VC: {len(y_vc)} obs,"
            f" {len(set(gids_vc))} genotypes,"
            f" {len(set(eids_vc))} environments"
        )
        if use_photo:
            K_G_vc, K_E_vc, K_P_vc, K_GE_vc, K_GP_vc = _build_obs_kernels(
                gids_vc, eids_vc, G, E_K, P_K=P_K
            )
            vc = estimate_vc(y_vc, K_G_vc, K_E_vc, K_GE_vc,
                             K_P=K_P_vc, K_GP=K_GP_vc, verbose=True)
        else:
            K_G_vc, K_E_vc, K_GE_vc = _build_obs_kernels(gids_vc, eids_vc, G, E_K)
            vc = estimate_vc(y_vc, K_G_vc, K_E_vc, K_GE_vc, verbose=True)
    else:
        if use_photo:
            K_G_tr, K_E_tr, K_P_tr, K_GE_tr, K_GP_tr = _build_obs_kernels(
                gids_train, eids_train, G, E_K, P_K=P_K
            )
            vc = estimate_vc(y_train, K_G_tr, K_E_tr, K_GE_tr,
                             K_P=K_P_tr, K_GP=K_GP_tr, verbose=True)
        else:
            K_G_tr, K_E_tr, K_GE_tr = _build_obs_kernels(gids_train, eids_train, G, E_K)
            vc = estimate_vc(y_train, K_G_tr, K_E_tr, K_GE_tr, verbose=False)

    # ── Build held-out cells DataFrame for prediction iteration ──────────────
    # When test_pheno_df is provided, iterate its rows (one row per held-out
    # (gid, env) cell). Otherwise fall back to iterating the training df rows
    # at each predict env (original behavior).
    if test_pheno_df is not None:
        test_df = test_pheno_df.copy()
        test_df["id"] = test_df["id"].astype(str).str.strip()
        test_df["Planting"] = test_df["Planting"].astype(str)
        test_df["ft"] = test_df["ft"].astype(float)
        test_df["g_idx"] = test_df["id"].str.upper().map(gid_to_idx)
        test_df = test_df.dropna(subset=["g_idx"])
        test_df["g_idx"] = test_df["g_idx"].astype(int)
        test_df["e_idx"] = test_df["Planting"].map(env_to_eidx)
        test_df = test_df.dropna(subset=["e_idx"])
        test_df["e_idx"] = test_df["e_idx"].astype(int)
    else:
        test_df = None

    # ── Predict for each predict planting ─────────────────────────────────────
    prediction_rows: list = []
    for pred_env in predict_plantings:
        if pred_env not in env_to_eidx:
            print(
                f"  [warn] Jarquín: predict planting '{pred_env}' not in env_order",
                file=sys.stderr,
            )
            continue

        eid_pred = env_to_eidx[pred_env]
        if test_df is not None:
            df_pred_env = test_df[test_df["Planting"] == pred_env]
        else:
            df_pred_env = df[df["Planting"] == pred_env].dropna(subset=["e_idx"])
        if df_pred_env.empty:
            continue

        gids_pred_unique = np.array(sorted(df_pred_env["g_idx"].unique()))
        preds, pred_vars_arr = predict_held_out(
            y_train, gids_train, eids_train,
            gids_pred_unique, eid_pred,
            G, E_K, vc,
            P_K=P_K,
        )
        pred_map = {int(g): float(p) for g, p in zip(gids_pred_unique, preds)}
        var_map = {int(g): float(v) for g, v in zip(gids_pred_unique, pred_vars_arr)}

        for _, row in df_pred_env.iterrows():
            gid = str(row["id"])
            g_idx = int(row["g_idx"])
            observed_dap = float(row["ft"]) if pd.notna(row["ft"]) else None
            predicted_raw = pred_map.get(g_idx)
            predicted_dap = round(predicted_raw) if predicted_raw is not None else None
            pred_var = var_map.get(g_idx)
            error = (
                (predicted_dap - observed_dap)
                if (predicted_dap is not None and observed_dap is not None)
                else None
            )
            prediction_rows.append({
                "id": gid,
                "planting": pred_env,
                "observed_dap": int(observed_dap) if observed_dap is not None else "NA",
                "predicted_dap": int(predicted_dap) if predicted_dap is not None else "NA",
                "error": f"{error:.1f}" if error is not None else "NA",
                "pred_var": float(pred_var) if pred_var is not None else "NA",
            })

    # Build per-genotype variance map for ensemble weighting
    pred_var_map = {}
    for r in prediction_rows:
        if r["pred_var"] != "NA":
            pred_var_map[(r["id"], r["planting"])] = r["pred_var"]

    info_dict = {
        "vc": vc,
        "pred_var_map": pred_var_map,
        "pooled_vc": _use_pooled_vc,
        "overlap_ratio": _overlap_ratio,
    }
    return prediction_rows, info_dict


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    return float(stats.pearsonr(a[mask], b[mask])[0])


def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 1:
        return np.nan
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def print_metrics(label: str, df: pd.DataFrame):
    r_all  = pearson(df["predicted"], df["observed"])
    r_rmse = rmse(df["predicted"], df["observed"])
    print(f"  {label}: PA={r_all:.3f}  RMSE={r_rmse:.2f}")
    for pl in PLANTINGS:
        sub = df[df["planting"] == pl]
        if sub.empty:
            continue
        r_pl   = pearson(sub["predicted"], sub["observed"])
        rm_pl  = rmse(sub["predicted"], sub["observed"])
        print(f"    {pl}: PA={r_pl:.3f}  RMSE={rm_pl:.2f}  n={len(sub)}")


# ──────────────────────────────────────────────────────────────────────────────
# Comparison figure
# ──────────────────────────────────────────────────────────────────────────────

def make_comparison_figure(
    df_jarquin: pd.DataFrame,
    df_envcov: pd.DataFrame,
    out_path: Path,
):
    """
    2-panel scatter: Jarquín model | EnvCov GBLUP
    Points colored by planting; per-planting and overall r in legend/title.
    """
    panels = [
        (df_jarquin, "Jarquín Reaction Norm"),
        (df_envcov,  "EnvCov GBLUP"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))

    for ax, (df, title) in zip(axes, panels):
        all_v = pd.concat([df["observed"], df["predicted"]]).dropna()
        lo, hi = all_v.min() - 2, all_v.max() + 2

        ax.plot([lo, hi], [lo, hi], color="0.55", lw=0.9, ls="--", zorder=1)

        for pl in PLANTINGS:
            sub = df[df["planting"] == pl]
            if sub.empty:
                continue
            r_pl = pearson(sub["predicted"], sub["observed"])
            label = f"P{PLANTINGS.index(pl)+1}  r={r_pl:.2f}"
            ax.scatter(
                sub["observed"], sub["predicted"],
                color=PLANTING_COLORS[pl],
                s=10, alpha=0.65, zorder=3,
                label=label, rasterized=True,
            )

        r_all = pearson(df["predicted"], df["observed"])
        rmse_all = rmse(df["predicted"], df["observed"])
        ax.set_title(f"{title}\nr = {r_all:.2f}   RMSE = {rmse_all:.1f} d", fontsize=10)
        ax.set_xlabel("Observed DAP (days)")
        if ax is axes[0]:
            ax.set_ylabel("Predicted DAP (days)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(
            fontsize=7.5, loc="upper left",
            handlelength=0.8, handletextpad=0.4, borderpad=0.4,
        )
        ax.grid(lw=0.3, color="0.9", zorder=0)

    fig.suptitle("LOO-CV: Jarquín Reaction Norm vs EnvCov GBLUP — Broccoli", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def make_per_planting_bar(
    df_jarquin: pd.DataFrame,
    df_envcov: pd.DataFrame,
    out_path_pa: Path,
    out_path_rmse: Path,
):
    """
    Two side-by-side grouped bar charts: PA and RMSE per planting.
    Bars: Jarquín (blue-gray) vs EnvCov GBLUP (red).
    """
    models  = ["Jarquín", "EnvCov GBLUP"]
    colors  = ["#4393c3", "#d7191c"]
    dfs     = [df_jarquin, df_envcov]

    x = np.arange(len(PLANTINGS))
    w = 0.35
    offsets = [-w / 2, w / 2]
    labels  = [f"P{i+1}" for i in range(len(PLANTINGS))]

    for metric, ylabel, out_path in [
        ("PA",   "Pearson r",  out_path_pa),
        ("RMSE", "RMSE (days)", out_path_rmse),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for df, model, color, offset in zip(dfs, models, colors, offsets):
            vals = []
            for pl in PLANTINGS:
                sub = df[df["planting"] == pl]
                v = pearson(sub["predicted"], sub["observed"]) if metric == "PA" else rmse(sub["predicted"], sub["observed"])
                vals.append(v)
            ax.bar(x + offset, vals, width=w, label=model, color=color, alpha=0.85, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Planting")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(axis="y", lw=0.3, color="0.9", zorder=0)
        if metric == "PA":
            ax.axhline(0, color="0.4", lw=0.8, ls="--")
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Jarquín reaction norm LOO-CV")
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "pipeline" / "output" / "publication_figures"),
        help="Output directory for figures and predictions CSV",
    )
    parser.add_argument(
        "--no-refit", action="store_true",
        help="Load existing jarquin_loo_predictions.csv instead of refitting",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / "jarquin_loo_predictions.csv"

    # ── Load phenotype data ──────────────────────────────────────────────────
    print("Loading phenotype data...")
    pheno_df = pd.read_csv(PHENO_PATH)

    # ── Load or compute EVD → G ──────────────────────────────────────────────
    print("Loading G-matrix EVD...")
    V, d, gmatrix_ids = load_evd(EVD_PATH)
    # Reconstruct G = V @ diag(d) @ V.T   (n_g × n_g)
    d_clipped = np.maximum(d, 0.0)   # numerical PSD guarantee
    G = (V * d_clipped) @ V.T
    print(f"  G-matrix: {G.shape[0]} genotypes")

    # ── Environmental kernel ─────────────────────────────────────────────────
    print("Computing environmental kernel E_K...")
    weather_df = pd.read_csv(WEATHER_PATH)
    E_K, ec_df = compute_env_kernel(pheno_df, weather_df, PLANTINGS)
    print("  Weather covariates per planting:")
    print(ec_df.round(2).to_string())
    print()
    print("  E_K (environmental kernel):")
    ek_labels = [f"P{i+1}" for i in range(len(PLANTINGS))]
    ek_df = pd.DataFrame(E_K, index=ek_labels, columns=ek_labels)
    print(ek_df.round(3).to_string())
    print()

    # ── LOO-CV ───────────────────────────────────────────────────────────────
    if args.no_refit and pred_csv.exists():
        print(f"Loading cached predictions from {pred_csv}")
        df_jarquin = pd.read_csv(pred_csv)
    else:
        print("Running Jarquín LOO-CV (this may take a few minutes)...")
        df_jarquin = run_loo_cv(pheno_df, G, gmatrix_ids, E_K, PLANTINGS, verbose=True)
        df_jarquin.to_csv(pred_csv, index=False)
        print(f"  Saved predictions → {pred_csv.name}")

    print()
    print_metrics("Jarquín model", df_jarquin)

    # ── Load EnvCov GBLUP predictions for comparison ─────────────────────────
    print()
    df_loo_all = pd.read_csv(LOO_PRED_PATH)
    df_envcov = df_loo_all[df_loo_all["model"] == "EnvCov GBLUP"].copy()
    df_envcov = df_envcov.rename(columns={"gid": "gid", "planting": "planting",
                                           "predicted": "predicted", "observed": "observed"})
    df_envcov = df_envcov.dropna(subset=["predicted", "observed"])
    print_metrics("EnvCov GBLUP", df_envcov)

    # ── Figures ───────────────────────────────────────────────────────────────
    print()
    print("Generating figures...")

    make_comparison_figure(
        df_jarquin,
        df_envcov,
        out_dir / "fig13_jarquin_vs_envcov_scatter.png",
    )
    make_per_planting_bar(
        df_jarquin,
        df_envcov,
        out_dir / "fig14_jarquin_vs_envcov_pa.png",
        out_dir / "fig15_jarquin_vs_envcov_rmse.png",
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
