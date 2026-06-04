#!/usr/bin/env python3
"""
joint_fit.py

Joint CGM-WGP fitting following the Messina/Technow framework.

Estimates per-genotype physiological parameters (Theta_g, S_g, Pc_g) jointly
across all genotypes under a multivariate normal genomic prior, using an
EM / Profile Likelihood algorithm with L-BFGS-B for the inner loop.

Cardinal temperatures (Tb, Topt, Tc) are fixed at species level. Genetic
variation is in developmental requirements (Theta_g), photoperiod sensitivity
(S_g), and critical photoperiod (Pc_g).

Reference:
    Messina CD, Technow F, Tang T, et al. (2018) Leveraging biological insight
    and environmental variation to improve phenotypic prediction: Integrating
    crop growth models (CGM) with whole genome prediction (WGP).
    Eur J Agron 100:151-162.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from cgm_wgp.gblup import load_evd
from cgm_wgp.mech_fit import (
    cardinal_beta,
    logistic3_photoperiod_array,
    load_and_build_dicts,
)

# ============================================================
# Constants & Derived Parameters
# ============================================================

def derive_fixed_alpha_beta(
    Tb: float, Topt: float, Tc: float, beta_base: float = 1.0
) -> Tuple[float, float]:
    """Derive alpha from fixed cardinals using Yin & Kropff (1995).

    The cardinal beta function f(T) = ((T-Tb)/(Topt-Tb))^alpha * ((Tc-T)/(Tc-Topt))^beta
    has its maximum at Topt when alpha/beta = (Topt-Tb)/(Tc-Topt).
    """
    alpha_fixed = beta_base * (Topt - Tb) / (Tc - Topt)
    return alpha_fixed, beta_base


# ============================================================
# Forward Model
# ============================================================

def forward_dap_fractional(
    Theta_g: float,
    S_g: float,
    Pc_g: float,
    thermal_rates: np.ndarray,
    DL_series: Optional[np.ndarray],
    a_fixed: float = 1.0,
    photo_enabled: bool = True,
    photo_direction: str = "short_day",
) -> float:
    """Predict fractional DAP for one genotype in one environment.

    Hadley additive photoperiod formulation (S_g >= 0):
        short_day:  deviation_t = max(0, DL_t - Pc_g)   (penalty on long days)
        long_day:   deviation_t = max(0, Pc_g - DL_t)   (penalty on short days)
        base_rate_t  = max(0, thermal_rate_t - S_g * deviation_t)
        daily_rate_t = base_rate_t / Theta_g
        flowering when cumsum(daily_rate) >= 1.0

    Pc_g is the critical daylength (h). `a_fixed` is retained for call-site
    compatibility and unused in the additive formulation.

    Returns fractional DAP (linear interpolation at threshold crossing).
    Returns -1.0 if threshold never reached.
    """
    if Theta_g <= 0:
        return -1.0

    n = len(thermal_rates)
    if n == 0:
        return -1.0

    # Compute daily rates (Hadley additive)
    if photo_enabled and DL_series is not None and len(DL_series) > 0:
        n = min(n, len(DL_series))
        if photo_direction == "long_day":
            deviation = np.maximum(0.0, Pc_g - DL_series[:n])
        else:
            deviation = np.maximum(0.0, DL_series[:n] - Pc_g)
        base = np.maximum(0.0, thermal_rates[:n] - S_g * deviation)
        daily_rates = base / Theta_g
    else:
        daily_rates = thermal_rates[:n] / Theta_g

    cumsum = np.cumsum(daily_rates)
    target = 1.0

    indices = np.where(cumsum >= target)[0]
    if len(indices) == 0:
        return -1.0  # never reached

    day_idx = indices[0]
    # Fractional interpolation
    if day_idx == 0:
        c_curr = cumsum[0]
        frac = target / c_curr if c_curr > 0 else 1.0
        return frac
    else:
        c_prev = cumsum[day_idx - 1]
        c_curr = cumsum[day_idx]
        denom = c_curr - c_prev
        frac = (target - c_prev) / denom if denom > 0 else 1.0
        return float(day_idx) + frac


# ============================================================
# Parameter Packing / Unpacking
# ============================================================

def pack_params(
    Theta: np.ndarray, S: np.ndarray, Pc: np.ndarray, photo_enabled: bool
) -> np.ndarray:
    if photo_enabled:
        return np.concatenate([Theta, S, Pc])
    else:
        return Theta.copy()


def unpack_params(
    x: np.ndarray, G: int, photo_enabled: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if photo_enabled:
        Theta = x[:G]
        S = x[G:2*G]
        Pc = x[2*G:3*G]
    else:
        Theta = x[:G]
        S = np.zeros(G)
        Pc = np.full(G, 14.0)
    return Theta, S, Pc


def build_bounds(
    G: int,
    photo_enabled: bool,
    Theta_bounds: Tuple[float, float] = (5.0, 200.0),
    S_bounds: Tuple[float, float] = (-5.0, 5.0),
    Pc_bounds: Tuple[float, float] = (10.0, 18.0),
) -> list:
    bounds = [Theta_bounds] * G
    if photo_enabled:
        bounds += [S_bounds] * G
        bounds += [Pc_bounds] * G
    return bounds


# ============================================================
# Genomic Prior (in eigenspace)
# ============================================================

def compute_genomic_prior(
    param_vectors: List[np.ndarray],
    mu: np.ndarray,
    sigma2: np.ndarray,
    V: np.ndarray,
    d_reg: np.ndarray,
) -> Tuple[float, float]:
    """Compute genomic prior and log-determinant penalty in eigenspace.

    prior = sum_k (1/sigma2_k) * sum_j z_{k,j}^2 / d_reg_j
    logdet = sum_k G * log(sigma2_k)

    Returns (prior_cost, logdet_cost).
    """
    G = V.shape[0]
    inv_d = 1.0 / d_reg  # (n_eigenvalues,)
    prior = 0.0
    logdet = 0.0
    for k, theta_k in enumerate(param_vectors):
        centered = theta_k - mu[k]
        z = V.T @ centered  # rotate into eigenspace
        quadform = np.sum(z**2 * inv_d)
        prior += quadform / max(sigma2[k], 1e-12)
        logdet += G * np.log(max(sigma2[k], 1e-12))
    return prior, logdet


# ============================================================
# Inner Objective with Analytic Gradients
# ============================================================

def _forward_one_cell(
    Theta_g: float, S_g: float, Pc_g: float,
    thermal: np.ndarray, dl: Optional[np.ndarray],
    a_fixed: float, photo_enabled: bool,
    photo_direction: str = "short_day",
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Forward model for one (genotype, environment) cell.

    Hadley additive photoperiod:
        short_day:  base_rate_t = max(0, thermal_t - S_g * max(0, DL_t - Pc_g))
        long_day:   base_rate_t = max(0, thermal_t - S_g * max(0, Pc_g - DL_t))

    Returns:
        n: number of valid days
        base_rates: per-day rate (before /Theta)
        daily: base_rates / Theta (the actual daily rates)
    """
    n = len(thermal)
    if photo_enabled and dl is not None and len(dl) > 0:
        n = min(n, len(dl))
        if photo_direction == "long_day":
            deviation = np.maximum(0.0, Pc_g - dl[:n])
        else:
            deviation = np.maximum(0.0, dl[:n] - Pc_g)
        base_rates = np.maximum(0.0, thermal[:n] - S_g * deviation)
    else:
        base_rates = thermal[:n].copy()

    Theta_safe = max(Theta_g, 1e-8)
    daily = base_rates / Theta_safe
    return n, base_rates, daily


def _compute_pred_and_grad_theta(
    Theta_g: float, base_rates: np.ndarray, n: int,
) -> Tuple[float, float]:
    """Compute predicted DAP and its gradient w.r.t. Theta_g.

    Since daily_rate = base_rate / Theta, and cumsum crosses 1.0:
      pred = Theta / cumulative_base_at_crossing (approximately)

    More precisely, with fractional interpolation:
      cs_base = cumsum(base_rates)  (independent of Theta)
      The crossing occurs when cs_base[k] / Theta >= 1, i.e., cs_base[k] >= Theta
      So pred = k + (Theta - cs_base[k-1]) / base_rates[k]

    Gradient: d(pred)/d(Theta) = 1 / base_rates[k]  (at the crossing day)
    """
    Theta_safe = max(Theta_g, 1e-8)
    cs_base = np.cumsum(base_rates[:n])

    # Find crossing: cs_base[k] >= Theta
    idx = np.searchsorted(cs_base, Theta_safe, side='left')

    if idx >= n:
        # Never flowered — pred = n (penalty)
        pred = float(n)
        dpred_dTheta = 0.0  # flat penalty region
    elif idx == 0:
        # Crosses on first day
        if cs_base[0] > 0:
            pred = Theta_safe / cs_base[0]
            dpred_dTheta = 1.0 / cs_base[0]
        else:
            pred = 1.0
            dpred_dTheta = 0.0
    else:
        cb_prev = cs_base[idx - 1]
        r_k = base_rates[idx]
        if r_k > 1e-12:
            frac = (Theta_safe - cb_prev) / r_k
            pred = float(idx) + frac
            dpred_dTheta = 1.0 / r_k
        else:
            pred = float(idx) + 1.0
            dpred_dTheta = 0.0

    return pred, dpred_dTheta


def _compute_pred_and_grad_photo(
    Theta_g: float, S_g: float, Pc_g: float,
    thermal: np.ndarray, dl: np.ndarray,
    a_fixed: float, n: int,
    photo_direction: str = "short_day",
) -> Tuple[float, float, float]:
    """Compute d(pred)/d(S_g) and d(pred)/d(Pc_g) for Hadley additive model.

    Model:
        short_day:  deviation_t = max(0, DL_t - Pc_g)
        long_day:   deviation_t = max(0, Pc_g - DL_t)
        base_t     = max(0, thermal_t - S_g * deviation_t)

    Per-day derivatives in the active region
      (deviation_t > 0  AND  thermal_t > S_g * deviation_t):
        d(base_t)/d(S_g)  = -deviation_t
        short_day: d(base_t)/d(Pc_g) = -S_g * d(deviation)/d(Pc) = -S_g*(-1) = +S_g
        long_day:  d(base_t)/d(Pc_g) = -S_g * (+1)                        = -S_g
      Otherwise: both derivatives = 0.

    Implicit function theorem at the crossing F(pred, ·) = Theta:
        d(pred)/d(param) = -cum[d(base)/d(param)] / base_rates[idx]
    """
    Theta_safe = max(Theta_g, 1e-8)

    dl_arr = dl[:n]
    thermal_arr = thermal[:n]
    if photo_direction == "long_day":
        deviation = np.maximum(0.0, Pc_g - dl_arr)
        dPc_sign = -1.0
    else:
        deviation = np.maximum(0.0, dl_arr - Pc_g)
        dPc_sign = +1.0
    base_rates = np.maximum(0.0, thermal_arr - S_g * deviation)

    cs_base = np.cumsum(base_rates)
    idx = np.searchsorted(cs_base, Theta_safe, side='left')

    if idx >= n or idx == 0:
        return 0.0, 0.0  # penalty or edge — no useful gradient

    r_k = base_rates[idx]
    if r_k < 1e-12:
        return 0.0, 0.0

    # Active: deviation > 0 AND rate not clamped to 0.
    active = (deviation > 0.0) & (thermal_arr > S_g * deviation)

    dbase_dS = np.where(active, -deviation, 0.0)
    dbase_dPc = np.where(active, dPc_sign * S_g, 0.0)

    cb_prev = cs_base[idx - 1] if idx > 0 else 0.0
    frac = (Theta_safe - cb_prev) / r_k

    cum_dbase_dS = np.sum(dbase_dS[:idx]) + frac * dbase_dS[idx]
    cum_dbase_dPc = np.sum(dbase_dPc[:idx]) + frac * dbase_dPc[idx]

    dpred_dS = -cum_dbase_dS / r_k
    dpred_dPc = -cum_dbase_dPc / r_k

    return dpred_dS, dpred_dPc


def _batch_forward_sse_and_grad(
    Theta: np.ndarray,
    S: np.ndarray,
    Pc: np.ndarray,
    G: int,
    obs_cells_by_env: Dict[str, List[Tuple[int, int]]],
    precomputed_thermal: Dict[str, np.ndarray],
    precomputed_dl: Dict[str, Optional[np.ndarray]],
    a_fixed: float,
    photo_enabled: bool,
    photo_direction: str = "short_day",
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute SSE and gradients w.r.t. Theta, S, Pc."""
    sse = 0.0
    grad_Theta = np.zeros(G)
    grad_S = np.zeros(G)
    grad_Pc = np.zeros(G)

    for env_key, cells in obs_cells_by_env.items():
        thermal = precomputed_thermal.get(env_key)
        if thermal is None:
            continue
        dl = precomputed_dl.get(env_key)

        for g_idx, obs_dap in cells:
            n, base_rates, daily = _forward_one_cell(
                Theta[g_idx], S[g_idx], Pc[g_idx],
                thermal, dl, a_fixed, photo_enabled,
                photo_direction=photo_direction,
            )

            # Predicted DAP and gradient w.r.t. Theta
            pred, dpred_dTheta = _compute_pred_and_grad_theta(
                Theta[g_idx], base_rates, n,
            )

            residual = pred - obs_dap
            sse += residual ** 2

            # Gradient contribution: 2 * residual * dpred/dparam
            grad_Theta[g_idx] += 2.0 * residual * dpred_dTheta

            # Photoperiod gradients
            if photo_enabled and dl is not None and len(dl) > 0:
                dpred_dS, dpred_dPc = _compute_pred_and_grad_photo(
                    Theta[g_idx], S[g_idx], Pc[g_idx],
                    thermal, dl, a_fixed, n,
                    photo_direction=photo_direction,
                )
                grad_S[g_idx] += 2.0 * residual * dpred_dS
                grad_Pc[g_idx] += 2.0 * residual * dpred_dPc

    return sse, grad_Theta, grad_S, grad_Pc


def _genomic_prior_and_grad(
    param_vectors: List[np.ndarray],
    mu: np.ndarray,
    sigma2: np.ndarray,
    V: np.ndarray,
    d_reg: np.ndarray,
) -> Tuple[float, float, List[np.ndarray]]:
    """Genomic prior value and gradient w.r.t. each param vector.

    prior_k = (1/sigma2_k) * (theta_k - mu_k)' G^{-1} (theta_k - mu_k)
            = (1/sigma2_k) * sum_j z_{k,j}^2 / d_reg_j

    d(prior_k)/d(theta_k) = (2/sigma2_k) * G^{-1} (theta_k - mu_k)
                           = (2/sigma2_k) * V @ diag(1/d_reg) @ V' @ (theta_k - mu_k)
    """
    G = V.shape[0]
    inv_d = 1.0 / d_reg
    prior = 0.0
    logdet = 0.0
    grads = []

    for k, theta_k in enumerate(param_vectors):
        centered = theta_k - mu[k]
        z = V.T @ centered
        quadform = np.sum(z**2 * inv_d)
        s2 = max(sigma2[k], 1e-12)
        prior += quadform / s2
        logdet += G * np.log(s2)
        # Gradient: (2/sigma2_k) * V @ diag(1/d_reg) @ z
        grad_k = (2.0 / s2) * (V @ (inv_d * z))
        grads.append(grad_k)

    return prior, logdet, grads


def inner_objective_and_grad(
    x: np.ndarray,
    G: int,
    obs_cells_by_env: Dict[str, List[Tuple[int, int]]],
    precomputed_thermal: Dict[str, np.ndarray],
    precomputed_dl: Dict[str, Optional[np.ndarray]],
    mu: np.ndarray,
    sigma2: np.ndarray,
    V: np.ndarray,
    d_reg: np.ndarray,
    a_fixed: float,
    photo_enabled: bool,
    photo_direction: str = "short_day",
) -> Tuple[float, np.ndarray]:
    """Joint loss and analytic gradient."""
    Theta, S, Pc = unpack_params(x, G, photo_enabled)

    # SSE + gradients
    sse, grad_Theta, grad_S, grad_Pc = _batch_forward_sse_and_grad(
        Theta, S, Pc, G, obs_cells_by_env,
        precomputed_thermal, precomputed_dl, a_fixed, photo_enabled,
        photo_direction=photo_direction,
    )

    # Genomic prior + gradients
    param_vectors = [Theta]
    if photo_enabled:
        param_vectors.extend([S, Pc])
    prior, logdet, prior_grads = _genomic_prior_and_grad(
        param_vectors, mu, sigma2, V, d_reg,
    )

    # Total loss
    total = sse + prior + logdet

    # Assemble gradient
    grad_Theta += prior_grads[0]
    if photo_enabled:
        grad_S += prior_grads[1]
        grad_Pc += prior_grads[2]
        grad = np.concatenate([grad_Theta, grad_S, grad_Pc])
    else:
        grad = grad_Theta

    return total, grad


# ============================================================
# EM Outer Loop (Closed-Form Hyperparameter Updates)
# ============================================================

def update_hyperparams(
    param_vectors: List[np.ndarray],
    V: np.ndarray,
    d_reg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form update of population means and variance components.

    mu_k = mean(theta_k)
    sigma2_k = (1/G) * sum_j z_{k,j}^2 / d_reg_j
    """
    G = V.shape[0]
    K = len(param_vectors)
    mu = np.zeros(K)
    sigma2 = np.zeros(K)
    inv_d = 1.0 / d_reg

    for k, theta_k in enumerate(param_vectors):
        mu[k] = np.mean(theta_k)
        centered = theta_k - mu[k]
        z = V.T @ centered
        sigma2[k] = np.sum(z**2 * inv_d) / G
        # Floor to prevent degenerate zero variance
        sigma2[k] = max(sigma2[k], 1e-6)

    return mu, sigma2


# ============================================================
# Initialization
# ============================================================

def initialize_params(
    genotype_ids: List[str],
    T_dict: dict,
    DL_dict: Optional[dict],
    DAP_obs_dict: dict,
    plantings_of_g_fn: Callable,
    precomputed_thermal: Dict[str, np.ndarray],
    Theta_bounds: Tuple[float, float],
    photo_enabled: bool,
    V: Optional[np.ndarray] = None,
    d: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize Theta_g from observed data + G-matrix smoothing; S_g=0, Pc_g=14.

    Step 1: Compute raw Theta from cumulative thermal rates at observed DAP.
    Step 2: Smooth via GBLUP shrinkage so genomically related genotypes start
            with similar values. This helps L-BFGS-B converge faster.
    """
    G = len(genotype_ids)
    Theta_raw = np.full(G, np.nan)
    S = np.zeros(G)
    Pc = np.full(G, 14.0)

    for i, gid in enumerate(genotype_ids):
        if gid not in DAP_obs_dict:
            continue
        h_vals = []
        for env in plantings_of_g_fn(gid):
            if env not in DAP_obs_dict.get(gid, {}):
                continue
            obs_dap = int(DAP_obs_dict[gid][env])
            thermal = precomputed_thermal.get(env)
            if thermal is None:
                continue
            cut = min(obs_dap, len(thermal))
            if cut > 0:
                h_vals.append(float(np.sum(thermal[:cut])))
        if h_vals:
            h_mean = np.mean(h_vals)
            if h_mean > 0:
                Theta_raw[i] = h_mean

    # Fill missing with population mean
    valid_mask = ~np.isnan(Theta_raw)
    if valid_mask.any():
        pop_mean = np.nanmean(Theta_raw)
        Theta_raw[~valid_mask] = pop_mean
    else:
        Theta_raw[:] = np.mean(Theta_bounds)
        return np.clip(Theta_raw, Theta_bounds[0], Theta_bounds[1]), S, Pc

    # GBLUP shrinkage on raw Theta to smooth across related genotypes
    if V is not None and d is not None and G == V.shape[0]:
        d_reg = np.maximum(d, 1e-6 * np.max(np.abs(d)))
        mu_raw = np.mean(Theta_raw)
        centered = Theta_raw - mu_raw
        z = V.T @ centered
        # Estimate lambda from data variance
        var_theta = np.var(Theta_raw)
        if var_theta > 0:
            # Simple lambda estimate: residual / genetic variance ratio
            lambda_est = max(var_theta * 0.1, 1.0)
        else:
            lambda_est = 1.0
        # Shrinkage: d/(d + lambda)
        shrinkage = d_reg / (d_reg + lambda_est)
        Theta_smoothed = mu_raw + V @ (shrinkage * z)
        Theta = np.clip(Theta_smoothed, Theta_bounds[0], Theta_bounds[1])
    else:
        Theta = np.clip(Theta_raw, Theta_bounds[0], Theta_bounds[1])

    return Theta, S, Pc


# ============================================================
# Prediction for New Genotypes (BLUP)
# ============================================================

def blup_predict_new(
    V: np.ndarray,
    d_reg: np.ndarray,
    fitted_params: List[np.ndarray],
    mu: np.ndarray,
    sigma2: np.ndarray,
    train_indices: List[int],
    new_index: int,
) -> np.ndarray:
    """BLUP prediction for a new genotype not in training set.

    theta_new_k = mu_k + sigma2_k * V[new,:] @ diag(1/d_reg) @ V' @ (theta_k - mu_k*1)
    """
    K = len(fitted_params)
    result = np.zeros(K)
    inv_d = 1.0 / d_reg

    for k in range(K):
        centered = fitted_params[k] - mu[k]
        z = V.T @ centered  # (n,) in eigenspace
        # V[new,:] @ diag(inv_d) @ z
        pred_deviation = np.sum(V[new_index, :] * inv_d * z)
        result[k] = mu[k] + sigma2[k] * pred_deviation

    return result


# ============================================================
# Result Dataclass
# ============================================================

@dataclass
class JointFitResult:
    Theta: np.ndarray
    S: np.ndarray
    Pc: np.ndarray
    mu: np.ndarray
    sigma2: np.ndarray
    alpha_fixed: float
    beta_fixed: float
    a_fixed: float
    Tb: float
    Topt: float
    Tc: float
    genotype_ids: List[str]
    photo_enabled: bool
    converged: bool
    n_em_iterations: int
    final_loss: float
    sse: float
    prior: float
    logdet: float
    history: List[dict] = field(default_factory=list)
    photo_direction: str = "short_day"


# ============================================================
# Main Joint Fit Solver
# ============================================================

def joint_fit(
    T_dict: dict,
    DL_dict: Optional[dict],
    DAP_obs_dict: dict,
    plantings_of_g_fn: Callable,
    V: np.ndarray,
    d: np.ndarray,
    genotype_ids: List[str],
    Tb: float,
    Topt: float,
    Tc: float,
    photo_enabled: bool = True,
    photo_direction: str = "short_day",
    a_fixed: float = 1.0,
    beta_base: float = 1.0,
    Theta_bounds: Tuple[float, float] = (5.0, 200.0),
    S_bounds: Optional[Tuple[float, float]] = (-5.0, 5.0),
    Pc_bounds: Tuple[float, float] = (10.0, 18.0),
    max_em_iters: int = 20,
    em_tol: float = 1e-4,
    lbfgsb_maxiter: int = 500,
    lbfgsb_maxfun: int = 15000,
    verbose: bool = True,
) -> JointFitResult:
    """Joint CGM-WGP fitting via EM / Profile Likelihood."""

    alpha_fixed, beta_fixed = derive_fixed_alpha_beta(Tb, Topt, Tc, beta_base)
    G = len(genotype_ids)

    if verbose:
        print(f"\n=== Joint CGM-WGP Fit ===")
        print(f"  Genotypes: {G}")
        print(f"  Cardinals: Tb={Tb}, Topt={Topt}, Tc={Tc}")
        print(f"  Derived: alpha={alpha_fixed:.4f}, beta={beta_fixed:.4f}")
        print(f"  Photo enabled: {photo_enabled}")
        n_params = 3 * G if photo_enabled else G
        print(f"  Parameters: {n_params} genotypic + 6 hyperparams")

    # Build genotype ID -> index mapping
    gid_to_idx = {gid: i for i, gid in enumerate(genotype_ids)}

    # --- Precompute thermal rates per environment (constant across genotypes) ---
    all_envs = set()
    for gid in genotype_ids:
        if gid in DAP_obs_dict:
            all_envs.update(DAP_obs_dict[gid].keys())
    all_envs = sorted(all_envs)

    precomputed_thermal: Dict[str, np.ndarray] = {}
    precomputed_dl: Dict[str, Optional[np.ndarray]] = {}

    for env in all_envs:
        # Find any genotype that has weather for this env to get the temp series
        temps = None
        dl = None
        for gid in genotype_ids:
            if gid in T_dict and env in T_dict[gid]:
                temps = np.asarray(T_dict[gid][env].values, dtype=float)
                if DL_dict and gid in DL_dict and env in DL_dict[gid]:
                    dl = np.asarray(DL_dict[gid][env], dtype=float)
                break
        if temps is not None:
            precomputed_thermal[env] = cardinal_beta(
                temps, Tb, Topt, Tc, alpha_fixed, beta_fixed
            )
            precomputed_dl[env] = dl

    if verbose:
        print(f"  Environments precomputed: {len(precomputed_thermal)}")

    # --- Build observation cells (grouped by environment for batch processing) ---
    obs_cells_by_env: Dict[str, List[Tuple[int, int]]] = {}
    n_obs = 0
    for gid in genotype_ids:
        g_idx = gid_to_idx[gid]
        if gid not in DAP_obs_dict:
            continue
        for env, dap in DAP_obs_dict[gid].items():
            if env in precomputed_thermal:
                obs_cells_by_env.setdefault(env, []).append((g_idx, int(dap)))
                n_obs += 1

    if verbose:
        print(f"  Observation cells: {n_obs}")

    # --- Regularize eigenvalues ---
    d_reg = np.maximum(d, 1e-6 * np.max(np.abs(d)))

    # --- Initialize genotype parameters ---
    Theta, S_arr, Pc_arr = initialize_params(
        genotype_ids, T_dict, DL_dict, DAP_obs_dict,
        plantings_of_g_fn, precomputed_thermal, Theta_bounds, photo_enabled,
        V=V, d=d,
    )

    if verbose:
        print(f"  Init Theta: mean={np.mean(Theta):.2f}, "
              f"min={np.min(Theta):.2f}, max={np.max(Theta):.2f}")

    # --- Initialize hyperparams ---
    param_vectors = [Theta.copy()]
    if photo_enabled:
        param_vectors.extend([S_arr.copy(), Pc_arr.copy()])
    mu, sigma2 = update_hyperparams(param_vectors, V, d_reg)

    if verbose:
        param_names = ["Theta", "S", "Pc"] if photo_enabled else ["Theta"]
        for k, name in enumerate(param_names):
            print(f"  Init {name}: mu={mu[k]:.4f}, sigma2={sigma2[k]:.6f}")

    # --- Build L-BFGS-B bounds ---
    if S_bounds is None:
        S_bounds = (-5.0, 5.0)
    bounds = build_bounds(G, photo_enabled, Theta_bounds, S_bounds, Pc_bounds)

    # --- EM Loop ---
    history = []
    prev_loss = np.inf
    converged = False

    for em_iter in range(max_em_iters):
        t0 = time.time()

        # --- INNER: L-BFGS-B ---
        x0 = pack_params(Theta, S_arr, Pc_arr, photo_enabled)

        result = minimize(
            inner_objective_and_grad,
            x0,
            args=(
                G, obs_cells_by_env,
                precomputed_thermal, precomputed_dl,
                mu, sigma2, V, d_reg, a_fixed, photo_enabled,
                photo_direction,
            ),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": lbfgsb_maxiter,
                "maxfun": lbfgsb_maxfun,
                "ftol": 1e-10,
                "gtol": 1e-7,
            },
        )

        Theta, S_arr, Pc_arr = unpack_params(result.x, G, photo_enabled)

        # --- OUTER: Closed-form hyperparameter update ---
        param_vectors = [Theta.copy()]
        if photo_enabled:
            param_vectors.extend([S_arr.copy(), Pc_arr.copy()])
        mu, sigma2 = update_hyperparams(param_vectors, V, d_reg)

        # --- Compute loss components ---
        sse_val, _, _, _ = _batch_forward_sse_and_grad(
            Theta, S_arr, Pc_arr, G, obs_cells_by_env,
            precomputed_thermal, precomputed_dl, a_fixed, photo_enabled,
            photo_direction=photo_direction,
        )
        n_valid = n_obs

        prior_val, logdet_val = compute_genomic_prior(
            param_vectors, mu, sigma2, V, d_reg
        )
        total_loss = sse_val + prior_val + logdet_val

        elapsed = time.time() - t0

        iter_info = {
            "em_iter": em_iter + 1,
            "total_loss": total_loss,
            "sse": sse_val,
            "prior": prior_val,
            "logdet": logdet_val,
            "lbfgsb_nit": result.nit,
            "lbfgsb_nfev": result.nfev,
            "elapsed_s": elapsed,
        }
        param_names = ["Theta", "S", "Pc"] if photo_enabled else ["Theta"]
        for k, name in enumerate(param_names):
            iter_info[f"mu_{name}"] = float(mu[k])
            iter_info[f"sigma2_{name}"] = float(sigma2[k])

        history.append(iter_info)

        if verbose:
            rmse = np.sqrt(sse_val / max(n_valid, 1))
            print(f"  EM iter {em_iter + 1}: loss={total_loss:.2f} "
                  f"(SSE={sse_val:.1f}, prior={prior_val:.1f}, logdet={logdet_val:.1f}) "
                  f"RMSE={rmse:.2f}d, L-BFGS-B nit={result.nit}, "
                  f"time={elapsed:.1f}s")

        # --- Check convergence ---
        if prev_loss < np.inf:
            rel_change = abs(total_loss - prev_loss) / max(abs(prev_loss), 1.0)
            if rel_change < em_tol:
                if verbose:
                    print(f"  Converged at EM iter {em_iter + 1} "
                          f"(rel_change={rel_change:.2e} < {em_tol})")
                converged = True
                break
        prev_loss = total_loss

    if verbose and not converged:
        print(f"  EM did not converge after {max_em_iters} iterations")

    return JointFitResult(
        Theta=Theta,
        S=S_arr,
        Pc=Pc_arr,
        mu=mu,
        sigma2=sigma2,
        alpha_fixed=alpha_fixed,
        beta_fixed=beta_fixed,
        a_fixed=a_fixed,
        Tb=Tb,
        Topt=Topt,
        Tc=Tc,
        genotype_ids=genotype_ids,
        photo_enabled=photo_enabled,
        photo_direction=photo_direction,
        converged=converged,
        n_em_iterations=len(history),
        final_loss=history[-1]["total_loss"] if history else np.inf,
        sse=history[-1]["sse"] if history else np.inf,
        prior=history[-1]["prior"] if history else 0.0,
        logdet=history[-1]["logdet"] if history else 0.0,
        history=history,
    )


# ============================================================
# Prediction
# ============================================================

def predict_all(
    result: JointFitResult,
    T_dict: dict,
    DL_dict: Optional[dict],
    predict_plantings: List[str],
    DAP_obs_dict: Optional[dict] = None,
    test_obs: Optional[Dict[Tuple[str, str], int]] = None,
) -> List[dict]:
    """Generate predictions for all genotypes at prediction plantings."""

    # Precompute thermal rates for prediction environments
    pred_thermal: Dict[str, np.ndarray] = {}
    pred_dl: Dict[str, Optional[np.ndarray]] = {}

    for env in predict_plantings:
        temps = None
        dl = None
        for gid in result.genotype_ids:
            if gid in T_dict and env in T_dict[gid]:
                temps = np.asarray(T_dict[gid][env].values, dtype=float)
                if DL_dict and gid in DL_dict and env in DL_dict[gid]:
                    dl = np.asarray(DL_dict[gid][env], dtype=float)
                break
        if temps is not None:
            pred_thermal[env] = cardinal_beta(
                temps, result.Tb, result.Topt, result.Tc,
                result.alpha_fixed, result.beta_fixed,
            )
            pred_dl[env] = dl

    rows = []
    for i, gid in enumerate(result.genotype_ids):
        for env in predict_plantings:
            thermal = pred_thermal.get(env)
            if thermal is None:
                continue

            # Check if this genotype has weather for this env
            if gid not in T_dict or env not in T_dict[gid]:
                continue

            # Use genotype-specific weather if available
            temps_g = np.asarray(T_dict[gid][env].values, dtype=float)
            thermal_g = cardinal_beta(
                temps_g, result.Tb, result.Topt, result.Tc,
                result.alpha_fixed, result.beta_fixed,
            )
            dl_g = None
            if DL_dict and gid in DL_dict and env in DL_dict[gid]:
                dl_g = np.asarray(DL_dict[gid][env], dtype=float)

            pred = forward_dap_fractional(
                result.Theta[i], result.S[i], result.Pc[i],
                thermal_g, dl_g, result.a_fixed, result.photo_enabled,
                photo_direction=result.photo_direction,
            )
            pred_dap = round(pred) if pred > 0 else "NA"

            # Get observed DAP — only from test_obs if provided (prevents leakage)
            obs_dap = None
            if test_obs is not None:
                obs_dap = test_obs.get((gid, env))
            elif DAP_obs_dict and gid in DAP_obs_dict and env in DAP_obs_dict[gid]:
                obs_dap = int(DAP_obs_dict[gid][env])

            error = "NA"
            if obs_dap is not None and pred_dap != "NA":
                error = f"{pred_dap - obs_dap:.1f}"

            rows.append({
                "id": gid,
                "planting": env,
                "observed_dap": obs_dap if obs_dap is not None else "NA",
                "predicted_dap": pred_dap,
                "error": error,
            })

    return rows


def predict_new_genotypes(
    result: JointFitResult,
    V_full: np.ndarray,
    d: np.ndarray,
    all_gmatrix_ids: List[str],
    T_dict: dict,
    DL_dict: Optional[dict],
    predict_plantings: List[str],
    DAP_obs_dict: Optional[dict] = None,
    test_obs: Optional[Dict[Tuple[str, str], int]] = None,
) -> List[dict]:
    """BLUP-predict Theta for new genotypes using G-matrix, use population S/Pc.

    Only Theta_g is BLUP-predicted per genotype. S and Pc use the population
    mean (mu_S, mu_Pc) because photoperiod parameters have low heritability
    and extrapolate poorly, especially with limited environmental variation.

    Uses the proper BLUP formula:
      Theta_new = mu_Theta + G[new, train] @ G[train,train]^{-1} @ (Theta_train - mu_Theta)
    Implemented via spectral decomposition to avoid explicit inversion.
    """

    d_reg = np.maximum(d, 1e-6 * np.max(np.abs(d)))
    train_set = set(result.genotype_ids)
    gid_to_gmat_idx = {gid: i for i, gid in enumerate(all_gmatrix_ids)}

    # Map training genotypes to their G-matrix indices
    train_gmat_indices = []
    train_Theta = []
    for i, gid in enumerate(result.genotype_ids):
        for gmat_gid in all_gmatrix_ids:
            if gmat_gid.upper() == gid.upper():
                train_gmat_indices.append(gid_to_gmat_idx[gmat_gid])
                train_Theta.append(result.Theta[i])
                break

    train_gmat_indices = np.array(train_gmat_indices)
    train_Theta = np.array(train_Theta)
    mu_Theta = result.mu[0]

    # Population-mean photoperiod params (NOT BLUP-predicted)
    mu_S = result.mu[1] if result.photo_enabled and len(result.mu) > 1 else 0.0
    mu_Pc = result.mu[2] if result.photo_enabled and len(result.mu) > 2 else 14.0

    # Reconstruct G from full EVD: G = V @ diag(d) @ V'
    # For BLUP: Theta_new = mu + G[new, train] @ G[train,train]^{-1} @ (Theta_train - mu)
    # Use the spectral form for numerical stability
    G_matrix = (V_full * d[None, :]) @ V_full.T

    # G[train, train] and its inverse (regularized)
    G_tt = G_matrix[np.ix_(train_gmat_indices, train_gmat_indices)]
    G_tt += np.eye(len(train_gmat_indices)) * 1e-4 * np.mean(np.diag(G_tt))
    try:
        G_tt_inv = np.linalg.inv(G_tt)
    except np.linalg.LinAlgError:
        G_tt_inv = np.linalg.pinv(G_tt)

    # Precompute: G_tt_inv @ (Theta_train - mu)
    centered_train = train_Theta - mu_Theta
    blup_coeffs = G_tt_inv @ centered_train  # (n_train,)

    rows = []
    for gmat_gid in all_gmatrix_ids:
        if gmat_gid.upper() in {g.upper() for g in train_set}:
            continue  # skip training genotypes

        j = gid_to_gmat_idx[gmat_gid]

        # G[new, train] @ blup_coeffs
        g_new_train = G_matrix[j, train_gmat_indices]
        Theta_new = mu_Theta + float(g_new_train @ blup_coeffs)
        Theta_new = np.clip(Theta_new, 5.0, 200.0)

        # Use population-mean photoperiod
        S_new = mu_S
        Pc_new = mu_Pc

        for env in predict_plantings:
            if gmat_gid not in T_dict or env not in T_dict[gmat_gid]:
                continue
            temps = np.asarray(T_dict[gmat_gid][env].values, dtype=float)
            thermal = cardinal_beta(
                temps, result.Tb, result.Topt, result.Tc,
                result.alpha_fixed, result.beta_fixed,
            )
            dl = None
            if DL_dict and gmat_gid in DL_dict and env in DL_dict[gmat_gid]:
                dl = np.asarray(DL_dict[gmat_gid][env], dtype=float)

            pred = forward_dap_fractional(
                Theta_new, S_new, Pc_new,
                thermal, dl, result.a_fixed, result.photo_enabled,
                photo_direction=result.photo_direction,
            )
            pred_dap = round(pred) if pred > 0 else "NA"

            obs_dap = None
            if test_obs is not None:
                obs_dap = test_obs.get((gmat_gid, env))
            elif DAP_obs_dict and gmat_gid in DAP_obs_dict and env in DAP_obs_dict[gmat_gid]:
                obs_dap = int(DAP_obs_dict[gmat_gid][env])

            error = "NA"
            if obs_dap is not None and pred_dap != "NA":
                error = f"{pred_dap - obs_dap:.1f}"

            rows.append({
                "id": gmat_gid,
                "planting": env,
                "observed_dap": obs_dap if obs_dap is not None else "NA",
                "predicted_dap": pred_dap,
                "error": error,
                "method": "blup_new_genotype",
            })

    return rows


# ============================================================
# Output Writers
# ============================================================

def write_fitted_csv(result: JointFitResult, path: str):
    fields = ["id", "Theta", "S", "Pc", "alpha_fixed", "beta_fixed"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, gid in enumerate(result.genotype_ids):
            writer.writerow({
                "id": gid,
                "Theta": f"{result.Theta[i]:.6f}",
                "S": f"{result.S[i]:.6f}",
                "Pc": f"{result.Pc[i]:.4f}",
                "alpha_fixed": f"{result.alpha_fixed:.6f}",
                "beta_fixed": f"{result.beta_fixed:.6f}",
            })
    print(f"  Wrote {len(result.genotype_ids)} rows to {path}")


def write_predictions_csv(rows: List[dict], path: str):
    if not rows:
        return
    fields = ["id", "planting", "observed_dap", "predicted_dap", "error"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} prediction rows to {path}")


def write_convergence_csv(history: List[dict], path: str):
    if not history:
        return
    fields = list(history[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v
                             for k, v in row.items()})


def write_validation_comparison(
    pred_rows: List[dict], method_name: str, path: str,
    train_str: str = "", predict_str: str = "",
):
    """Write validation_comparison.csv compatible with existing pipeline."""
    valid = [r for r in pred_rows
             if r["observed_dap"] != "NA" and r["predicted_dap"] != "NA"]
    if not valid:
        return

    errors = [float(r["predicted_dap"]) - float(r["observed_dap"]) for r in valid]
    abs_errors = [abs(e) for e in errors]
    n = len(errors)

    obs = [float(r["observed_dap"]) for r in valid]
    pred = [float(r["predicted_dap"]) for r in valid]

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    bias = np.mean(errors)
    median_err = np.median(abs_errors)
    within_3d = sum(1 for e in abs_errors if e <= 3) / n * 100
    within_5d = sum(1 for e in abs_errors if e <= 5) / n * 100
    within_7d = sum(1 for e in abs_errors if e <= 7) / n * 100
    within_10d = sum(1 for e in abs_errors if e <= 10) / n * 100

    # Pearson correlation
    pa = float(np.corrcoef(obs, pred)[0, 1]) if n > 2 else float("nan")

    row = {
        "method": method_name,
        "train_plantings": train_str,
        "predict_plantings": predict_str,
        "n_evaluated": n,
        "mae": f"{mae:.2f}",
        "rmse": f"{rmse:.2f}",
        "bias": f"{bias:.2f}",
        "median_error": f"{median_err:.2f}",
        "within_3d": f"{within_3d:.1f}",
        "within_5d": f"{within_5d:.1f}",
        "within_7d": f"{within_7d:.1f}",
        "within_10d": f"{within_10d:.1f}",
        "pearson_r": f"{pa:.4f}",
        "description": "Joint CGM-WGP (Messina/Technow EM + L-BFGS-B)",
    }

    fields = list(row.keys())
    write_header = not Path(path).exists()
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Joint CGM-WGP fitting (Messina/Technow framework)"
    )
    parser.add_argument("--phenotypes", required=True)
    parser.add_argument("--weather", required=True)
    parser.add_argument("--evd-path", required=True)
    parser.add_argument("--Tb", type=float, required=True)
    parser.add_argument("--Topt", type=float, required=True)
    parser.add_argument("--Tc", type=float, required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--crop", default="unknown")
    parser.add_argument("--train-plantings", default=None)
    parser.add_argument("--predict-plantings", default=None)
    parser.add_argument("--test-phenotypes", default=None)
    parser.add_argument("--latitude-map", default=None)
    parser.add_argument("--photo-enabled", action="store_true", default=False)
    parser.add_argument("--a-fixed", type=float, default=1.0)
    parser.add_argument("--S-bounds", type=float, nargs=2, default=[-5.0, 5.0])
    parser.add_argument("--Pc-bounds", type=float, nargs=2, default=[10.0, 18.0])
    parser.add_argument("--Theta-bounds", type=float, nargs=2, default=[5.0, 200.0])
    parser.add_argument("--max-em-iters", type=int, default=50)
    parser.add_argument("--em-tol", type=float, default=1e-4)
    parser.add_argument("--lbfgsb-maxiter", type=int, default=500)
    parser.add_argument("--lbfgsb-maxfun", type=int, default=15000)
    parser.add_argument("--exclude-genotypes", default=None)
    parser.add_argument("--run-label", default=None)

    args = parser.parse_args()

    # --- Parse latitude map ---
    latitude_map = None
    if args.latitude_map:
        latitude_map = json.loads(args.latitude_map)

    # --- Output directory ---
    if args.run_dir:
        run_dir = Path(args.run_dir)
        for sub in ("params", "predictions", "diagnostics"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
    else:
        run_dir = Path(".")

    # --- Load data ---
    print(f"\nLoading data...")
    pheno_obs, T_dict, DAP_obs_dict, plantings_of_g_fn, DL_dict = load_and_build_dicts(
        args.phenotypes, args.weather, latitude_map=latitude_map,
    )

    # --- Load test phenotypes if provided ---
    test_obs: Optional[Dict[Tuple[str, str], int]] = None
    if args.test_phenotypes:
        test_df = pd.read_csv(args.test_phenotypes)
        _gcol = "id" if "id" in test_df.columns else "Genotype"
        _pcol = "Planting" if "Planting" in test_df.columns else "planting"
        _ftcol = "ft" if "ft" in test_df.columns else "FT"
        test_obs = {}
        for _, row in test_df.iterrows():
            gid = str(row[_gcol]).strip()
            env = str(row[_pcol]).strip()
            ft = row.get(_ftcol)
            if pd.notna(ft):
                test_obs[(gid, env)] = int(round(float(ft)))
        print(f"  Loaded {len(test_obs)} test observation cells")

        # Augment T_dict/DL_dict with test env weather
        for (gid, env), _ in test_obs.items():
            if gid not in T_dict:
                # Find weather from any genotype at this env
                for other_gid in T_dict:
                    if env in T_dict[other_gid]:
                        T_dict.setdefault(gid, {})[env] = T_dict[other_gid][env]
                        if DL_dict and other_gid in DL_dict and env in DL_dict[other_gid]:
                            DL_dict.setdefault(gid, {})[env] = DL_dict[other_gid][env]
                        break
            elif env not in T_dict[gid]:
                for other_gid in T_dict:
                    if env in T_dict[other_gid]:
                        T_dict[gid][env] = T_dict[other_gid][env]
                        if DL_dict and other_gid in DL_dict and env in DL_dict[other_gid]:
                            DL_dict.setdefault(gid, {})[env] = DL_dict[other_gid][env]
                        break

    # --- Load EVD ---
    print(f"Loading EVD from {args.evd_path}...")
    V, d, gmatrix_ids = load_evd(args.evd_path)
    print(f"  G-matrix: {len(gmatrix_ids)} genotypes, {V.shape[1]} eigenvectors")

    # --- Build genotype list (intersection of phenotypes and G-matrix) ---
    gmatrix_upper = {gid.upper(): gid for gid in gmatrix_ids}
    all_pheno_gids = sorted(DAP_obs_dict.keys())

    # Exclude genotypes
    exclude = set()
    if args.exclude_genotypes:
        exclude = {g.strip().upper() for g in args.exclude_genotypes.split(",")}

    genotype_ids = []
    gmat_indices = []
    for gid in all_pheno_gids:
        if gid.upper() in exclude:
            continue
        if gid.upper() in gmatrix_upper:
            genotype_ids.append(gid)
            gmat_idx = list(gmatrix_ids).index(gmatrix_upper[gid.upper()])
            gmat_indices.append(gmat_idx)

    if not genotype_ids:
        print("ERROR: No genotypes in common between phenotypes and G-matrix.")
        sys.exit(1)

    # Subset V and d to training genotypes
    V_sub = V[np.ix_(gmat_indices, range(V.shape[1]))]
    # Re-orthogonalize the subset (extract submatrix rows from eigenvectors)
    # For the genomic prior we need the full V but indexed by our genotype ordering

    print(f"  Training genotypes: {len(genotype_ids)}")

    # --- Resolve train/predict plantings ---
    train_plantings = None
    predict_plantings = None
    if args.train_plantings:
        train_plantings = [p.strip() for p in args.train_plantings.split(",")]
    if args.predict_plantings:
        predict_plantings = [p.strip() for p in args.predict_plantings.split(",")]

    # Augment T_dict/DL_dict: ensure all genotypes have weather for prediction envs
    if predict_plantings:
        for pred_env in predict_plantings:
            ref_temps = None
            ref_dl = None
            for gid in T_dict:
                if pred_env in T_dict[gid]:
                    ref_temps = T_dict[gid][pred_env]
                    if DL_dict and gid in DL_dict and pred_env in DL_dict[gid]:
                        ref_dl = DL_dict[gid][pred_env]
                    break
            if ref_temps is not None:
                for gid in genotype_ids:
                    if gid not in T_dict:
                        T_dict[gid] = {}
                    if pred_env not in T_dict[gid]:
                        T_dict[gid][pred_env] = ref_temps
                    if ref_dl is not None and DL_dict is not None:
                        if gid not in DL_dict:
                            DL_dict[gid] = {}
                        if pred_env not in DL_dict[gid]:
                            DL_dict[gid][pred_env] = ref_dl

    # Filter plantings_of_g_fn if train_plantings specified
    if train_plantings:
        _orig_fn = plantings_of_g_fn
        def _filtered_fn(gid):
            return [p for p in _orig_fn(gid) if p in train_plantings]
        plantings_of_g_fn = _filtered_fn

        # Filter DAP_obs_dict to training plantings
        for gid in list(DAP_obs_dict.keys()):
            DAP_obs_dict[gid] = {
                e: v for e, v in DAP_obs_dict[gid].items()
                if e in train_plantings
            }

    # --- Run joint fit ---
    fit_result = joint_fit(
        T_dict=T_dict,
        DL_dict=DL_dict,
        DAP_obs_dict=DAP_obs_dict,
        plantings_of_g_fn=plantings_of_g_fn,
        V=V_sub,
        d=d,
        genotype_ids=genotype_ids,
        Tb=args.Tb,
        Topt=args.Topt,
        Tc=args.Tc,
        photo_enabled=args.photo_enabled,
        a_fixed=args.a_fixed,
        S_bounds=tuple(args.S_bounds),
        Pc_bounds=tuple(args.Pc_bounds),
        Theta_bounds=tuple(args.Theta_bounds),
        max_em_iters=args.max_em_iters,
        em_tol=args.em_tol,
        lbfgsb_maxiter=args.lbfgsb_maxiter,
        lbfgsb_maxfun=args.lbfgsb_maxfun,
    )

    # --- Write outputs ---
    fitted_path = str(run_dir / "params" / "fitted.csv")
    write_fitted_csv(fit_result, fitted_path)

    conv_path = str(run_dir / "diagnostics" / "joint_convergence.csv")
    write_convergence_csv(fit_result.history, conv_path)

    # Write run config
    config_out = {
        "method": "joint",
        "crop": args.crop,
        "Tb": args.Tb, "Topt": args.Topt, "Tc": args.Tc,
        "alpha_fixed": fit_result.alpha_fixed,
        "beta_fixed": fit_result.beta_fixed,
        "a_fixed": args.a_fixed,
        "photo_enabled": args.photo_enabled,
        "n_genotypes": len(genotype_ids),
        "n_em_iterations": fit_result.n_em_iterations,
        "converged": fit_result.converged,
        "final_loss": fit_result.final_loss,
        "final_sse": fit_result.sse,
        "mu": fit_result.mu.tolist(),
        "sigma2": fit_result.sigma2.tolist(),
    }
    with open(run_dir / "diagnostics" / "run_config.json", "w") as fh:
        json.dump(config_out, fh, indent=2)

    # --- Predictions on held-out plantings ---
    if predict_plantings:
        print(f"\nPredicting on {len(predict_plantings)} plantings...")
        pred_rows = predict_all(
            fit_result, T_dict, DL_dict, predict_plantings,
            DAP_obs_dict, test_obs,
        )

        # Also predict for new genotypes via BLUP
        new_geno_rows = predict_new_genotypes(
            fit_result, V, d, gmatrix_ids,
            T_dict, DL_dict, predict_plantings,
            DAP_obs_dict, test_obs,
        )
        all_pred_rows = pred_rows + new_geno_rows

        pred_path = str(run_dir / "predictions" / "mechanistic.csv")
        write_predictions_csv(all_pred_rows, pred_path)

        # Write validation comparison
        train_str = args.train_plantings or "all"
        predict_str = args.predict_plantings or "all"
        val_path = str(run_dir / "diagnostics" / "validation_comparison.csv")
        write_validation_comparison(
            all_pred_rows, "Joint_CGM_WGP", val_path, train_str, predict_str,
        )

        # Print summary
        valid = [r for r in all_pred_rows
                 if r["observed_dap"] != "NA" and r["predicted_dap"] != "NA"]
        if valid:
            errors = [float(r["predicted_dap"]) - float(r["observed_dap"])
                      for r in valid]
            obs = [float(r["observed_dap"]) for r in valid]
            pred = [float(r["predicted_dap"]) for r in valid]
            rmse = np.sqrt(np.mean([e**2 for e in errors]))
            mae = np.mean([abs(e) for e in errors])
            pa = float(np.corrcoef(obs, pred)[0, 1]) if len(valid) > 2 else float("nan")
            print(f"\n  Results: n={len(valid)}, RMSE={rmse:.2f}, "
                  f"MAE={mae:.2f}, PA(r)={pa:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
