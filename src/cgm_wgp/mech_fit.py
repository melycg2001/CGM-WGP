"""
cgm_mech_fit.py

Mechanistic fitting of (alpha, beta) parameters for a single genotype using:
- Cardinal beta temperature response
- Mechanistic "equal cumulative progress at flowering" loss
- SciPy dual_annealing (with Nelder-Mead local search) to minimize the loss
- Optional GBLUP penalty for iterative coupling (quadratic pull toward genomic targets)
- Top-K dual annealing: records search trajectory, keeps best K for uncertainty
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import dual_annealing


# ============================================================
# Daylength computation (Spencer 1971)
# ============================================================
def compute_daylength(doy: int, latitude: float) -> float:
    """Compute astronomical daylength (hours) from day-of-year and latitude.

    Uses the Spencer (1971) approximation for solar declination.
    """
    lat_rad = math.radians(latitude)
    # Solar declination (radians)
    decl = 0.4093 * math.sin(2.0 * math.pi * (284 + doy) / 365.0)
    # Hour angle at sunrise/sunset
    cos_ha = -math.tan(lat_rad) * math.tan(decl)
    cos_ha = max(-1.0, min(1.0, cos_ha))  # clamp for polar regions
    ha = math.acos(cos_ha)
    return (2.0 * ha / math.pi) * 12.0


def compute_daylength_series(
    doy_series: Sequence[int],
    latitude: float,
) -> np.ndarray:
    """Compute daylength for a sequence of DOY values. Returns array of hours."""
    return np.array([compute_daylength(int(d), latitude) for d in doy_series])


# ============================================================
# Sinclair (1991) photoperiod response
# ============================================================
def sinclair_photoperiod(
    daylength: float,
    B: Optional[float] = None,
    Pc: Optional[float] = None,
    photo_type: str = "short_day",
) -> float:
    """Sinclair (1991) photoperiod response.

    Returns a value in [0, 1].  When B or Pc is None, returns 1.0 (no effect).

    Short-day (default): g(P) = 1 - exp[B*(P - Pc)] when P <= Pc, else 0.
        Development is full at short daylengths, zero when P > Pc.
    Long-day:  g(P) = exp[B*(P - Pc)] when P < Pc, else 1.
        Development is zero at short daylengths, full when P >= Pc.
    """
    if B is None or Pc is None:
        return 1.0
    if photo_type == "long_day":
        if daylength >= Pc:
            return 1.0
        val = math.exp(B * (daylength - Pc))
        return max(0.0, min(1.0, val))
    else:  # short_day
        if daylength > Pc:
            return 0.0
        val = 1.0 - math.exp(B * (daylength - Pc))
        return max(0.0, min(1.0, val))


def sinclair_photoperiod_array(
    daylengths: Sequence[float],
    B: Optional[float] = None,
    Pc: Optional[float] = None,
    photo_type: str = "short_day",
) -> np.ndarray:
    """Vectorized Sinclair photoperiod for an array of daylengths."""
    if B is None or Pc is None:
        return np.ones(len(daylengths), dtype=float)
    dl = np.asarray(daylengths, dtype=float)
    if photo_type == "long_day":
        result = np.where(dl >= Pc, 1.0, np.exp(B * (dl - Pc)))
    else:  # short_day
        result = np.where(dl > Pc, 0.0, 1.0 - np.exp(B * (dl - Pc)))
    return np.clip(result, 0.0, 1.0)


# ============================================================
# Logistic (smooth) photoperiod response — no hard cutoff
# ============================================================
def logistic_photoperiod_array(
    daylengths: Sequence[float],
    B: Optional[float] = None,
    Pc: Optional[float] = None,
    photo_type: str = "short_day",
) -> np.ndarray:
    """Smooth logistic photoperiod response (no hard cutoff).

    Short-day: g(P) = 1 / (1 + exp(B * (P - Pc)))
    Long-day:  g(P) = 1 / (1 + exp(-B * (P - Pc)))

    Same B, Pc params as Sinclair. B controls steepness, Pc is inflection point.
    Never exactly 0 or 1 — development is always partially possible.
    """
    if B is None or Pc is None:
        return np.ones(len(daylengths), dtype=float)
    dl = np.asarray(daylengths, dtype=float)
    if photo_type == "long_day":
        return 1.0 / (1.0 + np.exp(-B * (dl - Pc)))
    else:  # short_day
        return 1.0 / (1.0 + np.exp(B * (dl - Pc)))


# ============================================================
# 3-parameter logistic photoperiod (Charlie's formulation)
# ============================================================
def logistic3_photoperiod_array(
    daylengths: Sequence[float],
    a_photo: Optional[float] = None,
    b_photo: Optional[float] = None,
    d_photo: Optional[float] = None,
) -> np.ndarray:
    """3-parameter logistic photoperiod response (Messina formulation).

    g(P) = 1 / (1 + a * exp(b * (P - d)))

    Parameters:
        a_photo: amplitude (controls floor — min response = 1/(1+a))
        b_photo: slope (negative = short-day, positive = long-day)
        d_photo: inflection point (daylength at half-max response)

    Key property: never reaches 0. Floor = 1/(1+a).
    With a=0.8: floor = 0.56. With a=2.0: floor = 0.33.

    When any param is None, returns 1.0 (no photoperiod effect).
    """
    if a_photo is None or b_photo is None or d_photo is None:
        return np.ones(len(daylengths), dtype=float)
    dl = np.asarray(daylengths, dtype=float)
    return 1.0 / (1.0 + a_photo * np.exp(b_photo * (dl - d_photo)))


# ============================================================
# Linear-plateau photoperiod (Grimm et al. 1993)
# ============================================================
def linear_plateau_night_array(
    daylengths: Sequence[float],
    n_min: Optional[float] = None,
    n_opt: Optional[float] = None,
) -> np.ndarray:
    """Linear-plateau photoperiod response (Grimm et al. 1993).

    Operates on NIGHT length (= 24 - daylength). For short-day plants like
    soybean and beans, long nights = full development, short nights = suppressed.

    F(N) = 0                              if N <= n_min
    F(N) = (N - n_min) / (n_opt - n_min)  if n_min < N < n_opt
    F(N) = 1                              if N >= n_opt

    Parameters:
        daylengths: array of daylengths in hours (will be converted to night length)
        n_min: minimum night length (below which no development toward flowering)
        n_opt: optimal night length (at or above which full development)

    Returns F(N) in [0, 1]. When n_min or n_opt is None, returns 1.0 (no effect).
    """
    if n_min is None or n_opt is None:
        return np.ones(len(daylengths), dtype=float)
    if n_opt <= n_min:
        return np.ones(len(daylengths), dtype=float)
    dl = np.asarray(daylengths, dtype=float)
    nl = 24.0 - dl  # convert to night length
    result = np.clip((nl - n_min) / (n_opt - n_min), 0.0, 1.0)
    return result


# ============================================================
# Pure photoperiod flowering prediction (diagnostic mode)
#
# These functions implement a flowering model that uses ONLY photoperiod
# (no temperature anywhere). Intended as a research/diagnostic tool to
# isolate photo-fitting problems from thermal-fitting problems and to
# test the hypothesis that short-day crops can be predicted from
# photoperiod alone.
#
# Model form (analogous to the thermal model):
#   cum_photo_g(t) = sum_{i=1..t} F_photo(DL_i)
#   DAP_g = min t such that cum_photo_g(t) >= H_g
#
# where F_photo is linear_plateau / logistic3 / sinclair and H_g is a
# per-genotype "photoperiod time to flower" threshold, fitted by SSE
# minimization across training (g, env) cells.
# ============================================================
def photo_rates_from_model(
    daylengths: Sequence[float],
    photo_model: str,
    params: Dict[str, float],
    photo_type: str = "short_day",
) -> np.ndarray:
    """Unified dispatcher: daylengths → daily F_photo rates in [0, 1].

    Reuses the existing per-model photo array functions so there's no
    duplication. The `params` dict must contain the keys appropriate to
    the selected `photo_model`:

      - "linear_plateau": {"n_min": float, "n_opt": float}
      - "logistic3":      {"a": float, "b": float, "d": float}
      - "sinclair":       {"B": float, "Pc": float}
      - "logistic":       {"B": float, "Pc": float}  (smooth variant)

    Returns an array of daily photoperiod rates in [0, 1] with the same
    length as `daylengths`.
    """
    if photo_model == "linear_plateau":
        return linear_plateau_night_array(
            daylengths,
            n_min=params.get("n_min"),
            n_opt=params.get("n_opt"),
        )
    elif photo_model == "logistic3":
        return logistic3_photoperiod_array(
            daylengths,
            a_photo=params.get("a"),
            b_photo=params.get("b"),
            d_photo=params.get("d"),
        )
    elif photo_model == "sinclair":
        return sinclair_photoperiod_array(
            daylengths,
            B=params.get("B"),
            Pc=params.get("Pc"),
            photo_type=photo_type,
        )
    elif photo_model == "logistic":
        return logistic_photoperiod_array(
            daylengths,
            B=params.get("B"),
            Pc=params.get("Pc"),
            photo_type=photo_type,
        )
    else:
        raise ValueError(
            f"Unknown photo_model {photo_model!r}; expected one of "
            "'linear_plateau', 'logistic3', 'sinclair', 'logistic'"
        )


def predict_dap_photo_only(
    daylengths: Sequence[float],
    photo_model: str,
    params: Dict[str, float],
    h_threshold: float,
    photo_type: str = "short_day",
) -> int:
    """Pure photoperiod flowering DAP prediction (no temperature).

    Integrates F_photo(DL_t) day-by-day until the cumulative sum reaches
    `h_threshold`. Returns the 1-based day index when the threshold is
    first reached, or -1 if the threshold is never reached within the
    weather window (caller can decide how to handle: skip, penalize, or
    extrapolate).

    This is deliberately the photoperiod analog of the thermal-only
    `predict_flowering_dap`: there is NO thermal component anywhere.
    Used by the `--photo-only` diagnostic mode.
    """
    rates = photo_rates_from_model(daylengths, photo_model, params, photo_type)
    if rates.size == 0:
        return -1
    cum = np.cumsum(rates)
    idx = int(np.searchsorted(cum, h_threshold, side="left"))
    if idx >= len(rates):
        return -1
    return idx + 1  # 1-based DAP


def _fit_h_per_genotype_given_shape(
    genotypes: List[str],
    plantings_of_g_fn: Callable[[str], Sequence[str]],
    DL_dict: Dict[str, Dict[str, np.ndarray]],
    DAP_obs_dict: Dict[str, Dict[str, int]],
    photo_model: str,
    shape_params: Dict[str, float],
    photo_type: str = "short_day",
) -> Dict[str, float]:
    """Closed-form per-genotype H_g fit given a fixed photo shape.

    For each genotype g, compute the cumulative photo progress at each
    training env's observed DAP. The SSE-minimizing H_g is the mean of
    those cum[obs_dap-1] values across the genotype's training envs.

    Rationale: predict_dap_photo_only() is monotonic in H (the larger H
    is, the later flowering occurs), so the best-fit H for a single
    genotype across multiple envs is the mean of the "cum_photo needed
    to flower at the observed day" values.
    """
    h_per_genotype: Dict[str, float] = {}
    for g in genotypes:
        if g not in DL_dict:
            continue
        cum_at_obs: List[float] = []
        for e in plantings_of_g_fn(g):
            if e not in DL_dict[g] or e not in DAP_obs_dict.get(g, {}):
                continue
            dl = np.asarray(DL_dict[g][e], dtype=float)
            if dl.size == 0:
                continue
            rates = photo_rates_from_model(dl, photo_model, shape_params, photo_type)
            obs_dap = int(DAP_obs_dict[g][e])
            idx = max(0, min(obs_dap - 1, len(rates) - 1))
            cum_val = float(np.sum(rates[: idx + 1]))
            if cum_val > 0:
                cum_at_obs.append(cum_val)
        if cum_at_obs:
            h_per_genotype[g] = float(np.mean(cum_at_obs))
    return h_per_genotype


def fit_photo_only(
    genotypes: List[str],
    plantings_of_g_fn: Callable[[str], Sequence[str]],
    DL_dict: Dict[str, Dict[str, np.ndarray]],
    DAP_obs_dict: Dict[str, Dict[str, int]],
    photo_model: str,
    shape_bounds: List[Tuple[float, float]],
    shape_param_names: List[str],
    photo_type: str = "short_day",
    maxiter: int = 500,
    seed: Optional[int] = 42,
) -> Dict:
    """Fit photo-only model (shape params population-level + per-genotype H).

    Two-stage optimization:
      1. Outer: dual_annealing over photo shape params (population level)
      2. Inner: closed-form per-genotype H given current shape

    Loss: mean SSE between predicted and observed DAP across all training
    (g, env) cells. When pred_dap == -1 (photo never reaches threshold
    within the weather window), a penalty of (weather_window - obs_dap)^2
    is added.

    Args:
        genotypes: list of genotype IDs to fit
        plantings_of_g_fn: function gid -> list of training plantings
        DL_dict: {gid -> {env -> daylength array}}
        DAP_obs_dict: {gid -> {env -> observed DAP}}
        photo_model: "linear_plateau", "logistic3", or "sinclair"
        shape_bounds: list of (lo, hi) tuples, one per shape parameter
        shape_param_names: list of parameter names, same order as shape_bounds
        photo_type: "short_day" or "long_day" (sinclair only)
        maxiter: dual_annealing max iterations
        seed: RNG seed

    Returns dict with:
        photo_model, shape_params, h_per_genotype, loss, n_cells_fit,
        per_env_sse, optimizer_result
    """
    from scipy.optimize import dual_annealing

    def _unpack_shape(xvec) -> Dict[str, float]:
        return {name: float(xvec[i]) for i, name in enumerate(shape_param_names)}

    def outer_objective(xvec):
        shape_params = _unpack_shape(xvec)
        # Reject invalid linear_plateau where n_opt <= n_min
        if photo_model == "linear_plateau":
            if shape_params.get("n_opt", 0) <= shape_params.get("n_min", 0):
                return 1e9
        # Stage 1: closed-form per-genotype H given this shape
        h_per_g = _fit_h_per_genotype_given_shape(
            genotypes, plantings_of_g_fn, DL_dict, DAP_obs_dict,
            photo_model, shape_params, photo_type,
        )
        if not h_per_g:
            return 1e9
        # Stage 2: evaluate SSE across all training cells
        total_sse = 0.0
        n = 0
        for g in genotypes:
            h_g = h_per_g.get(g)
            if h_g is None or h_g <= 0:
                continue
            if g not in DL_dict:
                continue
            for e in plantings_of_g_fn(g):
                if e not in DL_dict[g] or e not in DAP_obs_dict.get(g, {}):
                    continue
                dl = np.asarray(DL_dict[g][e], dtype=float)
                if dl.size == 0:
                    continue
                pred_dap = predict_dap_photo_only(
                    dl, photo_model, shape_params, h_g, photo_type
                )
                obs_dap = int(DAP_obs_dict[g][e])
                if pred_dap < 0:
                    # Photo never reaches threshold → penalty = (window - obs)^2
                    penalty = (float(len(dl)) - float(obs_dap)) ** 2
                    total_sse += penalty
                else:
                    total_sse += (float(pred_dap) - float(obs_dap)) ** 2
                n += 1
        if n == 0:
            return 1e9
        return total_sse / n

    minimizer_kwargs = {
        "method": "Nelder-Mead",
        "options": {"maxiter": 150, "xatol": 1e-4, "fatol": 1e-6, "adaptive": True},
    }
    result = dual_annealing(
        outer_objective, shape_bounds, maxiter=maxiter, seed=seed,
        minimizer_kwargs=minimizer_kwargs,
    )
    best_shape = _unpack_shape(result.x)
    best_h_per_g = _fit_h_per_genotype_given_shape(
        genotypes, plantings_of_g_fn, DL_dict, DAP_obs_dict,
        photo_model, best_shape, photo_type,
    )

    # Per-env SSE for diagnostics
    per_env_sse: Dict[str, float] = {}
    per_env_n: Dict[str, int] = {}
    n_total = 0
    for g in genotypes:
        h_g = best_h_per_g.get(g)
        if h_g is None or h_g <= 0 or g not in DL_dict:
            continue
        for e in plantings_of_g_fn(g):
            if e not in DL_dict[g] or e not in DAP_obs_dict.get(g, {}):
                continue
            dl = np.asarray(DL_dict[g][e], dtype=float)
            if dl.size == 0:
                continue
            pred_dap = predict_dap_photo_only(
                dl, photo_model, best_shape, h_g, photo_type
            )
            obs_dap = int(DAP_obs_dict[g][e])
            if pred_dap < 0:
                err_sq = (float(len(dl)) - float(obs_dap)) ** 2
            else:
                err_sq = (float(pred_dap) - float(obs_dap)) ** 2
            per_env_sse[e] = per_env_sse.get(e, 0.0) + err_sq
            per_env_n[e] = per_env_n.get(e, 0) + 1
            n_total += 1

    return {
        "photo_model": photo_model,
        "shape_params": best_shape,
        "h_per_genotype": best_h_per_g,
        "loss": float(result.fun),
        "n_cells_fit": n_total,
        "per_env_sse": per_env_sse,
        "per_env_n": per_env_n,
    }


def fit_photo_shape_with_thermal_frozen(
    genotypes: List[str],
    fitted_alpha: Dict[str, float],
    fitted_beta: Dict[str, float],
    temp_base: float,
    temp_optimal: float,
    temp_critical: float,
    plantings_of_g_fn: Callable[[str], Sequence[str]],
    T_dict: Dict[str, Dict[str, np.ndarray]],
    DL_dict: Dict[str, Dict[str, np.ndarray]],
    DAP_obs_dict: Dict[str, Dict[str, int]],
    photo_model: str,
    shape_bounds: List[Tuple[float, float]],
    shape_param_names: List[str],
    photo_type: str = "short_day",
    maxiter: int = 500,
    seed: Optional[int] = 42,
) -> Dict:
    """Fit photo shape with thermal frozen (Gauss-Seidel inner step).

    Mirrors `fit_photo_only` but uses `cardinal_beta(T) × F_photo(DL)`
    instead of pure photoperiod. Per-genotype H_g is the closed-form mean
    of `cum(thermal × photo)` at observed DAPs across training cells.

    This is the photo half of the Gauss-Seidel mech+photo refit: holds
    `(α, β)` per genotype frozen, optimizes `(a, b, d)` (or whatever the
    shape parameters are) against the thermal × photo cumulative DAP-error
    loss with adaptive H_g.

    Returns dict with the same shape as `fit_photo_only` plus
    `h_per_genotype` keyed by gid (combined-rate threshold).
    """
    from scipy.optimize import dual_annealing

    def _unpack_shape(xvec) -> Dict[str, float]:
        return {name: float(xvec[i]) for i, name in enumerate(shape_param_names)}

    def _h_per_g_with_thermal(shape_params: Dict[str, float]) -> Dict[str, float]:
        """Closed-form per-genotype H using thermal × photo at observed DAP."""
        h_per_g: Dict[str, float] = {}
        for g in genotypes:
            a_g = fitted_alpha.get(g)
            b_g = fitted_beta.get(g)
            if a_g is None or b_g is None or g not in DL_dict or g not in T_dict:
                continue
            cum_at_obs: List[float] = []
            for e in plantings_of_g_fn(g):
                if (e not in DL_dict[g] or e not in T_dict[g]
                        or e not in DAP_obs_dict.get(g, {})):
                    continue
                temps = np.asarray(T_dict[g][e], dtype=float)
                dl = np.asarray(DL_dict[g][e], dtype=float)
                if temps.size == 0 or dl.size == 0:
                    continue
                thermal = cardinal_beta(temps, temp_base, temp_optimal,
                                        temp_critical, a_g, b_g)
                photo = photo_rates_from_model(dl, photo_model, shape_params,
                                               photo_type)
                n = min(len(thermal), len(photo))
                if n == 0:
                    continue
                rates = thermal[:n] * photo[:n]
                obs_dap = int(DAP_obs_dict[g][e])
                idx = max(0, min(obs_dap - 1, n - 1))
                cum_val = float(np.sum(rates[: idx + 1]))
                if cum_val > 0:
                    cum_at_obs.append(cum_val)
            if cum_at_obs:
                h_per_g[g] = float(np.mean(cum_at_obs))
        return h_per_g

    def _predict_dap_combined(
        temps: np.ndarray, dl: np.ndarray, a_g: float, b_g: float,
        shape_params: Dict[str, float], h_threshold: float,
    ) -> int:
        thermal = cardinal_beta(temps, temp_base, temp_optimal, temp_critical, a_g, b_g)
        photo = photo_rates_from_model(dl, photo_model, shape_params, photo_type)
        n = min(len(thermal), len(photo))
        if n == 0:
            return -1
        rates = thermal[:n] * photo[:n]
        cum = np.cumsum(rates)
        idx = int(np.searchsorted(cum, h_threshold, side="left"))
        if idx >= n:
            return -1
        return idx + 1

    def outer_objective(xvec):
        shape_params = _unpack_shape(xvec)
        if photo_model == "linear_plateau":
            if shape_params.get("n_opt", 0) <= shape_params.get("n_min", 0):
                return 1e9
        h_per_g = _h_per_g_with_thermal(shape_params)
        if not h_per_g:
            return 1e9
        total_sse = 0.0
        n = 0
        for g in genotypes:
            h_g = h_per_g.get(g)
            a_g = fitted_alpha.get(g)
            b_g = fitted_beta.get(g)
            if h_g is None or h_g <= 0 or a_g is None or b_g is None:
                continue
            if g not in DL_dict or g not in T_dict:
                continue
            for e in plantings_of_g_fn(g):
                if (e not in DL_dict[g] or e not in T_dict[g]
                        or e not in DAP_obs_dict.get(g, {})):
                    continue
                temps = np.asarray(T_dict[g][e], dtype=float)
                dl = np.asarray(DL_dict[g][e], dtype=float)
                if temps.size == 0 or dl.size == 0:
                    continue
                pred_dap = _predict_dap_combined(temps, dl, a_g, b_g,
                                                 shape_params, h_g)
                obs_dap = int(DAP_obs_dict[g][e])
                if pred_dap < 0:
                    penalty = (float(min(len(temps), len(dl))) - float(obs_dap)) ** 2
                    total_sse += penalty
                else:
                    total_sse += (float(pred_dap) - float(obs_dap)) ** 2
                n += 1
        if n == 0:
            return 1e9
        return total_sse / n

    minimizer_kwargs = {
        "method": "Nelder-Mead",
        "options": {"maxiter": 150, "xatol": 1e-4, "fatol": 1e-6, "adaptive": True},
    }
    result = dual_annealing(
        outer_objective, shape_bounds, maxiter=maxiter, seed=seed,
        minimizer_kwargs=minimizer_kwargs,
    )
    best_shape = _unpack_shape(result.x)
    best_h_per_g = _h_per_g_with_thermal(best_shape)
    return {
        "photo_model": photo_model,
        "shape_params": best_shape,
        "h_per_genotype": best_h_per_g,
        "loss": float(result.fun),
    }


def refit_alphabeta_with_photo_frozen(
    g: str,
    plantings_of_g_fn: Callable[[str], Sequence[str]],
    T_dict: Dict[str, Dict[str, np.ndarray]],
    DL_dict: Dict[str, Dict[str, np.ndarray]],
    DAP_obs_dict: Dict[str, Dict[str, int]],
    temp_base: float,
    temp_optimal: float,
    temp_critical: float,
    alpha_bounds: Tuple[float, float],
    beta_bounds: Tuple[float, float],
    photo_model: str,
    photo_params: Dict[str, float],
    alpha_init: float,
    beta_init: float,
    target_H: float,
    photo_type: str = "short_day",
    maxiter: int = 200,
) -> Tuple[float, float, float]:
    """Refit (alpha, beta) for one genotype with photoperiod params frozen.

    Used by the Gauss-Seidel mech+photo refit. Two design choices break the
    failure modes seen in earlier attempts:

    1. **Fixed target_H** (passed in by caller): the loss is
       `Σ (cum(thermal × photo) at obs_dap - target_H)²`, NOT the variance
       loss `mech_loss_for_genotype` uses. The variance loss is scale-
       degenerate when photo is non-trivial (any α, β rescaling can be
       compensated by photo rescaling), so the GS alternation diverges.
       The fixed target — computed once from the temp-only Phase 1 fit
       per genotype — anchors the absolute scale.

    2. **Nelder-Mead with warm start** from (alpha_init, beta_init): the
       refit is a local correction to the temp-only solution, not a global
       re-search. dual_annealing here would jump to arbitrary basins on
       the still-degenerate manifold.

    Returns (alpha_map, beta_map, loss).
    """
    from scipy.optimize import minimize

    if photo_model == "logistic3":
        photo_kwargs = {
            "photo_a": float(photo_params.get("a")),
            "photo_b": float(photo_params.get("b")),
            "photo_d": float(photo_params.get("d")),
        }
    elif photo_model == "linear_plateau":
        photo_kwargs = {
            "photo_n_min": float(photo_params.get("n_min")),
            "photo_n_opt": float(photo_params.get("n_opt")),
        }
    else:
        photo_kwargs = {}

    def objective(xvec):
        a = float(xvec[0])
        b = float(xvec[1])
        if a < alpha_bounds[0] or a > alpha_bounds[1]:
            return 1e9
        if b < beta_bounds[0] or b > beta_bounds[1]:
            return 1e9
        sse = 0.0
        n = 0
        for e in plantings_of_g_fn(g):
            try:
                temps_e = np.asarray(T_dict[g][e], dtype=float)
                dap_e = int(DAP_obs_dict[g][e])
            except Exception:
                continue
            dl_e = None
            if DL_dict is not None and g in DL_dict and e in DL_dict[g]:
                dl_e = DL_dict[g][e]
            rates = daily_development_rate(
                temps_e, alpha=a, beta=b,
                temp_base=temp_base, temp_optimal=temp_optimal,
                temp_critical=temp_critical, daylengths=dl_e,
                photo_type=photo_type, photo_model=photo_model,
                **photo_kwargs,
            )
            dap_cut = max(0, min(dap_e, rates.size))
            cum_at_obs = float(np.sum(rates[:dap_cut]))
            sse += (cum_at_obs - target_H) ** 2
            n += 1
        if n == 0:
            return 1e9
        return sse / n

    x0 = np.array([
        max(alpha_bounds[0], min(alpha_bounds[1], float(alpha_init))),
        max(beta_bounds[0], min(beta_bounds[1], float(beta_init))),
    ])
    result = minimize(
        objective, x0, method="Nelder-Mead",
        options={"maxiter": int(maxiter), "xatol": 1e-4,
                 "fatol": 1e-6, "adaptive": True},
    )
    return float(result.x[0]), float(result.x[1]), float(result.fun)


# ============================================================
# Cardinal beta (daily temperature -> daily development rate)
# ============================================================
def cardinal_beta(
    temps: Sequence[float],
    temp_base: float,
    temp_optimal: float,
    temp_critical: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Cardinal beta temperature response.

    Returns raw cardinal-beta values (not clipped to 1.0) so debugging and penalty
    behaviour upstream can observe true values.
    """
    t = np.asarray(temps, dtype=float)
    out = np.zeros_like(t, dtype=float)

    # Development only between Tb and Tc
    mask = (t > temp_base) & (t < temp_critical)
    if np.any(mask):
        x = (t[mask] - temp_base) / (temp_optimal - temp_base)
        y = (temp_critical - t[mask]) / (temp_critical - temp_optimal)
        vals = (x ** alpha) * (y ** beta)
        out[mask] = vals  # do NOT clip here

    return out


def daily_development_rate(
    temps: Sequence[float],
    alpha: float,
    beta: float,
    temp_base: float,
    temp_optimal: float,
    temp_critical: float,
    daylengths: Optional[Sequence[float]] = None,
    photo_B: Optional[float] = None,
    photo_Pc: Optional[float] = None,
    photo_type: str = "short_day",
    photo_model: str = "sinclair",
    # 3-param logistic (Messina) photoperiod params
    photo_a: Optional[float] = None,
    photo_b: Optional[float] = None,
    photo_d: Optional[float] = None,
    # Linear-plateau (Grimm) photoperiod params
    photo_n_min: Optional[float] = None,
    photo_n_opt: Optional[float] = None,
) -> np.ndarray:
    """Combined daily development rate: cardinal_beta(T) × photoperiod(P).

    When daylengths and photo params are None, returns pure cardinal_beta output
    (backward compatible).

    photo_model:
        "sinclair"  — hard cutoff (Sinclair 1991), uses (B, Pc)
        "logistic"  — smooth 2-param sigmoid, uses (B, Pc)
        "logistic3" — smooth 3-param sigmoid (Messina), uses (a, b, d)
    """
    thermal = cardinal_beta(temps, temp_base, temp_optimal, temp_critical, alpha, beta)
    if daylengths is None:
        return thermal
    if photo_model == "linear_plateau":
        if photo_n_min is None or photo_n_opt is None:
            return thermal
        photo = linear_plateau_night_array(daylengths, photo_n_min, photo_n_opt)
    elif photo_model == "logistic3":
        if photo_a is None or photo_b is None or photo_d is None:
            return thermal
        photo = logistic3_photoperiod_array(daylengths, photo_a, photo_b, photo_d)
    elif photo_model == "logistic":
        if photo_B is None or photo_Pc is None:
            return thermal
        photo = logistic_photoperiod_array(daylengths, photo_B, photo_Pc, photo_type=photo_type)
    else:  # sinclair
        if photo_B is None or photo_Pc is None:
            return thermal
        photo = sinclair_photoperiod_array(daylengths, photo_B, photo_Pc, photo_type=photo_type)
    # Truncate/pad to match thermal length
    n = min(len(thermal), len(photo))
    return thermal[:n] * photo[:n]


# ============================================================
# Mechanistic loss for a genotype (scores alpha,beta)
# ============================================================
def mech_loss_for_genotype(
    g: str,
    alpha_val: float,
    beta_val: float,
    temp_base: float,
    temp_optimal: float,
    temp_critical: float,
    plantings_of_g_fn: Callable[[str], Sequence[str]],
    T_dict: Dict[str, Dict[str, np.ndarray]],
    DAP_obs_dict: Dict[str, Dict[str, int]],
    use_penalty: bool = False,
    alpha_target: float = 0.0,
    beta_target: float = 0.0,
    lambda_a: float = 0.0,
    lambda_b: float = 0.0,
    normalize_loss: bool = False,
    DL_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    photo_B: Optional[float] = None,
    photo_Pc: Optional[float] = None,
    photo_type: str = "short_day",
    photo_model: str = "sinclair",
    # 3-param logistic (Messina) photoperiod
    photo_a: Optional[float] = None,
    photo_b: Optional[float] = None,
    photo_d: Optional[float] = None,
    # Linear-plateau (Grimm 1993) photoperiod params
    photo_n_min: Optional[float] = None,
    photo_n_opt: Optional[float] = None,
    **kwargs
) -> float:
    """
    Compute mechanistic loss for genotype g.

    Accepts optional penalty parameters (use_penalty, alpha_target, beta_target, lambda_a, lambda_b).
    When normalize_loss=True, divides the variance term by the number of plantings so that
    genotypes with more training environments are not penalised disproportionately.
    Optional photoperiod: DL_dict + photo params enable multiplicative coupling.
    photo_model="logistic3" uses (photo_a, photo_b, photo_d) instead of (photo_B, photo_Pc).
    photo_model="linear_plateau" uses (photo_n_min, photo_n_opt) on night length.
    Extra kwargs are tolerated so callers using different keyword sets won't raise unexpected-arg errors.
    """
    C_list: List[float] = []

    for e in plantings_of_g_fn(g):
        try:
            temps_e = np.asarray(T_dict[g][e], dtype=float)
            dap_e = int(DAP_obs_dict[g][e])
        except Exception:
            # missing data for this planting -> skip
            continue

        # Get daylength array if available
        dl_e = None
        if DL_dict is not None and g in DL_dict and e in DL_dict[g]:
            dl_e = DL_dict[g][e]

        daily_rates = daily_development_rate(
            temps_e,
            alpha=alpha_val,
            beta=beta_val,
            temp_base=temp_base,
            temp_optimal=temp_optimal,
            temp_critical=temp_critical,
            daylengths=dl_e,
            photo_B=photo_B,
            photo_Pc=photo_Pc,
            photo_type=photo_type,
            photo_model=photo_model,
            photo_a=photo_a,
            photo_b=photo_b,
            photo_d=photo_d,
            photo_n_min=photo_n_min,
            photo_n_opt=photo_n_opt,
        )

        # Clip DAP to available weather length
        dap_cut = max(0, min(dap_e, daily_rates.size))
        total_progress_e = float(np.sum(daily_rates[:dap_cut]))
        C_list.append(total_progress_e)

    if len(C_list) == 0:
        # no observations -> large loss
        return 1e9

    Hbar = float(np.mean(C_list))
    diffs = np.asarray(C_list, dtype=float) - Hbar
    var_term = float(np.sum(diffs ** 2))
    if normalize_loss and len(C_list) > 1:
        var_term /= len(C_list)

    if not use_penalty:
        return var_term

    # quadratic penalty towards genomic targets
    penalty_term = float(lambda_a * (alpha_val - alpha_target) ** 2 + lambda_b * (beta_val - beta_target) ** 2)
    return var_term + penalty_term


def simulate_dap_to_threshold(
    thermal_rates: np.ndarray,
    photo_rates: np.ndarray,
    threshold: float,
) -> int:
    """Simulate forward day-by-day until cumulative R(t) reaches threshold.

    R(t) = thermal_rate[t] * photo_rate[t]
    Returns the day index (1-based) when cumulative sum reaches threshold,
    or len(rates) if threshold not reached.

    Used by the Grimm-style direct DAP error loss.
    """
    n = min(len(thermal_rates), len(photo_rates))
    if n == 0:
        return 1
    daily = thermal_rates[:n] * photo_rates[:n]
    cumsum = np.cumsum(daily)
    # Find first index where cumsum >= threshold
    reached = np.searchsorted(cumsum, threshold, side="left")
    return int(reached) + 1 if reached < n else n


def dap_error_loss_for_genotype_linplat(
    g: str,
    alpha_val: float,
    beta_val: float,
    temp_base: float,
    temp_optimal: float,
    temp_critical: float,
    plantings_of_g_fn: Callable[[str], Sequence[str]],
    T_dict: Dict[str, Dict[str, np.ndarray]],
    DAP_obs_dict: Dict[str, Dict[str, int]],
    DL_dict: Dict[str, Dict[str, np.ndarray]],
    photo_n_min: float,
    photo_n_opt: float,
) -> float:
    """Direct DAP error loss (Grimm et al. 1993) for linear-plateau photoperiod.

    Simulates forward day-by-day with R(t) = cardinal_beta(T) * F_N(night).
    Finds the day when cumulative R(t) reaches the genotype's threshold (1/alpha).
    Returns sum of squared errors between observed and simulated DAP across plantings.

    This is fundamentally different from variance-based loss — it directly matches
    predicted flowering dates to observed, which prevents degenerate solutions
    because the threshold gives the optimizer a concrete target.
    """
    threshold = 1.0 / alpha_val if alpha_val > 0 else 1.0

    sse = 0.0
    n_valid = 0
    for e in plantings_of_g_fn(g):
        try:
            temps_e = np.asarray(T_dict[g][e], dtype=float)
            dap_obs = int(DAP_obs_dict[g][e])
        except Exception:
            continue
        if g not in DL_dict or e not in DL_dict[g]:
            continue
        dl_e = DL_dict[g][e]

        thermal = cardinal_beta(temps_e, temp_base, temp_optimal, temp_critical,
                                alpha_val, beta_val)
        photo = linear_plateau_night_array(dl_e, photo_n_min, photo_n_opt)
        dap_pred = simulate_dap_to_threshold(thermal, photo, threshold)

        sse += (dap_pred - dap_obs) ** 2
        n_valid += 1

    if n_valid == 0:
        return 1e9
    return sse / n_valid


def photo_loss_for_genotype(
    g: str,
    photo_b: float,
    photo_d: float,
    photo_a: float,
    plantings_of_g_fn: Callable[[str], Sequence[str]],
    DL_dict: Dict[str, Dict[str, np.ndarray]],
    DAP_obs_dict: Dict[str, Dict[str, int]],
    normalize_loss: bool = False,
) -> float:
    """Pure photoperiod loss — no temperature, only daylength.

    Computes cumulative photoperiod progress = sum(g(DL)) up to observed DAP,
    then returns variance across plantings. Same principle as the thermal loss
    but using only the daylength response.

    g(DL) = 1 / (1 + a * exp(b * (DL - d)))
    """
    R_list: List[float] = []  # ratio: mean daily g(P) per planting

    for e in plantings_of_g_fn(g):
        try:
            dap_e = int(DAP_obs_dict[g][e])
        except Exception:
            continue
        if g not in DL_dict or e not in DL_dict[g]:
            continue
        dl_e = DL_dict[g][e]

        photo_rates = logistic3_photoperiod_array(dl_e, photo_a, photo_b, photo_d)

        dap_cut = max(1, min(dap_e, photo_rates.size))
        # Ratio: cumulative_photo / DAP = mean daily g(P)
        # If g(P)=1 everywhere, ratio=1.0. Photoperiod-sensitive genotypes
        # will have ratio < 1.0 at long-day sites but should be CONSISTENT.
        mean_daily_gP = float(np.sum(photo_rates[:dap_cut])) / dap_cut
        R_list.append(mean_daily_gP)

    if len(R_list) == 0:
        return 1e9

    Rbar = float(np.mean(R_list))
    diffs = np.asarray(R_list, dtype=float) - Rbar
    var_term = float(np.sum(diffs ** 2))
    if normalize_loss and len(R_list) > 1:
        var_term /= len(R_list)
    return var_term


def fit_population_photo_linplat(
    genotypes: List[str],
    fitted_alpha: Dict[str, float],
    fitted_beta: Dict[str, float],
    temp_base: float,
    temp_optimal: float,
    temp_critical: float,
    plantings_of_g_fn: Callable[[str], Sequence[str]],
    T_dict: Dict[str, Dict[str, np.ndarray]],
    DL_dict: Dict[str, Dict[str, np.ndarray]],
    DAP_obs_dict: Dict[str, Dict[str, int]],
    n_min_bounds: Tuple[float, float] = (6.0, 14.0),
    n_opt_bounds: Tuple[float, float] = (8.0, 16.0),
    maxiter: int = 500,
    seed: Optional[int] = 42,
) -> Dict:
    """Fit population-level linear-plateau photoperiod (N_min, N_opt) using
    the Grimm et al. (1993) direct DAP error loss.

    Uses REAL per-genotype alpha/beta (from a prior temp-only fit) as the
    fixed temperature model. The photoperiod (N_min, N_opt) is fitted by
    minimizing the sum of squared errors between observed and simulated
    flowering dates across all genotypes and plantings.

    This is the key improvement over variance-based loss: the optimizer
    has a concrete target (observed DAP), so it can't find degenerate
    solutions where g(P) is constant.

    Args:
        fitted_alpha: {genotype_id: alpha} from prior temp-only fit
        fitted_beta: {genotype_id: beta} from prior temp-only fit

    Returns dict with photo_n_min, photo_n_opt, loss.
    """
    from scipy.optimize import dual_annealing

    def objective(xvec):
        n_min_v = float(xvec[0])
        n_opt_v = float(xvec[1])
        if n_opt_v <= n_min_v:
            return 1e9  # invalid: plateau must be above minimum
        total_sse = 0.0
        n_count = 0
        for g in genotypes:
            a = fitted_alpha.get(g)
            b = fitted_beta.get(g)
            if a is None or b is None:
                continue
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            loss_g = dap_error_loss_for_genotype_linplat(
                g=g, alpha_val=a, beta_val=b,
                temp_base=temp_base, temp_optimal=temp_optimal,
                temp_critical=temp_critical,
                plantings_of_g_fn=plantings_of_g_fn,
                T_dict=T_dict, DAP_obs_dict=DAP_obs_dict,
                DL_dict=DL_dict,
                photo_n_min=n_min_v, photo_n_opt=n_opt_v,
            )
            if loss_g < 1e8:
                total_sse += loss_g
                n_count += 1
        return total_sse / max(1, n_count)

    # ── Dynamic n_min upper bound constraint ───────────────────────────
    # Compute the minimum night length across all (genotype, training-env)
    # cells, then force n_min to stay at least SAFETY_MARGIN_H below it.
    # Without this, dual_annealing can land on n_min just above the lowest
    # training night length, which makes F(N)=0 at that env. Mech then
    # cannot accumulate progress there, simulate_dap_to_threshold returns
    # the weather window length (capping the per-cell error at a constant
    # ~150-200 days regardless of how wrong it is), and the optimizer has
    # no incentive to escape that local minimum. The downstream effect is
    # mech predictions exploding to ~192 days for held-out high-latitude
    # cells (see TECHNICAL.md photoperiod methodology section).
    SAFETY_MARGIN_H = 0.5  # see TECHNICAL.md
    min_night_train = float("inf")
    for g in genotypes:
        if g not in DL_dict:
            continue
        for e in plantings_of_g_fn(g):
            if e in DL_dict[g]:
                dl_arr = np.asarray(DL_dict[g][e], dtype=float)
                if dl_arr.size > 0:
                    night = 24.0 - dl_arr
                    cell_min_night = float(np.min(night))
                    if cell_min_night < min_night_train:
                        min_night_train = cell_min_night

    if not np.isfinite(min_night_train):
        # No daylength data — can't constrain; fall back to user bounds
        effective_n_min_max = float(n_min_bounds[1])
    else:
        effective_n_min_max = min(float(n_min_bounds[1]), min_night_train - SAFETY_MARGIN_H)

    if effective_n_min_max <= float(n_min_bounds[0]):
        print(
            f"  WARNING: min training night length ({min_night_train:.2f}h) "
            f"too low for n_min bounds [{n_min_bounds[0]}, {n_min_bounds[1]}h] "
            f"with {SAFETY_MARGIN_H}h safety margin → effective upper bound "
            f"{effective_n_min_max:.2f}h ≤ lower bound. Disabling photoperiod."
        )
        return {
            "photo_n_min": None,
            "photo_n_opt": None,
            "loss": float("inf"),
            "temp_only_loss": float("inf"),
            "improvement": 0.0,
        }

    print(
        f"  Constrained n_min upper bound to {effective_n_min_max:.2f}h "
        f"(min training night length = {min_night_train:.2f}h, "
        f"safety margin = {SAFETY_MARGIN_H}h)"
    )

    bounds = [(float(n_min_bounds[0]), effective_n_min_max), n_opt_bounds]
    minimizer_kwargs = {
        "method": "Nelder-Mead",
        "options": {"maxiter": 150, "xatol": 1e-4, "fatol": 1e-6, "adaptive": True},
    }

    result = dual_annealing(
        objective, bounds, maxiter=maxiter, seed=seed,
        minimizer_kwargs=minimizer_kwargs,
    )

    best_n_min, best_n_opt = float(result.x[0]), float(result.x[1])
    best_loss = float(result.fun)

    # Baseline: temp-only loss (n_min = n_opt → F(N) = 1 always, but our
    # function returns 1.0 when n_opt <= n_min... we need a different baseline)
    # Compute temp-only SSE for comparison
    temp_only_sse = 0.0
    n_count = 0
    for g in genotypes:
        a = fitted_alpha.get(g)
        b = fitted_beta.get(g)
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
            continue
        threshold = 1.0 / a if a > 0 else 1.0
        sse = 0.0
        n_valid = 0
        for e in plantings_of_g_fn(g):
            try:
                temps_e = np.asarray(T_dict[g][e], dtype=float)
                dap_obs = int(DAP_obs_dict[g][e])
            except Exception:
                continue
            thermal = cardinal_beta(temps_e, temp_base, temp_optimal, temp_critical, a, b)
            photo_ones = np.ones(len(thermal))
            dap_pred = simulate_dap_to_threshold(thermal, photo_ones, threshold)
            sse += (dap_pred - dap_obs) ** 2
            n_valid += 1
        if n_valid > 0:
            temp_only_sse += sse / n_valid
            n_count += 1
    temp_only_loss = temp_only_sse / max(1, n_count)

    print(f"  Population linear-plateau photoperiod (Grimm 1993):")
    print(f"    N_min={best_n_min:.3f}h, N_opt={best_n_opt:.3f}h (night length)")
    print(f"    Daylength equivalent: max DL={24-best_n_min:.2f}h, min full-rate DL={24-best_n_opt:.2f}h")
    print(f"    Photo loss (SSE/n): {best_loss:.2f}  |  Temp-only baseline: {temp_only_loss:.2f}")
    if best_loss < temp_only_loss - 0.5:
        print(f"    -> Photoperiod IMPROVES fit (delta = {temp_only_loss - best_loss:.2f})")
    else:
        print(f"    -> Photoperiod provides no signal (delta = {temp_only_loss - best_loss:.2f})")

    # Show F(N) at representative daylengths
    for dl in [10, 12, 14, 16]:
        gp = linear_plateau_night_array([dl], best_n_min, best_n_opt)[0]
        print(f"    DL={dl}h (night={24-dl}h): F(N) = {gp:.4f}")

    return {
        "photo_n_min": best_n_min,
        "photo_n_opt": best_n_opt,
        "loss": best_loss,
        "temp_only_loss": temp_only_loss,
        "improvement": temp_only_loss - best_loss,
    }


def dap_error_loss_for_genotype_logistic3(
    g: str,
    alpha_val: float,
    beta_val: float,
    temp_base: float,
    temp_optimal: float,
    temp_critical: float,
    plantings_of_g_fn: Callable[[str], Sequence[str]],
    T_dict: Dict[str, Dict[str, np.ndarray]],
    DAP_obs_dict: Dict[str, Dict[str, int]],
    DL_dict: Dict[str, Dict[str, np.ndarray]],
    photo_a: float,
    photo_b: float,
    photo_d: float,
) -> float:
    """Direct DAP error loss for logistic3 (Messina) photoperiod.

    Mirror of dap_error_loss_for_genotype_linplat but uses the smooth
    3-parameter logistic photoperiod response instead of the linear-plateau
    function. Simulates forward day-by-day with
      R(t) = cardinal_beta(T_t, α, β) × logistic3(DL_t, a, b, d)
    and finds the day when cumulative R(t) reaches the genotype's threshold
    (1/α). Returns SSE across plantings.

    Unlike linear_plateau, logistic3 has no hard floor — F(DL) is always
    strictly positive (floor = 1/(1+a)), so simulate_dap_to_threshold never
    caps at the weather window length and the optimizer sees a smooth
    gradient everywhere.
    """
    threshold = 1.0 / alpha_val if alpha_val > 0 else 1.0

    sse = 0.0
    n_valid = 0
    for e in plantings_of_g_fn(g):
        try:
            temps_e = np.asarray(T_dict[g][e], dtype=float)
            dap_obs = int(DAP_obs_dict[g][e])
        except Exception:
            continue
        if g not in DL_dict or e not in DL_dict[g]:
            continue
        dl_e = DL_dict[g][e]

        thermal = cardinal_beta(temps_e, temp_base, temp_optimal, temp_critical,
                                alpha_val, beta_val)
        photo = logistic3_photoperiod_array(dl_e, photo_a, photo_b, photo_d)
        dap_pred = simulate_dap_to_threshold(thermal, photo, threshold)

        sse += (dap_pred - dap_obs) ** 2
        n_valid += 1

    if n_valid == 0:
        return 1e9
    return sse / n_valid


def fit_population_photo_logistic3_dap_error(
    genotypes: List[str],
    fitted_alpha: Dict[str, float],
    fitted_beta: Dict[str, float],
    temp_base: float,
    temp_optimal: float,
    temp_critical: float,
    plantings_of_g_fn: Callable[[str], Sequence[str]],
    T_dict: Dict[str, Dict[str, np.ndarray]],
    DL_dict: Dict[str, Dict[str, np.ndarray]],
    DAP_obs_dict: Dict[str, Dict[str, int]],
    a_bounds: Tuple[float, float] = (0.1, 5.0),
    b_bounds: Tuple[float, float] = (-5.0, 5.0),
    d_bounds: Tuple[float, float] = (10.0, 18.0),
    photo_type: str = "short_day",
    maxiter: int = 500,
    seed: Optional[int] = 42,
) -> Dict:
    """Fit population-level logistic3 (a, b, d) using direct DAP-error loss.

    This is the logistic3 analog of `fit_population_photo_linplat`. It takes
    fitted per-genotype (alpha, beta) from temp-only coupling and fits photo
    shape params that minimize SSE between predicted and observed DAP under
    the multiplicative model:
        R(t) = cardinal_beta(T_t, α_g, β_g) × logistic3(DL_t, a, b, d)
        pred_dap = min t such that sum(R[:t]) >= 1/α_g

    Logistic3 advantages over linear_plateau:
      - Smooth sigmoid has no hard floor → no cliff in the loss surface
      - Floor = 1/(1+a) > 0 always, so rates never collapse to zero
      - No need for a min-night safety margin (the hard-floor cliff doesn't exist)
      - Photo-only experiments showed this shape fits Beans much better

    Args:
        fitted_alpha/beta: {gid: value} from prior temp-only fit
        a_bounds: amplitude (controls floor = 1/(1+a))
        b_bounds: slope bounds. AUTOMATICALLY CONSTRAINED BY photo_type:
                  - short_day (beans, soybean): forces b >= 0 so F(DL)
                    is HIGH at low DL (long nights) and LOW at high DL.
                  - long_day (broccoli): forces b <= 0 so F(DL) is HIGH
                    at high DL.
                  Without the sign constraint the optimizer can land on a
                  wrong-direction local minimum that fits tropical training
                  but extrapolates catastrophically to extreme latitudes.
        d_bounds: inflection point (daylength at half-max response, hours)
        photo_type: "short_day" or "long_day" — controls b sign constraint

    Returns dict with photo_a, photo_b, photo_d, loss, temp_only_loss, improvement.
    """
    from scipy.optimize import dual_annealing

    # Enforce physiological direction via b sign. Without this the
    # optimizer has a degenerate freedom to pick either short-day OR
    # long-day response; with tropical-only training the fit can't
    # distinguish them, and the wrong-direction solution blows up
    # predictions at extreme latitudes (ND extrapolation).
    b_lo, b_hi = float(b_bounds[0]), float(b_bounds[1])
    if photo_type == "short_day":
        # Short-day plants flower when nights are long → b > 0 makes F
        # decrease as DL increases (i.e. higher F at low DL = long nights)
        b_lo = max(b_lo, 0.0)
    elif photo_type == "long_day":
        # Long-day plants flower when days are long → b < 0 makes F
        # increase as DL increases (i.e. higher F at high DL)
        b_hi = min(b_hi, 0.0)
    if b_lo >= b_hi:
        # Degenerate after constraint — caller's bounds don't span the
        # required sign. Widen slightly so dual_annealing doesn't crash.
        b_lo, b_hi = -0.01 if photo_type == "long_day" else 0.0, 0.01
    b_bounds_eff = (b_lo, b_hi)

    def objective(xvec):
        a_val = float(xvec[0])
        b_val = float(xvec[1])
        d_val = float(xvec[2])
        total_sse = 0.0
        n_count = 0
        for g in genotypes:
            a_g = fitted_alpha.get(g)
            b_g = fitted_beta.get(g)
            if a_g is None or b_g is None:
                continue
            if not (np.isfinite(a_g) and np.isfinite(b_g)):
                continue
            loss_g = dap_error_loss_for_genotype_logistic3(
                g=g, alpha_val=a_g, beta_val=b_g,
                temp_base=temp_base, temp_optimal=temp_optimal,
                temp_critical=temp_critical,
                plantings_of_g_fn=plantings_of_g_fn,
                T_dict=T_dict, DAP_obs_dict=DAP_obs_dict,
                DL_dict=DL_dict,
                photo_a=a_val, photo_b=b_val, photo_d=d_val,
            )
            if loss_g < 1e8:
                total_sse += loss_g
                n_count += 1
        return total_sse / max(1, n_count)

    bounds = [a_bounds, b_bounds_eff, d_bounds]
    print(f"    b_bounds after {photo_type} sign constraint: {b_bounds_eff}")
    minimizer_kwargs = {
        "method": "Nelder-Mead",
        "options": {"maxiter": 150, "xatol": 1e-4, "fatol": 1e-6, "adaptive": True},
    }

    result = dual_annealing(
        objective, bounds, maxiter=maxiter, seed=seed,
        minimizer_kwargs=minimizer_kwargs,
    )

    best_a, best_b, best_d = float(result.x[0]), float(result.x[1]), float(result.x[2])
    best_loss = float(result.fun)

    # Baseline: temp-only SSE (no photo applied)
    temp_only_sse = 0.0
    n_count = 0
    for g in genotypes:
        a_g = fitted_alpha.get(g)
        b_g = fitted_beta.get(g)
        if a_g is None or b_g is None or not np.isfinite(a_g) or not np.isfinite(b_g):
            continue
        threshold = 1.0 / a_g if a_g > 0 else 1.0
        sse = 0.0
        n_valid = 0
        for e in plantings_of_g_fn(g):
            try:
                temps_e = np.asarray(T_dict[g][e], dtype=float)
                dap_obs = int(DAP_obs_dict[g][e])
            except Exception:
                continue
            thermal = cardinal_beta(temps_e, temp_base, temp_optimal, temp_critical, a_g, b_g)
            photo_ones = np.ones(len(thermal))
            dap_pred = simulate_dap_to_threshold(thermal, photo_ones, threshold)
            sse += (dap_pred - dap_obs) ** 2
            n_valid += 1
        if n_valid > 0:
            temp_only_sse += sse / n_valid
            n_count += 1
    temp_only_loss = temp_only_sse / max(1, n_count)

    print(f"  Population logistic3 photoperiod (Messina, DAP-error loss):")
    print(f"    a={best_a:.4f}, b={best_b:.4f}, d={best_d:.3f}h")
    print(f"    Floor: 1/(1+a) = {1.0/(1.0+best_a):.4f}")
    print(f"    Photo loss (SSE/n): {best_loss:.2f}  |  Temp-only baseline: {temp_only_loss:.2f}")
    if best_loss < temp_only_loss - 0.5:
        print(f"    -> Photoperiod IMPROVES fit (delta = {temp_only_loss - best_loss:.2f})")
    else:
        print(f"    -> Photoperiod provides no signal (delta = {temp_only_loss - best_loss:.2f})")

    # Show F(DL) at representative daylengths
    for dl in [10, 12, 14, 16]:
        gp = logistic3_photoperiod_array([dl], best_a, best_b, best_d)[0]
        print(f"    DL={dl}h: F(DL) = {gp:.4f}")

    return {
        "photo_a": best_a,
        "photo_b": best_b,
        "photo_d": best_d,
        "loss": best_loss,
        "temp_only_loss": temp_only_loss,
        "improvement": temp_only_loss - best_loss,
    }


def fit_population_photo(
    genotypes: List[str],
    plantings_of_g_fn: Callable[[str], Sequence[str]],
    DL_dict: Dict[str, Dict[str, np.ndarray]],
    DAP_obs_dict: Dict[str, Dict[str, int]],
    photo_a: float = 0.05,
    photo_b_bounds: Tuple[float, float] = (0.0, 2.0),
    photo_d_bounds: Tuple[float, float] = (8.0, 18.0),
    maxiter: int = 500,
    seed: Optional[int] = 42,
) -> Dict:
    """Fit population-level photoperiod params (b, d) shared by all genotypes.

    One dual annealing call, 2D. Sums the ratio-based photo loss across all
    genotypes. The `a` parameter (floor) is fixed from config.

    Returns dict with photo_a (fixed), photo_b, photo_d, loss.
    """
    from scipy.optimize import dual_annealing

    def objective(xvec):
        b_val = float(xvec[0])
        d_val = float(xvec[1])
        total = 0.0
        for g in genotypes:
            loss_g = photo_loss_for_genotype(
                g=g, photo_b=b_val, photo_d=d_val, photo_a=photo_a,
                plantings_of_g_fn=plantings_of_g_fn,
                DL_dict=DL_dict, DAP_obs_dict=DAP_obs_dict,
            )
            if loss_g < 1e8:  # skip genotypes with missing data
                total += loss_g
        return total

    bounds = [photo_b_bounds, photo_d_bounds]
    minimizer_kwargs = {
        "method": "Nelder-Mead",
        "options": {"maxiter": 150, "xatol": 1e-4, "fatol": 1e-6, "adaptive": True},
    }

    result = dual_annealing(
        objective, bounds, maxiter=maxiter, seed=seed,
        minimizer_kwargs=minimizer_kwargs,
    )

    best_b, best_d = float(result.x[0]), float(result.x[1])
    best_loss = float(result.fun)

    print(f"  Population photoperiod: b={best_b:.4f}, d={best_d:.4f}, "
          f"a={photo_a:.4f} (fixed), loss={best_loss:.4f}")

    # Show g(P) at representative daylengths
    for dl in [10, 12, 14, 16]:
        gp = logistic3_photoperiod_array([dl], photo_a, best_b, best_d)[0]
        print(f"    DL={dl}h: g(P) = {gp:.4f}")

    return {
        "photo_a": photo_a,
        "photo_b": best_b,
        "photo_d": best_d,
        "loss": best_loss,
    }


# ============================================================
# Optimizer wrapper (dual annealing finds best alpha,beta)
# ============================================================
def fit_alpha_beta_for_genotype(
    g: str,
    temp_base: float,
    temp_optimal: float,
    temp_critical: float,
    plantings_of_g_fn: Callable[[str], Sequence[str]],
    T_dict: Dict[str, Dict[str, np.ndarray]],
    DAP_obs_dict: Dict[str, Dict[str, int]],
    alpha_bounds: Tuple[float, float] = (0.5, 8.0),
    beta_bounds: Tuple[float, float] = (0.5, 8.0),
    maxiter: int = 2000,
    initial_temp: float = 5230.0,
    visit: float = 2.62,
    accept: float = -5.0,
    local_max_iter: int = 150,
    seed: Optional[int] = 42,
    n_restarts: int = 1,
    # penalty inputs (defaults -> no penalty). Added so callers may pass use_penalty/alpha_gen/beta_gen/lambda_a/lambda_b
    use_penalty: bool = False,
    alpha_gen: float = 0.0,
    beta_gen: float = 0.0,
    lambda_a: float = 0.0,
    lambda_b: float = 0.0,
    normalize_loss: bool = False,
    # photoperiod (optional)
    DL_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    photo_B: Optional[float] = None,
    photo_Pc: Optional[float] = None,
    photo_type: str = "short_day",
    photo_model: str = "sinclair",
) -> Tuple[float, float, float, int, int]:
    """
    Fit (alpha, beta) for one genotype using SciPy dual_annealing.

    Returns (alpha_hat, beta_hat, best_loss, nit, nfev)
    """
    def objective(xvec):
        a = float(xvec[0])
        b = float(xvec[1])
        # Enforce bounds (Nelder-Mead can wander outside DA's bounds)
        if a < alpha_bounds[0] or a > alpha_bounds[1] or b < beta_bounds[0] or b > beta_bounds[1]:
            return 1e9
        return mech_loss_for_genotype(
            g=g,
            alpha_val=a,
            beta_val=b,
            temp_base=temp_base,
            temp_optimal=temp_optimal,
            temp_critical=temp_critical,
            plantings_of_g_fn=plantings_of_g_fn,
            T_dict=T_dict,
            DAP_obs_dict=DAP_obs_dict,
            use_penalty=use_penalty,
            alpha_target=alpha_gen,
            beta_target=beta_gen,
            lambda_a=lambda_a,
            lambda_b=lambda_b,
            normalize_loss=normalize_loss,
            DL_dict=DL_dict,
            photo_B=photo_B,
            photo_Pc=photo_Pc,
            photo_type=photo_type,
            photo_model=photo_model,
        )

    bounds = [alpha_bounds, beta_bounds]

    minimizer_kwargs = {
        "method": "Nelder-Mead",
        "options": {
            "maxiter": local_max_iter,
            "xatol": 1e-4,
            "fatol": 1e-6,
            "adaptive": True,
        },
    }

    best_res = None
    best_fun = float("inf")

    for r in range(max(1, int(n_restarts))):
        seed_r = None if seed is None else int(seed) + r
        try:
            res = dual_annealing(
                func=objective,
                bounds=bounds,
                maxiter=maxiter,
                initial_temp=initial_temp,
                visit=visit,
                accept=accept,
                minimizer_kwargs=minimizer_kwargs,
                seed=seed_r,
            )
        except TypeError:
            # older SciPy might not accept seed keyword — fall back without it
            res = dual_annealing(
                func=objective,
                bounds=bounds,
                maxiter=maxiter,
                initial_temp=initial_temp,
                visit=visit,
                accept=accept,
                minimizer_kwargs=minimizer_kwargs,
            )

        if res.fun < best_fun:
            best_fun = float(res.fun)
            best_res = res

    if best_res is None:
        raise RuntimeError("Optimizer failed to return a result")

    alpha_hat = float(best_res.x[0])
    beta_hat = float(best_res.x[1])
    best_L = float(best_res.fun)
    nit = int(getattr(best_res, "nit", -1))
    nfev = int(getattr(best_res, "nfev", -1))
    return alpha_hat, beta_hat, best_L, nit, nfev


# ============================================================
# Top-K dual annealing (replaces ABC for iterative coupling)
# ============================================================
def topk_dual_annealing(
    g: str,
    temp_base: float,
    temp_optimal: float,
    temp_critical: float,
    plantings_of_g_fn: Callable[[str], Sequence[str]],
    T_dict: Dict[str, Dict[str, np.ndarray]],
    DAP_obs_dict: Dict[str, Dict[str, int]],
    alpha_bounds: Tuple[float, float] = (0.5, 8.0),
    beta_bounds: Tuple[float, float] = (0.5, 8.0),
    top_k: int = 100,
    maxiter: int = 2000,
    local_max_iter: int = 150,
    seed: Optional[int] = 42,
    n_restarts: int = 1,
    normalize_loss: bool = False,
    # Penalty (GBLUP prior)
    use_penalty: bool = False,
    alpha_target: float = 0.0,
    beta_target: float = 0.0,
    lambda_a: float = 0.0,
    lambda_b: float = 0.0,
    # Topt fitting (3rd parameter)
    topt_bounds: Optional[Tuple[float, float]] = None,
    topt_target: float = 0.0,
    lambda_t: float = 0.0,
    # Photoperiod
    DL_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    photo_B: Optional[float] = None,
    photo_Pc: Optional[float] = None,
    photo_type: str = "short_day",
    photo_model: str = "sinclair",
    # Per-genotype photoperiod fitting (B, Pc as optimization params — legacy 2-param)
    photo_B_bounds: Optional[Tuple[float, float]] = None,
    photo_Pc_bounds: Optional[Tuple[float, float]] = None,
    photo_B_target: float = 0.0,
    photo_Pc_target: float = 0.0,
    lambda_bphoto: float = 0.0,
    lambda_pc: float = 0.0,
    # Per-genotype 3-param logistic photoperiod (Messina formulation)
    photo_a_bounds: Optional[Tuple[float, float]] = None,
    photo_b_bounds: Optional[Tuple[float, float]] = None,
    photo_d_bounds: Optional[Tuple[float, float]] = None,
    photo_a_target: float = 0.0,
    photo_b_target: float = 0.0,
    photo_d_target: float = 0.0,
    lambda_photo_a: float = 0.0,
    lambda_photo_b: float = 0.0,
    lambda_photo_d: float = 0.0,
    # Linear-plateau (Grimm 1993) photoperiod — population-level fixed params
    # passed in from outside, NOT optimized per genotype.
    photo_n_min: Optional[float] = None,
    photo_n_opt: Optional[float] = None,
    # Observation perturbation (ES-MDA-inspired posterior diversity)
    obs_perturbation_sd: float = 0.0,
    perturb_maxiter: int = 0,
) -> Dict:
    """Top-K dual annealing: record every evaluated point, keep the best K.

    Instead of DA-for-MAP + ABC rejection sampling, this instruments the
    objective function to record every (alpha, beta[, topt], loss) evaluation
    during dual annealing's search. After completion, the top K points (by
    loss) form a per-genotype distribution for uncertainty quantification.

    When topt_bounds is provided, Topt becomes a 3rd fitted parameter
    (3D optimization). Otherwise, temp_optimal is used as a fixed value
    (2D optimization, backward compatible).

    When photo_B_bounds and photo_Pc_bounds are provided, photoperiod B and Pc
    become per-genotype fitted parameters appended to the optimization vector.
    This replaces the slow nested grid search with per-genotype optimization.

    When use_penalty=True, the objective includes a quadratic penalty toward
    GBLUP targets. Top-K selection uses the penalized loss.

    When obs_perturbation_sd > 0, uses per-restart MAP collection: each
    restart optimizes against perturbed observations and yields one MAP point.
    All MAPs are re-evaluated on original data for comparable losses. This
    produces genuine posterior diversity (one sample per restart).

    perturb_maxiter: DA iterations for perturbed restarts (r > 0). Default 0
    means auto = max(200, maxiter // 10). Only used when perturbation active.

    Returns dict matching abc_sample_posterior() format for compatibility
    with posterior_predictive_dap() and ensemble weighting.
    """
    n_plantings = len(list(plantings_of_g_fn(g)))
    fit_topt = topt_bounds is not None
    fit_photo = photo_B_bounds is not None and photo_Pc_bounds is not None
    fit_photo3 = (photo_a_bounds is not None and photo_b_bounds is not None
                  and photo_d_bounds is not None)

    # Recording list: each entry is (alpha, beta, topt, photo_B, photo_Pc,
    #   photo_a, photo_b, photo_d, penalized_loss, pure_mech_loss)
    recorded: List[Tuple[float, float, float, float, float, float, float]] = []

    # Mutable reference for perturbed observations — swapped per restart
    active_DAP_obs = [DAP_obs_dict]
    rng_perturb = np.random.RandomState(seed if seed is not None else 42)

    def objective(xvec):
        a = float(xvec[0])
        b = float(xvec[1])
        t = float(xvec[2]) if fit_topt else temp_optimal
        # Enforce bounds (Nelder-Mead can wander outside DA's bounds)
        if a < alpha_bounds[0] or a > alpha_bounds[1] or b < beta_bounds[0] or b > beta_bounds[1]:
            return 1e9
        if fit_topt and (t < topt_bounds[0] or t > topt_bounds[1]):
            return 1e9

        # Unpack photoperiod params from optimization vector
        next_idx = 3 if fit_topt else 2
        B_val, Pc_val = photo_B, photo_Pc
        a_photo_val, b_photo_val, d_photo_val = None, None, None

        if fit_photo3:
            # 3-param logistic (Messina): (a_photo, b_photo, d_photo)
            a_photo_val = float(xvec[next_idx])
            b_photo_val = float(xvec[next_idx + 1])
            d_photo_val = float(xvec[next_idx + 2])
            if (a_photo_val < photo_a_bounds[0] or a_photo_val > photo_a_bounds[1] or
                b_photo_val < photo_b_bounds[0] or b_photo_val > photo_b_bounds[1] or
                d_photo_val < photo_d_bounds[0] or d_photo_val > photo_d_bounds[1]):
                return 1e9
            eff_photo_model = "logistic3"
        elif fit_photo:
            # Legacy 2-param: (B, Pc)
            B_val = float(xvec[next_idx])
            Pc_val = float(xvec[next_idx + 1])
            if B_val < photo_B_bounds[0] or B_val > photo_B_bounds[1]:
                return 1e9
            if Pc_val < photo_Pc_bounds[0] or Pc_val > photo_Pc_bounds[1]:
                return 1e9
            eff_photo_model = photo_model
        else:
            eff_photo_model = photo_model

        mech_loss = mech_loss_for_genotype(
            g=g, alpha_val=a, beta_val=b,
            temp_base=temp_base, temp_optimal=t,
            temp_critical=temp_critical,
            plantings_of_g_fn=plantings_of_g_fn,
            T_dict=T_dict, DAP_obs_dict=active_DAP_obs[0],
            use_penalty=False,
            normalize_loss=normalize_loss,
            DL_dict=DL_dict, photo_B=B_val, photo_Pc=Pc_val,
            photo_type=photo_type, photo_model=eff_photo_model,
            photo_a=a_photo_val, photo_b=b_photo_val, photo_d=d_photo_val,
            photo_n_min=photo_n_min, photo_n_opt=photo_n_opt,
        )
        if use_penalty:
            penalty = lambda_a * (a - alpha_target) ** 2 + lambda_b * (b - beta_target) ** 2
            if fit_topt:
                penalty += lambda_t * (t - topt_target) ** 2
            if fit_photo:
                penalty += lambda_bphoto * (B_val - photo_B_target) ** 2
                penalty += lambda_pc * (Pc_val - photo_Pc_target) ** 2
            if fit_photo3:
                penalty += lambda_photo_a * (a_photo_val - photo_a_target) ** 2
                penalty += lambda_photo_b * (b_photo_val - photo_b_target) ** 2
                penalty += lambda_photo_d * (d_photo_val - photo_d_target) ** 2
            total_loss = mech_loss + penalty
        else:
            total_loss = mech_loss

        recorded.append((a, b, t, B_val or 0.0, Pc_val or 0.0,
                         a_photo_val or 0.0, b_photo_val or 0.0, d_photo_val or 0.0,
                         total_loss, mech_loss))
        return total_loss

    bounds = [alpha_bounds, beta_bounds]
    if fit_topt:
        bounds.append(topt_bounds)
    if fit_photo3:
        bounds.extend([photo_a_bounds, photo_b_bounds, photo_d_bounds])
    elif fit_photo:
        bounds.extend([photo_B_bounds, photo_Pc_bounds])
    minimizer_kwargs = {
        "method": "Nelder-Mead",
        "options": {
            "maxiter": local_max_iter,
            "xatol": 1e-4,
            "fatol": 1e-6,
            "adaptive": True,
        },
    }

    # Per-restart MAP collection when perturbation is active
    use_per_restart_maps = obs_perturbation_sd > 0
    restart_maps: List[Tuple[float, float, float, float, float, float, float]] = []
    eff_perturb_maxiter = perturb_maxiter if perturb_maxiter > 0 else min(maxiter, max(50, maxiter // 5))

    for r in range(max(1, int(n_restarts))):
        # Perturb observations for restarts > 0 (restart 0 uses original data)
        if obs_perturbation_sd > 0 and r > 0:
            perturbed = {}
            if g in DAP_obs_dict:
                perturbed[g] = {}
                for e, dap_val in DAP_obs_dict[g].items():
                    noise = rng_perturb.normal(0, obs_perturbation_sd)
                    perturbed[g][e] = max(1, int(round(dap_val + noise)))
            active_DAP_obs[0] = perturbed
        else:
            active_DAP_obs[0] = DAP_obs_dict

        # Track start index for per-restart MAP extraction
        restart_start_idx = len(recorded)

        # Use reduced maxiter for perturbed restarts (only need MAP)
        iter_this_restart = eff_perturb_maxiter if (use_per_restart_maps and r > 0) else maxiter

        seed_r = None if seed is None else int(seed) + r
        try:
            dual_annealing(
                func=objective,
                bounds=bounds,
                maxiter=iter_this_restart,
                minimizer_kwargs=minimizer_kwargs,
                seed=seed_r,
            )
        except TypeError:
            dual_annealing(
                func=objective,
                bounds=bounds,
                maxiter=iter_this_restart,
                minimizer_kwargs=minimizer_kwargs,
            )

        # Extract best point from this restart
        if use_per_restart_maps:
            restart_entries = recorded[restart_start_idx:]
            if restart_entries:
                best_entry = min(restart_entries, key=lambda x: x[8])
                restart_maps.append(best_entry)

    # Reset to original observations
    active_DAP_obs[0] = DAP_obs_dict

    if not recorded:
        result = {
            "alpha_samples": np.array([]),
            "beta_samples": np.array([]),
            "loss_samples": np.array([]),
            "alpha_mean": float("nan"),
            "beta_mean": float("nan"),
            "alpha_std": float("nan"),
            "beta_std": float("nan"),
            "alpha_map": float("nan"),
            "beta_map": float("nan"),
            "map_loss": float("nan"),
            "n_accepted": 0,
            "n_proposals": 0,
            "accept_threshold": float("nan"),
            "n_training_plantings": n_plantings,
        }
        if fit_topt:
            result["topt_samples"] = np.array([])
            result["topt_mean"] = float("nan")
            result["topt_std"] = float("nan")
            result["topt_map"] = float("nan")
        if fit_photo:
            result["photo_B_samples"] = np.array([])
            result["photo_Pc_samples"] = np.array([])
            result["photo_B_mean"] = float("nan")
            result["photo_Pc_mean"] = float("nan")
            result["photo_B_std"] = float("nan")
            result["photo_Pc_std"] = float("nan")
            result["photo_B_map"] = float("nan")
            result["photo_Pc_map"] = float("nan")
        if fit_photo3:
            for k in ("photo_a", "photo_b", "photo_d"):
                result[f"{k}_samples"] = np.array([])
                result[f"{k}_mean"] = float("nan")
                result[f"{k}_std"] = float("nan")
                result[f"{k}_map"] = float("nan")
        return result

    if use_per_restart_maps and restart_maps:
        # Per-restart MAP collection: re-evaluate ALL MAPs on original data
        reeval_maps = []
        for a, b, t, B_r, Pc_r, a_r, b_r, d_r, _penalized, _mech in restart_maps:
            eff_model = "logistic3" if fit_photo3 else photo_model
            orig_mech = mech_loss_for_genotype(
                g=g, alpha_val=a, beta_val=b,
                temp_base=temp_base, temp_optimal=t if fit_topt else temp_optimal,
                temp_critical=temp_critical,
                plantings_of_g_fn=plantings_of_g_fn,
                T_dict=T_dict, DAP_obs_dict=DAP_obs_dict,
                use_penalty=False, normalize_loss=normalize_loss,
                DL_dict=DL_dict,
                photo_B=B_r if fit_photo else photo_B,
                photo_Pc=Pc_r if fit_photo else photo_Pc,
                photo_type=photo_type, photo_model=eff_model,
                photo_a=a_r if fit_photo3 else None,
                photo_b=b_r if fit_photo3 else None,
                photo_d=d_r if fit_photo3 else None,
            )
            if use_penalty:
                orig_pen = lambda_a * (a - alpha_target) ** 2 + lambda_b * (b - beta_target) ** 2
                if fit_topt:
                    orig_pen += lambda_t * (t - topt_target) ** 2
                if fit_photo:
                    orig_pen += lambda_bphoto * (B_r - photo_B_target) ** 2
                    orig_pen += lambda_pc * (Pc_r - photo_Pc_target) ** 2
                if fit_photo3:
                    orig_pen += lambda_photo_a * (a_r - photo_a_target) ** 2
                    orig_pen += lambda_photo_b * (b_r - photo_b_target) ** 2
                    orig_pen += lambda_photo_d * (d_r - photo_d_target) ** 2
                orig_total = orig_mech + orig_pen
            else:
                orig_total = orig_mech
            reeval_maps.append((a, b, t, B_r, Pc_r, a_r, b_r, d_r, orig_total, orig_mech))

        # Sort by original-data loss, take top_k
        reeval_maps.sort(key=lambda x: x[8])
        top = reeval_maps[:top_k]

        top_alphas = np.array([x[0] for x in top])
        top_betas = np.array([x[1] for x in top])
        top_topts = np.array([x[2] for x in top])
        top_Bs = np.array([x[3] for x in top])
        top_Pcs = np.array([x[4] for x in top])
        top_pa = np.array([x[5] for x in top])
        top_pb = np.array([x[6] for x in top])
        top_pd = np.array([x[7] for x in top])
        top_losses = np.array([x[8] for x in top])
        top_mech_losses = np.array([x[9] for x in top])

        best_a, best_b, best_t, best_B, best_Pc, best_pa_v, best_pb_v, best_pd_v, best_loss, best_mech = top[0]
    else:
        # Standard global pool: deduplicate, sort, take top-K
        seen = {}
        for rec in recorded:
            a, b, t, B_r, Pc_r, a3, b3, d3, total_loss, mech_loss = rec
            key_parts = [round(a, 6), round(b, 6)]
            if fit_topt:
                key_parts.append(round(t, 6))
            if fit_photo:
                key_parts.extend([round(B_r, 6), round(Pc_r, 6)])
            if fit_photo3:
                key_parts.extend([round(a3, 6), round(b3, 6), round(d3, 6)])
            key = tuple(key_parts)
            if key not in seen or total_loss < seen[key][8]:
                seen[key] = rec

        unique = sorted(seen.values(), key=lambda x: x[8])  # sort by penalized loss
        top = unique[:top_k]

        top_alphas = np.array([x[0] for x in top])
        top_betas = np.array([x[1] for x in top])
        top_topts = np.array([x[2] for x in top])
        top_Bs = np.array([x[3] for x in top])
        top_Pcs = np.array([x[4] for x in top])
        top_pa = np.array([x[5] for x in top])
        top_pb = np.array([x[6] for x in top])
        top_pd = np.array([x[7] for x in top])
        top_losses = np.array([x[8] for x in top])
        top_mech_losses = np.array([x[9] for x in top])

        best_a, best_b, best_t, best_B, best_Pc, best_pa_v, best_pb_v, best_pd_v, best_loss, best_mech = top[0]

    result = {
        "alpha_samples": top_alphas,
        "beta_samples": top_betas,
        "loss_samples": top_mech_losses,  # pure mech loss for downstream compatibility
        "alpha_mean": float(np.mean(top_alphas)),
        "beta_mean": float(np.mean(top_betas)),
        "alpha_std": float(np.std(top_alphas)),
        "beta_std": float(np.std(top_betas)),
        "alpha_map": best_a,
        "beta_map": best_b,
        "map_loss": best_mech,  # report pure mech loss as MAP loss
        "n_accepted": len(top_alphas),
        "n_proposals": len(recorded),
        "accept_threshold": float(top_losses[-1]) if len(top_losses) > 0 else float("nan"),
        "n_training_plantings": n_plantings,
    }
    if fit_topt:
        result["topt_samples"] = top_topts
        result["topt_mean"] = float(np.mean(top_topts))
        result["topt_std"] = float(np.std(top_topts))
        result["topt_map"] = best_t
    if fit_photo:
        result["photo_B_samples"] = top_Bs
        result["photo_Pc_samples"] = top_Pcs
        result["photo_B_mean"] = float(np.mean(top_Bs))
        result["photo_Pc_mean"] = float(np.mean(top_Pcs))
        result["photo_B_std"] = float(np.std(top_Bs))
        result["photo_Pc_std"] = float(np.std(top_Pcs))
        result["photo_B_map"] = best_B
        result["photo_Pc_map"] = best_Pc
    if fit_photo3:
        result["photo_a_samples"] = top_pa
        result["photo_b_samples"] = top_pb
        result["photo_d_samples"] = top_pd
        result["photo_a_mean"] = float(np.mean(top_pa))
        result["photo_b_mean"] = float(np.mean(top_pb))
        result["photo_d_mean"] = float(np.mean(top_pd))
        result["photo_a_std"] = float(np.std(top_pa))
        result["photo_b_std"] = float(np.std(top_pb))
        result["photo_d_std"] = float(np.std(top_pd))
        result["photo_a_map"] = best_pa_v
        result["photo_b_map"] = best_pb_v
        result["photo_d_map"] = best_pd_v
    return result


# ============================================================
# Top-K DA for photoperiod only (3D: a, b, d)
# ============================================================
def topk_da_photo3(
    g: str,
    plantings_of_g_fn: Callable[[str], Sequence[str]],
    DL_dict: Dict[str, Dict[str, np.ndarray]],
    DAP_obs_dict: Dict[str, Dict[str, int]],
    photo_a: float = 1.0,
    photo_b_bounds: Tuple[float, float] = (-2.0, 0.0),
    photo_d_bounds: Tuple[float, float] = (8.0, 18.0),
    top_k: int = 100,
    maxiter: int = 500,
    local_max_iter: int = 150,
    seed: Optional[int] = 42,
    n_restarts: int = 1,
    normalize_loss: bool = False,
    # GBLUP penalty
    use_penalty: bool = False,
    photo_b_target: float = 0.0,
    photo_d_target: float = 0.0,
    lambda_photo_b: float = 0.0,
    lambda_photo_d: float = 0.0,
    # Observation perturbation
    obs_perturbation_sd: float = 0.0,
    perturb_maxiter: int = 0,
    # Legacy compat — accepted but ignored
    **kwargs,
) -> Dict:
    """Top-K dual annealing for photoperiod params (b, d) using pure daylength loss.

    Optimizes logistic3 photoperiod parameters per genotype using ONLY daylength
    information — no temperature model involved. The `a` parameter (floor) is fixed
    from config.

    Loss = variance of cumulative photo progress across plantings.
    Same principle as the thermal loss but operating on daylength alone.

    Optimization is 2D: (b, d). Fast and decoupled from temperature fitting.

    Returns dict with photo_b/d samples, means, stds, MAP values, plus fixed photo_a.
    """
    from scipy.optimize import dual_annealing

    n_plantings = len(list(plantings_of_g_fn(g)))
    fixed_a = photo_a

    # Recording: (b_photo, d_photo, penalized_loss, pure_loss)
    recorded: List[Tuple[float, float, float, float]] = []

    active_DAP_obs = [DAP_obs_dict]
    rng_perturb = np.random.RandomState(seed if seed is not None else 42)

    def objective(xvec):
        b_val = float(xvec[0])
        d_val = float(xvec[1])
        if (b_val < photo_b_bounds[0] or b_val > photo_b_bounds[1] or
            d_val < photo_d_bounds[0] or d_val > photo_d_bounds[1]):
            return 1e9
        pure_loss = photo_loss_for_genotype(
            g=g, photo_b=b_val, photo_d=d_val, photo_a=fixed_a,
            plantings_of_g_fn=plantings_of_g_fn,
            DL_dict=DL_dict, DAP_obs_dict=active_DAP_obs[0],
            normalize_loss=normalize_loss,
        )
        if use_penalty:
            penalty = (lambda_photo_b * (b_val - photo_b_target) ** 2 +
                       lambda_photo_d * (d_val - photo_d_target) ** 2)
            total_loss = pure_loss + penalty
        else:
            total_loss = pure_loss
        recorded.append((b_val, d_val, total_loss, pure_loss))
        return total_loss

    bounds = [photo_b_bounds, photo_d_bounds]
    minimizer_kwargs = {
        "method": "Nelder-Mead",
        "options": {"maxiter": local_max_iter, "xatol": 1e-4, "fatol": 1e-6, "adaptive": True},
    }

    use_per_restart_maps = obs_perturbation_sd > 0
    restart_maps: List[Tuple[float, float, float, float, float]] = []
    eff_perturb_maxiter = perturb_maxiter if perturb_maxiter > 0 else min(maxiter, max(50, maxiter // 5))

    eff_n_restarts = max(1, n_restarts)
    for r_idx in range(eff_n_restarts):
        r_seed = (seed + r_idx * 1000) if seed is not None else None

        # Perturb observations for diversity (r > 0)
        if use_per_restart_maps and r_idx > 0:
            perturbed = {}
            for gid_k, env_map in DAP_obs_dict.items():
                perturbed[gid_k] = {}
                for env_k, dap_v in env_map.items():
                    noise = rng_perturb.normal(0, obs_perturbation_sd)
                    perturbed[gid_k][env_k] = max(1, int(round(dap_v + noise)))
                perturbed[gid_k] = perturbed[gid_k]
            active_DAP_obs[0] = perturbed
            eff_maxiter = eff_perturb_maxiter
        else:
            active_DAP_obs[0] = DAP_obs_dict
            eff_maxiter = maxiter

        try:
            dual_annealing(
                objective, bounds, maxiter=eff_maxiter, seed=r_seed,
                minimizer_kwargs=minimizer_kwargs,
            )
        except Exception:
            pass

        # Collect per-restart MAP
        if use_per_restart_maps and recorded:
            best_rec = min(recorded, key=lambda x: x[3])
            restart_maps.append(best_rec)

    if not recorded:
        return {
            "photo_a_map": fixed_a, "photo_b_map": float("nan"), "photo_d_map": float("nan"),
            "photo_a_mean": fixed_a, "photo_b_mean": float("nan"), "photo_d_mean": float("nan"),
            "photo_a_std": 0.0, "photo_b_std": float("nan"), "photo_d_std": float("nan"),
            "photo_b_samples": np.array([]), "photo_d_samples": np.array([]),
            "loss_samples": np.array([]),
            "map_loss": float("nan"),
            "n_accepted": 0, "n_proposals": 0,
            "n_training_plantings": n_plantings,
        }

    if use_per_restart_maps and restart_maps:
        # Re-evaluate all restart MAPs on original data
        reeval = []
        for b_r, d_r, _pen, _pure in restart_maps:
            orig_loss = photo_loss_for_genotype(
                g=g, photo_b=b_r, photo_d=d_r, photo_a=fixed_a,
                plantings_of_g_fn=plantings_of_g_fn,
                DL_dict=DL_dict, DAP_obs_dict=DAP_obs_dict,
                normalize_loss=normalize_loss,
            )
            if use_penalty:
                orig_pen = (lambda_photo_b * (b_r - photo_b_target) ** 2 +
                            lambda_photo_d * (d_r - photo_d_target) ** 2)
                orig_total = orig_loss + orig_pen
            else:
                orig_total = orig_loss
            reeval.append((b_r, d_r, orig_total, orig_loss))
        reeval.sort(key=lambda x: x[2])
        top = reeval[:top_k]
    else:
        # Deduplicate and take top-K
        seen = {}
        for rec in recorded:
            key = (round(rec[0], 6), round(rec[1], 6))
            if key not in seen or rec[2] < seen[key][2]:
                seen[key] = rec
        unique = sorted(seen.values(), key=lambda x: x[2])
        top = unique[:top_k]

    top_b = np.array([x[0] for x in top])
    top_d = np.array([x[1] for x in top])
    top_losses = np.array([x[2] for x in top])
    top_pure = np.array([x[3] for x in top])

    best = top[0]
    return {
        "photo_a_map": fixed_a,
        "photo_b_map": best[0],
        "photo_d_map": best[1],
        "photo_a_mean": fixed_a,
        "photo_b_mean": float(np.mean(top_b)),
        "photo_d_mean": float(np.mean(top_d)),
        "photo_a_std": 0.0,
        "photo_b_std": float(np.std(top_b)),
        "photo_d_std": float(np.std(top_d)),
        "photo_b_samples": top_b,
        "photo_d_samples": top_d,
        "loss_samples": top_pure,
        "map_loss": best[3],
        "n_accepted": len(top),
        "n_proposals": len(recorded),
        "n_training_plantings": n_plantings,
    }


def posterior_predictive_dap(
    g: str,
    posterior: Dict,
    predict_plantings: List[str],
    T_dict: Dict[str, Dict[str, np.ndarray]],
    DAP_obs_dict: Dict[str, Dict[str, int]],
    plantings_of_g_fn: Callable[[str], Sequence[str]],
    temp_base: float,
    temp_optimal: float,
    temp_critical: float,
    DL_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    photo_B: Optional[float] = None,
    photo_Pc: Optional[float] = None,
    photo_type: str = "short_day",
    photo_model: str = "sinclair",
    max_samples: int = 200,
    # Linear-plateau (Grimm 1993) photoperiod params (population-level, fixed)
    photo_n_min: Optional[float] = None,
    photo_n_opt: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute posterior predictive DAP distribution for prediction plantings.

    For each accepted posterior sample (alpha_s, beta_s):
      1. Compute H_vals across training plantings -> target_H_s = mean(H_vals)
      2. For each prediction planting, predict_flowering_dap -> DAP_s

    Parameters
    ----------
    max_samples : int
        Cap on posterior samples used (subsampled if n_accepted > max_samples).

    Returns
    -------
    dict mapping planting name -> {
        "mean": float,    # posterior predictive mean DAP
        "std": float,     # posterior predictive std DAP
        "median": float,  # posterior predictive median DAP
        "samples": list,  # raw DAP sample values
    }
    """
    alpha_samples = posterior["alpha_samples"]
    beta_samples = posterior["beta_samples"]
    topt_samples = posterior.get("topt_samples")  # None if not fitting Topt
    photo_B_samples = posterior.get("photo_B_samples")  # None if not fitting photo
    photo_Pc_samples = posterior.get("photo_Pc_samples")
    photo_a_samples = posterior.get("photo_a_samples")  # None if not fitting logistic3
    photo_b_samples = posterior.get("photo_b_samples")
    photo_d_samples = posterior.get("photo_d_samples")
    n_accepted = len(alpha_samples)

    if n_accepted == 0 or not predict_plantings:
        return {}

    # Subsample if too many accepted
    if n_accepted > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_accepted, size=max_samples, replace=False)
        alpha_samples = alpha_samples[idx]
        beta_samples = beta_samples[idx]
        if topt_samples is not None:
            topt_samples = topt_samples[idx]
        if photo_B_samples is not None:
            photo_B_samples = photo_B_samples[idx]
        if photo_Pc_samples is not None:
            photo_Pc_samples = photo_Pc_samples[idx]
        if photo_a_samples is not None:
            photo_a_samples = photo_a_samples[idx]
        if photo_b_samples is not None:
            photo_b_samples = photo_b_samples[idx]
        if photo_d_samples is not None:
            photo_d_samples = photo_d_samples[idx]
        n_accepted = max_samples

    # For each sample, compute target_H and predict DAP
    result: Dict[str, Dict[str, float]] = {}
    # Initialize collectors per planting
    dap_collectors: Dict[str, List[int]] = {e: [] for e in predict_plantings}

    for s in range(n_accepted):
        a_s = float(alpha_samples[s])
        b_s = float(beta_samples[s])
        t_s = float(topt_samples[s]) if topt_samples is not None else temp_optimal
        B_s = float(photo_B_samples[s]) if photo_B_samples is not None else photo_B
        Pc_s = float(photo_Pc_samples[s]) if photo_Pc_samples is not None else photo_Pc
        pa_s = float(photo_a_samples[s]) if photo_a_samples is not None else None
        pb_s = float(photo_b_samples[s]) if photo_b_samples is not None else None
        pd_s = float(photo_d_samples[s]) if photo_d_samples is not None else None

        # Compute target_H_s from training plantings
        H_vals = cumulative_progress_by_planting(
            g=g, alpha=a_s, beta=b_s,
            temp_base=temp_base, temp_optimal=t_s,
            temp_critical=temp_critical,
            plantings_of_g_fn=plantings_of_g_fn,
            T_dict=T_dict, DAP_obs_dict=DAP_obs_dict,
            DL_dict=DL_dict, photo_B=B_s, photo_Pc=Pc_s,
            photo_type=photo_type, photo_model=photo_model,
            photo_a=pa_s, photo_b=pb_s, photo_d=pd_s,
            photo_n_min=photo_n_min, photo_n_opt=photo_n_opt,
        )
        if not H_vals:
            continue
        target_H_s = float(np.mean(list(H_vals.values())))
        if target_H_s <= 0:
            continue

        # Predict DAP for each prediction planting
        for pred_e in predict_plantings:
            if g not in T_dict or pred_e not in T_dict[g]:
                continue
            temps_pred = T_dict[g][pred_e]
            dl_pred = None
            if DL_dict and g in DL_dict and pred_e in DL_dict[g]:
                dl_pred = DL_dict[g][pred_e]

            dap_s = predict_flowering_dap(
                temps=temps_pred,
                target_H=target_H_s,
                alpha=a_s,
                beta=b_s,
                temp_base=temp_base,
                temp_optimal=t_s,
                temp_critical=temp_critical,
                daylengths=dl_pred,
                photo_B=B_s,
                photo_Pc=Pc_s,
                photo_type=photo_type,
                photo_model=photo_model,
                fractional=True,  # Sub-day precision for variance estimation
                extrapolate=True,  # Extrapolate when weather window is too short
                photo_a=pa_s,
                photo_b=pb_s,
                photo_d=pd_s,
                photo_n_min=photo_n_min,
                photo_n_opt=photo_n_opt,
            )
            if dap_s > 0:  # -1 means not reached
                dap_collectors[pred_e].append(dap_s)

    # Summarize per planting
    for pred_e in predict_plantings:
        samples = dap_collectors[pred_e]
        if samples:
            arr = np.array(samples, dtype=float)
            result[pred_e] = {
                "mean": round(float(np.mean(arr))),   # Integer DAP for display
                "std": float(np.std(arr)),             # Fractional precision for variance
                "median": round(float(np.median(arr))),  # Integer DAP for display
                "samples": samples,
            }

    return result


# ============================================================
# Data loading utilities (CSV -> dicts)
# ============================================================
def _load_weather_df(
    path: str,
    weather_date_col: str = "Period",
    tmax_col: str = "tmx",
    tmin_col: str = "tmn",
) -> pd.DataFrame:
    """Load a single weather CSV and return it indexed by date with tmean computed."""
    wx = pd.read_csv(path)
    wx[weather_date_col] = pd.to_datetime(wx[weather_date_col])
    wx["tmean"] = (wx[tmax_col].astype(float) + wx[tmin_col].astype(float)) / 2.0
    wx = wx.sort_values(weather_date_col).set_index(weather_date_col)
    return wx


def load_and_build_dicts(
    phenotypes_csv: str,
    weather_csv: str,
    *,
    genotype_col: str = "id",
    planting_col: str = "Planting",
    ft_col: str = "ft",
    censored_col: str = "censored",
    start_col: str = "Start_Date",
    end_col: str = "End_Date",
    weather_date_col: str = "Period",
    tmax_col: str = "tmx",
    tmin_col: str = "tmn",
    latitude_map: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, pd.Series]], Dict[str, Dict[str, int]], Callable[[str], List[str]], Optional[Dict[str, Dict[str, np.ndarray]]]]:
    """
    Load phenotypes + weather and build:

      T_dict[g][planting]       = pandas Series of daily mean temps (index = dates)
      DAP_obs_dict[g][planting] = observed ft (DAP) for uncensored rows
      DL_dict[g][planting]      = numpy array of daylengths (hours), or None if no latitude

    Weather modes:
    - **Single file**: weather_csv points to a .csv file — all observations use this file.
    - **Multi-location directory**: weather_csv points to a directory — loads
      ``{dir}/{Planting}/weather.csv`` per unique Planting value.

    Daylength is computed from latitude + day-of-year when latitude_map is provided.
    If the weather data already contains a 'DAYLhr' column, that is used instead.

    Notes
    -----
    - Filters to rows where censored==0 and ft is present.
    - Daily mean temperature is computed as (tmx + tmn)/2.
    """
    from pathlib import Path as _Path

    pheno = pd.read_csv(phenotypes_csv)

    # Parse dates
    pheno[start_col] = pd.to_datetime(pheno[start_col])
    pheno[end_col] = pd.to_datetime(pheno[end_col])

    weather_path = _Path(weather_csv)
    multi_location = weather_path.is_dir()

    if multi_location:
        # Multi-location mode: load one weather file per unique Planting value
        wx_dict: Dict[str, pd.DataFrame] = {}
        for location in pheno[planting_col].unique():
            loc_file = weather_path / str(location) / "weather.csv"
            if not loc_file.exists():
                raise FileNotFoundError(
                    f"No weather file for location '{location}' at {loc_file}. "
                    f"Create the folder and place weather.csv inside it."
                )
            wx_dict[str(location)] = _load_weather_df(
                str(loc_file), weather_date_col, tmax_col, tmin_col
            )
        print(f"Loaded weather for {len(wx_dict)} locations: {sorted(wx_dict.keys())}")
    else:
        # Single-file mode (original behavior)
        wx_single = _load_weather_df(str(weather_path), weather_date_col, tmax_col, tmin_col)

    def _get_wx(planting_name: str) -> pd.DataFrame:
        if multi_location:
            return wx_dict[planting_name]
        return wx_single

    def _get_daylength_series(wx: pd.DataFrame, start, planting_name: str) -> Optional[np.ndarray]:
        """Extract or compute daylength array for weather from start onward."""
        sub = wx.loc[start:]
        if sub.empty:
            return None
        # If weather already has daylength column, use it
        if "DAYLhr" in wx.columns:
            return sub["DAYLhr"].values.astype(float)
        # Compute from latitude if available
        if latitude_map and planting_name in latitude_map:
            lat = latitude_map[planting_name]
            doys = sub.index.dayofyear
            return compute_daylength_series(doys, lat)
        return None

    # Filter: only uncensored + observed ft
    pheno_obs = pheno[(pheno[censored_col] == 0) & (pheno[ft_col].notna())].copy()
    if pheno_obs.empty:
        raise ValueError("No rows left after filtering censored==0 and ft notna().")

    # Average reps: when multiple rows exist for the same genotype×location,
    # average the ft (DAP) values and use the first row's start date.
    n_before = len(pheno_obs)
    pheno_obs = (
        pheno_obs.groupby([genotype_col, planting_col], sort=False)
        .agg({ft_col: "mean", start_col: "first", end_col: "first", censored_col: "first"})
        .reset_index()
    )
    n_after = len(pheno_obs)
    if n_after < n_before:
        print(f"Averaged reps: {n_before} observations → {n_after} genotype×location means")

    T_dict: Dict[str, Dict[str, pd.Series]] = {}
    DAP_obs_dict: Dict[str, Dict[str, int]] = {}
    DL_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None

    for _, row in pheno_obs.iterrows():
        g = str(row[genotype_col])
        e = str(row[planting_col])
        dap = round(float(row[ft_col]))
        start = row[start_col]
        end = row[end_col]

        wx = _get_wx(e)
        # Use full weather window from planting date onward (not just start:end)
        # so predict_flowering_dap() has enough data when predicted DAP > observed DAP.
        # Training loss functions already clip to observed DAP via dap_cut.
        temps_series = wx.loc[start:, "tmean"]
        if temps_series.size == 0:
            raise ValueError(f"No weather found for {g}, {e}, from {start.date()} onward")

        # store pandas Series (preserves dates in the index)
        T_dict.setdefault(g, {})[e] = temps_series
        DAP_obs_dict.setdefault(g, {})[e] = dap

        # Daylength
        dl = _get_daylength_series(wx, start, e)
        if dl is not None:
            if DL_dict is None:
                DL_dict = {}
            DL_dict.setdefault(g, {})[e] = dl

    # Second pass: load weather for ALL plantings (including censored) so
    # prediction plantings have weather data available even when they weren't
    # used for training.  Only adds entries not already present in T_dict.
    # Use the full available weather window (from planting date to end of
    # weather data) so that predict_flowering_dap() has enough data even
    # when predicted DAP exceeds the observed flowering time.
    for _, row in pheno.iterrows():
        g = str(row[genotype_col])
        e = str(row[planting_col])
        start = row[start_col]
        T_dict.setdefault(g, {})
        if e not in T_dict[g]:
            wx = _get_wx(e)
            temps_series = wx.loc[start:, "tmean"]
            if temps_series.size > 0:
                T_dict[g][e] = temps_series
                # Daylength for prediction plantings too
                dl = _get_daylength_series(wx, start, e)
                if dl is not None:
                    if DL_dict is None:
                        DL_dict = {}
                    DL_dict.setdefault(g, {})[e] = dl

    def plantings_of_g_fn(gid: str) -> List[str]:
        return list(DAP_obs_dict.get(str(gid), {}).keys())

    return pheno_obs, T_dict, DAP_obs_dict, plantings_of_g_fn, DL_dict


def cumulative_progress_by_planting(
    g: str,
    alpha: float,
    beta: float,
    temp_base: float,
    temp_optimal: float,
    temp_critical: float,
    plantings_of_g_fn: Callable[[str], Sequence[str]],
    T_dict: Dict[str, Dict[str, np.ndarray]],
    DAP_obs_dict: Dict[str, Dict[str, int]],
    DL_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    photo_B: Optional[float] = None,
    photo_Pc: Optional[float] = None,
    photo_type: str = "short_day",
    photo_model: str = "sinclair",
    # 3-param logistic (Messina) photoperiod params
    photo_a: Optional[float] = None,
    photo_b: Optional[float] = None,
    photo_d: Optional[float] = None,
    # Linear-plateau (Grimm 1993) photoperiod params
    photo_n_min: Optional[float] = None,
    photo_n_opt: Optional[float] = None,
) -> Dict[str, float]:
    """
    Convenience function to compute H_{g,e} (cumulative progress at flowering)
    for each planting e of genotype g, under a given (alpha,beta).
    """
    out: Dict[str, float] = {}
    for e in plantings_of_g_fn(g):
        dl_e = None
        if DL_dict is not None and g in DL_dict and e in DL_dict[g]:
            dl_e = DL_dict[g][e]
        rates = daily_development_rate(
            T_dict[g][e], alpha=alpha, beta=beta,
            temp_base=temp_base, temp_optimal=temp_optimal, temp_critical=temp_critical,
            daylengths=dl_e, photo_B=photo_B, photo_Pc=photo_Pc,
            photo_type=photo_type, photo_model=photo_model,
            photo_a=photo_a, photo_b=photo_b, photo_d=photo_d,
            photo_n_min=photo_n_min, photo_n_opt=photo_n_opt,
        )
        dap = int(DAP_obs_dict[g][e])
        dap_cut = max(0, min(dap, len(rates)))
        out[e] = float(np.sum(rates[:dap_cut]))
    return out


def loo_planting_variance(
    g: str,
    alpha: float,
    beta: float,
    temp_base: float,
    temp_optimal: float,
    temp_critical: float,
    T_dict: Dict[str, Dict[str, np.ndarray]],
    DAP_obs_dict: Dict[str, Dict[str, int]],
    training_plantings: Sequence[str],
    DL_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    photo_B: Optional[float] = None,
    photo_Pc: Optional[float] = None,
    photo_type: str = "short_day",
    photo_model: str = "sinclair",
    min_folds: int = 3,
    # 3-param logistic (Messina) photoperiod params
    photo_a: Optional[float] = None,
    photo_b: Optional[float] = None,
    photo_d: Optional[float] = None,
    # Linear-plateau (Grimm 1993) photoperiod params
    photo_n_min: Optional[float] = None,
    photo_n_opt: Optional[float] = None,
) -> Optional[float]:
    """
    Leave-one-planting-out cross-validation variance for a single genotype.

    For each of K training plantings, compute H* from the other K-1 plantings
    (using the same fitted alpha, beta), predict DAP for the held-out planting,
    and collect the squared residual. Returns MSE of LOO residuals (days²).

    This is "cheap LOO" — alpha, beta are NOT re-fit per fold, only H* is
    recomputed from K-1 plantings. This is optimistically biased but captures
    per-genotype prediction quality at negligible computational cost.

    Returns None if the genotype has fewer than min_folds training plantings.
    """
    # Get training plantings that this genotype has data for
    if g not in T_dict or g not in DAP_obs_dict:
        return None
    avail = [e for e in training_plantings
             if e in T_dict[g] and e in DAP_obs_dict[g]]
    if len(avail) < min_folds:
        return None

    residuals = []
    for held_out in avail:
        # Compute H* from the K-1 other plantings
        other = [e for e in avail if e != held_out]
        H_vals = []
        for e in other:
            dl_e = None
            if DL_dict is not None and g in DL_dict and e in DL_dict[g]:
                dl_e = DL_dict[g][e]
            rates = daily_development_rate(
                T_dict[g][e], alpha=alpha, beta=beta,
                temp_base=temp_base, temp_optimal=temp_optimal,
                temp_critical=temp_critical,
                daylengths=dl_e, photo_B=photo_B, photo_Pc=photo_Pc,
                photo_type=photo_type, photo_model=photo_model,
                photo_a=photo_a, photo_b=photo_b, photo_d=photo_d,
                photo_n_min=photo_n_min, photo_n_opt=photo_n_opt,
            )
            dap = int(DAP_obs_dict[g][e])
            dap_cut = max(0, min(dap, len(rates)))
            H_vals.append(float(np.sum(rates[:dap_cut])))

        if not H_vals:
            continue
        target_H = float(np.mean(H_vals))
        if target_H <= 0:
            continue

        # Predict DAP for the held-out planting
        dl_ho = None
        if DL_dict is not None and g in DL_dict and held_out in DL_dict[g]:
            dl_ho = DL_dict[g][held_out]
        pred_dap = predict_flowering_dap(
            temps=T_dict[g][held_out],
            target_H=target_H,
            alpha=alpha, beta=beta,
            temp_base=temp_base, temp_optimal=temp_optimal,
            temp_critical=temp_critical,
            daylengths=dl_ho, photo_B=photo_B, photo_Pc=photo_Pc,
            photo_type=photo_type, photo_model=photo_model,
            fractional=True,
            photo_a=photo_a, photo_b=photo_b, photo_d=photo_d,
            photo_n_min=photo_n_min, photo_n_opt=photo_n_opt,
        )
        if pred_dap < 0:
            # Target not reached — assign large penalty residual
            residuals.append(float(len(T_dict[g][held_out])))
            continue

        obs_dap = float(DAP_obs_dict[g][held_out])
        residuals.append(pred_dap - obs_dap)

    if len(residuals) < min_folds:
        return None

    # Return MSE of LOO residuals (days²)
    arr = np.array(residuals)
    return float(np.mean(arr ** 2))


def predict_flowering_dap(
    temps: Sequence[float],
    target_H: float,
    alpha: float,
    beta: float,
    temp_base: float,
    temp_optimal: float,
    temp_critical: float,
    daylengths: Optional[Sequence[float]] = None,
    photo_B: Optional[float] = None,
    photo_Pc: Optional[float] = None,
    photo_type: str = "short_day",
    photo_model: str = "sinclair",
    fractional: bool = False,
    extrapolate: bool = False,
    # 3-param logistic (Messina) photoperiod params
    photo_a: Optional[float] = None,
    photo_b: Optional[float] = None,
    photo_d: Optional[float] = None,
    # Linear-plateau (Grimm) photoperiod params
    photo_n_min: Optional[float] = None,
    photo_n_opt: Optional[float] = None,
) -> float:
    """
    Predict the flowering DAP given weather temps and a target cumulative progress H*.

    Accumulates daily development rates (cardinal_beta × photoperiod)
    until cumulative progress >= target_H.

    If fractional=False (default), returns integer DAP (1-based day count).
    If fractional=True, returns interpolated fractional DAP for sub-day precision
    (used for variance computation in posterior predictive sampling).

    If extrapolate=True and target_H is not reached within available weather,
    estimates DAP by projecting from the average daily rate over the last 14 days
    (or all days if fewer than 14). Returns -1 only if mean rate is effectively zero.

    Returns -1 if target_H is not reached and extrapolation is disabled or infeasible.
    """
    rates = daily_development_rate(
        temps,
        alpha=alpha,
        beta=beta,
        temp_base=temp_base,
        temp_optimal=temp_optimal,
        temp_critical=temp_critical,
        daylengths=daylengths,
        photo_B=photo_B,
        photo_Pc=photo_Pc,
        photo_type=photo_type,
        photo_model=photo_model,
        photo_a=photo_a,
        photo_b=photo_b,
        photo_d=photo_d,
        photo_n_min=photo_n_min,
        photo_n_opt=photo_n_opt,
    )
    cumsum = np.cumsum(rates)
    indices = np.where(cumsum >= target_H)[0]
    if len(indices) == 0:
        if not extrapolate:
            return -1
        # Extrapolate: estimate remaining days from recent average rate
        n = len(rates)
        if n == 0:
            return -1
        tail = min(14, n)
        avg_rate = float(rates[-tail:].mean())
        if avg_rate < 1e-10:
            return -1
        deficit = target_H - cumsum[-1]
        extra_days = deficit / avg_rate
        # Sanity cap: if extrapolation needs more days than the weather window
        # itself, the model effectively says "this environment can't flower
        # this genotype" — return -1 so callers fall back to temp-only or
        # mark the prediction as failed. Without this cap, an avg_rate of
        # ~1e-4 (small but above the 1e-10 threshold) produces astronomical
        # extra_days values (3000+ days for a 120-day weather window),
        # polluting downstream predictions with absurd DAPs.
        if extra_days > n:
            return -1
        if fractional:
            return float(n) + extra_days
        return int(n + extra_days) + 1

    day_idx = indices[0]
    if not fractional:
        return int(day_idx) + 1  # DAP is count of days (1-based)

    # Interpolate to fractional day for sub-day precision
    if day_idx == 0:
        # Crossed on first day — interpolate from 0 to cumsum[0]
        c_curr = cumsum[0]
        frac = target_H / c_curr if c_curr > 0 else 1.0
        return frac
    else:
        c_prev = cumsum[day_idx - 1]
        c_curr = cumsum[day_idx]
        denom = c_curr - c_prev
        frac = (target_H - c_prev) / denom if denom > 0 else 1.0
        return float(day_idx) + frac


def daily_rates_for_planting(
    g: str,
    e: str,
    T_dict: Dict[str, Dict[str, pd.Series]],
    alpha: float,
    beta: float,
    temp_base: float,
    temp_optimal: float,
    temp_critical: float,
) -> List[Dict[str, object]]:
    """
    Return per-day debug rows for genotype g / planting e:
      [{'day_index': int, 'date': ISO-date-or-str-or-None, 'temp': float, 'rate': float}, ...]
    This lets you confirm the exact dates & temperatures used (matches weather.csv) and the
    daily rates computed by cardinal_beta.
    """
    try:
        temps_series = T_dict[g][e]
    except Exception:
        return []

    # ensure we have a pandas Series (or something with .index and values)
    if hasattr(temps_series, "values"):
        temps_arr = np.asarray(temps_series.values, dtype=float)
        idx = list(temps_series.index)
    else:
        # fallback to array-like without dates
        temps_arr = np.asarray(temps_series, dtype=float)
        idx = [None] * len(temps_arr)

    rates = cardinal_beta(
        temps_arr,
        temp_base=temp_base,
        temp_optimal=temp_optimal,
        temp_critical=temp_critical,
        alpha=alpha,
        beta=beta,
    )

    rows: List[Dict[str, object]] = []
    for i, (d, t, r) in enumerate(zip(idx, temps_arr, rates)):
        date_val = None
        if d is not None:
            # convert pandas Timestamp to ISO string; otherwise str(d)
            try:
                date_val = pd.Timestamp(d).isoformat()
            except Exception:
                date_val = str(d)
        rows.append({"day_index": i, "date": date_val, "temp": float(t), "rate": float(r)})
    return rows
