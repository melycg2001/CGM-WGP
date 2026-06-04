"""
main_fit.py

CLI runner for mechanistic (alpha, beta) fitting plus GBLUP and RN-GBLUP DAP
prediction. Joint CGM-WGP runs as an additional comparison method when
--run-joint is set.

Example:
  python -m cgm_wgp.main_fit \\
    --phenotypes pipeline/1_data/Broccoli/phenotypes_dated.csv \\
    --weather pipeline/1_data/weather/FL_Hasting/weather.csv \\
    --genotype FA002 \\
    --Tb 5 --Topt 20 --Tc 45 \\
    --maxiter 1000
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import os
import datetime
import multiprocessing
import numpy as np
import pandas as pd
from pathlib import Path

# Import path configuration
from cgm_wgp.config import (
    DATA_DIR, GMATRIX_DIR, RESULTS_OUTPUT_DIR,
    get_phenotypes_path, get_evd_path,
    create_run_dir, extract_location_from_weather,
    get_output_path,
)

from cgm_wgp.mech_fit import (
    load_and_build_dicts,
    fit_alpha_beta_for_genotype,
    cumulative_progress_by_planting,
    daily_rates_for_planting,
    predict_flowering_dap,
    cardinal_beta,
    daily_development_rate,
    posterior_predictive_dap,
    topk_dual_annealing,
    topk_da_photo3,
)


# ── Structured vs legacy output path routing ──────────────────────────────

# Mapping: (category, new_filename) for each old suffix.
# Used by _output_path() to route files to structured subdirs.
_OUTPUT_MAP = {
    "_run_config.json":                ("diagnostics",  "run_config.json"),
    "":                                ("params",      "fitted.csv"),
    "_photoperiod.csv":                ("params",      "photoperiod.csv"),
    "_predictions.csv":                ("predictions",  "mechanistic.csv"),
    "_posterior_predictions.csv":       ("predictions",  "posterior.csv"),
    "_gblup_dap_predictions.csv":      ("predictions",  "gblup.csv"),
    "_jarquin_predictions.csv":        ("predictions",  "jarquin.csv"),
    "_raw_mech_predictions.csv":       ("predictions",  "raw_mechanistic.csv"),
    "_joint_predictions.csv":          ("predictions",  "joint.csv"),
    "_joint_params.csv":               ("params",       "joint_params.csv"),
    "_convergence_history.csv":        ("diagnostics",  "convergence.csv"),
    "_iteration_detail.csv":           ("diagnostics",  "iteration_detail.csv"),
    "_metrics_summary.csv":            ("diagnostics",  "metrics.csv"),
    "_gblup_dap_metrics_summary.csv":  ("diagnostics",  "gblup_metrics.csv"),
    "_progress.csv":                   ("diagnostics",  "progress.csv"),
    "_validation_comparison.csv":      ("diagnostics",  "validation_comparison.csv"),
    "_debug_daily_rates.csv":          ("diagnostics",  "debug_daily_rates.csv"),
}


def _output_path(args, suffix: str) -> str:
    """Return the correct output path for a given file suffix.

    Structured mode (--run-dir set): routes to params/predictions/diagnostics subdirs.
    Legacy mode (--out only): uses the old alpha_beta_results_{suffix} pattern.
    """
    run_dir = getattr(args, "run_dir", None)
    if run_dir:
        cat, fname = _OUTPUT_MAP.get(suffix, ("diagnostics", suffix.lstrip("_")))
        return get_output_path(Path(run_dir), cat, fname)
    # Legacy fallback
    base = args.out.replace(".csv", "")
    return base + suffix


def parse_args():
    p = argparse.ArgumentParser(
        description="Mechanistic fit of cardinal-beta (alpha, beta) with GBLUP and "
                    "RN-GBLUP DAP prediction; optional Joint CGM-WGP via --run-joint."
    )
    p.add_argument("--phenotypes", required=True, help="Path to phenotypes CSV (interval-censored table)")
    p.add_argument("--test-phenotypes", default=None,
                   help="Path to held-out phenotypes CSV for validation. When provided, --phenotypes "
                        "is used for fitting only and --test-phenotypes is used to look up observed "
                        "DAPs for validation/metrics. Used by the balanced_training_subset CV scheme "
                        "to implement per-environment (genotype, planting) cell hold-out. Rows must "
                        "have the same columns as --phenotypes.")
    p.add_argument("--weather", required=True,
                   help="Path to weather CSV file OR weather directory. "
                        "Single file: all observations use one weather.csv. "
                        "Directory: loads {dir}/{Planting}/weather.csv per location.")
    p.add_argument("--genotype", default=None, help="Genotype id to fit (e.g., FA002). If omitted, all genotypes in the phenotypes file are fitted.")
    p.add_argument("--Tb", type=float, default=5.0, help="Base temperature Tb")
    p.add_argument("--Topt", type=float, default=20.0, help="Optimal temperature Topt")
    p.add_argument("--Tc", type=float, default=35.0, help="Critical temperature Tc")

    # Optimization knobs (DA phase)
    p.add_argument("--maxiter", type=int, default=2000, help="dual_annealing maxiter for MAP estimation (Phase 1)")
    p.add_argument("--local_max_iter", type=int, default=150, help="Nelder-Mead local polish maxiter")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--alpha-bounds", type=float, nargs=2, default=[0.5, 8.0], metavar=("LO", "HI"), help="Alpha parameter bounds (default: 0.5 8.0)")
    p.add_argument("--beta-bounds", type=float, nargs=2, default=[0.5, 8.0], metavar=("LO", "HI"), help="Beta parameter bounds (default: 0.5 8.0)")
    p.add_argument("--fit-topt", action="store_true",
                   help="Fit Topt per genotype as 3rd parameter alongside alpha, beta. "
                        "When enabled, each genotype gets its own optimal temperature.")
    p.add_argument("--topt-bounds", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="Topt parameter bounds when --fit-topt is enabled (default: Tb+5, Tc-5)")
    p.add_argument("--n-restarts", type=int, default=1, help="Number of dual_annealing restarts per genotype with different seeds (default: 1)")
    p.add_argument("--obs-perturbation-sd", type=float, default=0.0,
                   help="Observation noise std for perturbed restarts. Creates posterior diversity "
                        "by perturbing training DAP values for restarts > 0 (default: 0 = off)")
    p.add_argument("--obs-perturbation-sd-auto", action="store_true",
                   help="Auto-estimate obs perturbation SD from Round 1 training residuals. "
                        "Uses population median training RMSE as perturbation scale.")
    p.add_argument("--perturb-maxiter", type=int, default=0,
                   help="DA iterations for perturbed restarts (r > 0). "
                        "Default: 0 = auto (max(200, maxiter // 10)).")
    p.add_argument("--normalize-loss", action="store_true", help="Divide loss variance term by number of plantings (normalizes across genotypes)")

    # Posterior sampling knob (top-K dual annealing)
    p.add_argument("--max-posterior-samples", type=int, default=200,
                   help="Cap on posterior samples for DAP prediction (default: 200)")

    p.add_argument("--out", default=None, help="Output CSV path (columns: id,alpha,beta). If omitted, auto-creates a timestamped run folder.")
    p.add_argument("--run-dir", default=None, help="Structured run directory (created by app.py). When set, outputs go into params/predictions/diagnostics/ subdirs.")
    p.add_argument("--run-label", default=None, help="Human-readable label for this run (stored in meta.json).")
    p.add_argument("--crop", default="unknown", help="Crop name for output folder (e.g. 'lettuce')")

    p.add_argument("--workers", type=int, default=0,
                   help="Number of parallel workers for genotype fitting (0 = all cores minus 2, 1 = sequential)")

    p.add_argument(
        "--train-plantings",
        default=None,
        help=(
            "Comma-separated planting numbers to use for fitting alpha/beta "
            "(e.g. '1,2' or 'Planting 1,Planting 2'). "
            "If omitted, all available (uncensored) plantings are used."
        ),
    )
    p.add_argument(
        "--predict-plantings",
        default=None,
        help=(
            "Comma-separated planting numbers to predict flowering time on "
            "(e.g. '5,6' or 'Planting 5,Planting 6'). "
            "After fitting, uses the trained alpha/beta to predict DAP for these plantings "
            "and compares with observed values (if available)."
        ),
    )

    p.add_argument(
        "--exclude-genotypes",
        default=None,
        help="Comma-separated genotype IDs to exclude from fitting and prediction (e.g. 'FA306,FA125').",
    )

    # GBLUP DAP prediction
    p.add_argument(
        "--evd-path",
        default=None,
        help="Path to EVD.rda (eigendecomposition of G-matrix). Default: pipeline/2_gmatrix/{crop}/EVD.rda",
    )
    p.add_argument("--no-gblup-dap", action="store_true", help="Skip GBLUP DAP prediction")
    p.add_argument("--loo-variance", action="store_true", default=True,
                   help="Use leave-one-planting-out CV variance for per-genotype mechanistic variance "
                        "instead of global training MSE (default: enabled)")
    p.add_argument("--no-loo-variance", dest="loo_variance", action="store_false",
                   help="Disable LOO variance; use global training MSE for all genotypes (old behavior)")
    p.add_argument("--full-loo", action="store_true",
                   help="Enable Full LOO: re-fit (alpha, beta) per fold via dual annealing "
                        "for genuinely independent LOO variance estimates. "
                        "Replaces cheap LOO. ~5x slower than main fitting.")
    p.add_argument("--full-loo-maxiter", type=int, default=None,
                   help="Override --maxiter for Full LOO fold fits (default: uses --maxiter). "
                        "Set lower (e.g. 500) for faster Full LOO runs.")

    # Photoperiod
    p.add_argument("--photo-model", type=str, default=None,
                   choices=["sinclair", "logistic", "logistic3", "linear_plateau"],
                   help='Photoperiod response model: "sinclair" (hard cutoff, default), '
                        '"logistic" (smooth 2-param sigmoid), '
                        '"logistic3" (3-param logistic, Messina formulation), or '
                        '"linear_plateau" (Grimm 1993, night-length based).')
    p.add_argument("--photo-type", type=str, default="short_day",
                   choices=["short_day", "long_day"],
                   help="Photoperiod model type: short_day (development suppressed by long days) "
                        "or long_day (development suppressed by short days). Default: short_day.")
    # 3-param logistic (Messina) photoperiod — initial values and bounds
    p.add_argument("--photo-a", type=float, default=None,
                   help="Logistic3 photoperiod amplitude parameter a (e.g. 0.8). Floor = 1/(1+a).")
    p.add_argument("--photo-b", type=float, default=None,
                   help="Logistic3 photoperiod slope parameter b (negative=short-day, positive=long-day).")
    p.add_argument("--photo-d", type=float, default=None,
                   help="Logistic3 photoperiod inflection point d in hours (daylength at half-max).")
    p.add_argument("--photo-a-bounds", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Search bounds for logistic3 a parameter (e.g. 0.01 5.0)")
    p.add_argument("--photo-b-bounds", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Search bounds for logistic3 b parameter (e.g. -5.0 5.0)")
    p.add_argument("--photo-d-bounds", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Search bounds for logistic3 d parameter in hours (e.g. 10.0 18.0)")
    # Linear-plateau (Grimm 1993) photoperiod
    p.add_argument("--photo-n-min-bounds", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Linear-plateau N_min bounds in hours (night length)")
    p.add_argument("--photo-n-opt-bounds", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Linear-plateau N_opt bounds in hours (night length)")
    p.add_argument("--latitude-map", default=None,
                   help='JSON string mapping planting/location names to latitudes, e.g. \'{"CIT": 10.0, "ND": 46.9}\'')
    p.add_argument("--rn-photoperiod", action="store_true",
                   help="Extend the RN-GBLUP (Jarquín 2014) reaction norm model with a "
                        "photoperiod environmental kernel P_K and G⊙P interaction kernel: "
                        "y = µ + g + e + p + ge + gp + ε  (6 variance components instead of 4). "
                        "Requires --latitude-map (or DAYLhr in weather). "
                        "Default OFF preserves the original 4-VC model byte-identically.")

    p.add_argument("--mech-no-photo", action="store_true",
                   help="Skip the bolted-on photoperiod stage entirely. Mechanistic "
                        "predictions use cardinal_beta(T) only (no photo coupling). "
                        "Useful for temp-only diagnostics and the photo/temp/mech "
                        "comparison. Cannot be combined with --photo-only.")
    p.add_argument("--mech-dual-threshold", action="store_true",
                   help="Use dual-threshold mech+photo coupling instead of the "
                        "multiplicative `cardinal_beta(T) × F_photo(DL)` recompute. "
                        "Tracks thermal and photoperiod as TWO independent state "
                        "variables, each with its own per-genotype threshold "
                        "(H_thermal from Phase 1 fit, H_photo from fit_photo_only). "
                        "Predicts pred_dap = max(day_thermal_done, day_photo_done) "
                        "— flowering happens only when BOTH thresholds are met. "
                        "Avoids the variance-loss degeneracy of the multiplicative "
                        "model by construction (no cross-component substitution).")

    # Top-K dual annealing
    p.add_argument("--top-k", type=int, default=100,
                   help="Number of best points to keep from dual annealing search "
                        "for per-genotype distribution (default: 100)")
    p.add_argument("--run-joint", action="store_true",
                   help="Run Joint CGM-WGP (Messina/Technow EM+L-BFGS-B) as a comparison "
                        "method. Fits per-genotype (Theta, S, Pc) under a multivariate "
                        "normal genomic prior. Cardinal temps fixed from config.")
    # Joint CGM-WGP per-method overrides (otherwise inherits model.Tb/Topt/Tc from --Tb etc.)
    p.add_argument("--joint-photo-enabled", action="store_true",
                   help="Enable photoperiod g(P) in Joint CGM-WGP forward model.")
    p.add_argument("--joint-Tb", type=float, default=None,
                   help="Joint-specific Tb override (defaults to --Tb).")
    p.add_argument("--joint-Topt", type=float, default=None,
                   help="Joint-specific Topt override (defaults to --Topt).")
    p.add_argument("--joint-Tc", type=float, default=None,
                   help="Joint-specific Tc override (defaults to --Tc).")
    p.add_argument("--joint-a-fixed", type=float, default=1.0,
                   help="Fixed photoperiod amplitude a (default: 1.0).")
    p.add_argument("--joint-Theta-bounds", type=float, nargs=2, default=[5.0, 200.0],
                   metavar=("LO", "HI"), help="Theta (thermal threshold) bounds.")
    p.add_argument("--joint-S-bounds", type=float, nargs=2, default=[-5.0, 5.0],
                   metavar=("LO", "HI"),
                   help="S (photo sensitivity) bounds. For long-day crops use [-5, 0].")
    p.add_argument("--joint-Pc-bounds", type=float, nargs=2, default=[10.0, 18.0],
                   metavar=("LO", "HI"), help="Pc (critical photoperiod) bounds.")
    p.add_argument("--joint-max-em-iters", type=int, default=20,
                   help="Joint EM max iterations (default: 20).")
    p.add_argument("--joint-em-tol", type=float, default=1e-4,
                   help="Joint EM convergence tolerance (default: 1e-4).")
    p.add_argument("--joint-lbfgsb-maxiter", type=int, default=500,
                   help="Joint inner L-BFGS-B max iterations (default: 500).")
    p.add_argument("--joint-photo-direction", choices=["short_day", "long_day"],
                   default="short_day",
                   help="Photoperiod response direction for Joint additive model "
                        "(short_day: long days delay; long_day: short days delay).")
    return p.parse_args()


def parse_planting_names(raw: str) -> list:
    """Parse a comma-separated planting spec into planting names.

    Accepts bare numbers ("1", "2") which become "Planting 1", "Planting 2"
    (Broccoli convention), full "Planting X" names, or location-style names
    like "IN_loc1" which are passed through as-is (Soybean convention).
    """
    names = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower().startswith("planting"):
            names.append(tok)
        elif tok.isdigit():
            names.append(f"Planting {tok}")
        else:
            names.append(tok)
    return names




def _topk_one_genotype(task):
    """Worker function for parallel top-K dual annealing per genotype."""
    g = task["g"]
    T_dict_local = {g: task["T_dict_g"]}
    DAP_obs_local = {g: task["DAP_obs_g"]}
    planting_list = task["planting_list"]
    plantings_fn = lambda _gid, _pl=planting_list: _pl

    DL_dict_local = None
    if task.get("DL_dict_g") is not None:
        DL_dict_local = {g: task["DL_dict_g"]}

    try:
        posterior = topk_dual_annealing(
            g=g,
            temp_base=task["Tb"],
            temp_optimal=task["Topt"],
            temp_critical=task["Tc"],
            plantings_of_g_fn=plantings_fn,
            T_dict=T_dict_local,
            DAP_obs_dict=DAP_obs_local,
            alpha_bounds=task.get("alpha_bounds", (0.5, 8.0)),
            beta_bounds=task.get("beta_bounds", (0.5, 8.0)),
            top_k=task.get("top_k", 100),
            maxiter=task["maxiter"],
            local_max_iter=task["local_max_iter"],
            seed=task["seed"],
            n_restarts=task.get("n_restarts", 1),
            normalize_loss=task.get("normalize_loss", False),
            use_penalty=task.get("use_penalty", False),
            alpha_target=task.get("alpha_target", 0.0),
            beta_target=task.get("beta_target", 0.0),
            lambda_a=task.get("lambda_a", 0.0),
            lambda_b=task.get("lambda_b", 0.0),
            topt_bounds=task.get("topt_bounds"),
            topt_target=task.get("topt_target", 0.0),
            lambda_t=task.get("lambda_t", 0.0),
            DL_dict=DL_dict_local,
            photo_B=task.get("photo_B"),
            photo_Pc=task.get("photo_Pc"),
            photo_type=task.get("photo_type", "short_day"),
            photo_model=task.get("photo_model", "sinclair"),
            photo_B_bounds=task.get("photo_B_bounds"),
            photo_Pc_bounds=task.get("photo_Pc_bounds"),
            photo_B_target=task.get("photo_B_target", 0.0),
            photo_Pc_target=task.get("photo_Pc_target", 0.0),
            lambda_bphoto=task.get("lambda_bphoto", 0.0),
            lambda_pc=task.get("lambda_pc", 0.0),
            # 3-param logistic (Messina) photoperiod
            photo_a_bounds=task.get("photo_a_bounds"),
            photo_b_bounds=task.get("photo_b_bounds"),
            photo_d_bounds=task.get("photo_d_bounds"),
            photo_a_target=task.get("photo_a_target", 0.0),
            photo_b_target=task.get("photo_b_target", 0.0),
            photo_d_target=task.get("photo_d_target", 0.0),
            lambda_photo_a=task.get("lambda_photo_a", 0.0),
            lambda_photo_b=task.get("lambda_photo_b", 0.0),
            lambda_photo_d=task.get("lambda_photo_d", 0.0),
            # Linear-plateau (Grimm 1993) photoperiod — fixed pop-level params
            photo_n_min=task.get("photo_n_min"),
            photo_n_opt=task.get("photo_n_opt"),
            obs_perturbation_sd=task.get("obs_perturbation_sd", 0.0),
            perturb_maxiter=task.get("perturb_maxiter", 0),
        )

        # Posterior predictive DAP (reuses posterior_predictive_dap)
        pp_result = {}
        predict_plantings = task.get("predict_plantings")
        if predict_plantings:
            pp_result = posterior_predictive_dap(
                g=g,
                posterior=posterior,
                predict_plantings=predict_plantings,
                T_dict=T_dict_local,
                DAP_obs_dict=DAP_obs_local,
                plantings_of_g_fn=plantings_fn,
                temp_base=task["Tb"],
                temp_optimal=task["Topt"],
                temp_critical=task["Tc"],
                DL_dict=DL_dict_local,
                photo_B=task.get("photo_B"),
                photo_Pc=task.get("photo_Pc"),
                photo_type=task.get("photo_type", "short_day"),
                photo_model=task.get("photo_model", "sinclair"),
                max_samples=task.get("max_posterior_samples", 200),
                photo_n_min=task.get("photo_n_min"),
                photo_n_opt=task.get("photo_n_opt"),
            )

        result = {
            "id": g,
            "alpha": posterior["alpha_map"],
            "beta": posterior["beta_map"],
            "loss": posterior["map_loss"],
            "alpha_mean": posterior["alpha_mean"],
            "beta_mean": posterior["beta_mean"],
            "alpha_std": posterior["alpha_std"],
            "beta_std": posterior["beta_std"],
            "n_accepted": posterior["n_accepted"],
            "accept_threshold": posterior["accept_threshold"],
            "n_training_plantings": posterior["n_training_plantings"],
            "posterior_predictive": pp_result,
            "error": None,
        }
        if "topt_map" in posterior:
            result["topt"] = posterior["topt_map"]
            result["topt_mean"] = posterior["topt_mean"]
            result["topt_std"] = posterior["topt_std"]
        if "photo_B_map" in posterior:
            result["photo_B"] = posterior["photo_B_map"]
            result["photo_Pc"] = posterior["photo_Pc_map"]
            result["photo_B_mean"] = posterior["photo_B_mean"]
            result["photo_Pc_mean"] = posterior["photo_Pc_mean"]
            result["photo_B_std"] = posterior["photo_B_std"]
            result["photo_Pc_std"] = posterior["photo_Pc_std"]
        if "photo_a_map" in posterior:
            result["photo_a"] = posterior["photo_a_map"]
            result["photo_b"] = posterior["photo_b_map"]
            result["photo_d"] = posterior["photo_d_map"]
            result["photo_a_mean"] = posterior["photo_a_mean"]
            result["photo_b_mean"] = posterior["photo_b_mean"]
            result["photo_d_mean"] = posterior["photo_d_mean"]
            result["photo_a_std"] = posterior["photo_a_std"]
            result["photo_b_std"] = posterior["photo_b_std"]
            result["photo_d_std"] = posterior["photo_d_std"]
        return result
    except Exception as exc:
        return {
            "id": g,
            "alpha": float("nan"), "beta": float("nan"),
            "loss": float("nan"),
            "alpha_mean": float("nan"), "beta_mean": float("nan"),
            "alpha_std": float("nan"), "beta_std": float("nan"),
            "n_accepted": 0, "accept_threshold": float("nan"),
            "n_training_plantings": 0,
            "posterior_predictive": {},
            "error": str(exc),
        }




def _build_topk_tasks(
    genotypes, T_dict, DAP_obs_dict, DL_dict, plantings_of_g_fn,
    args_Tb, args_Topt, args_Tc,
    alpha_bounds, beta_bounds, maxiter, local_max_iter, seed, n_restarts,
    normalize_loss, top_k, max_posterior_samples, predict_plantings,
    photo_B, photo_Pc, photo_type="short_day", photo_model="sinclair",
    gblup_targets=None, lambda_a=0.0, lambda_b=0.0,
    lambda_scale=None,
    topt_bounds=None, lambda_t=0.0,
    obs_perturbation_sd=0.0,
    perturb_maxiter=0,
    photo_B_bounds=None, photo_Pc_bounds=None,
    gblup_photo_targets=None,
    lambda_bphoto=0.0, lambda_pc=0.0,
    # 3-param logistic (Messina) photoperiod
    photo_a_bounds=None, photo_b_bounds=None, photo_d_bounds=None,
    gblup_photo3_targets=None,
    lambda_photo_a=0.0, lambda_photo_b=0.0, lambda_photo_d=0.0,
    # Linear-plateau (Grimm 1993) photoperiod — population-level fixed params
    photo_n_min=None, photo_n_opt=None,
):
    """Build task dicts for parallel top-K dual annealing.

    gblup_targets: dict {genotype_id: (alpha_pred, beta_pred[, topt_pred])} or None.
    When provided, sets use_penalty=True for genotypes with predictions.
    lambda_scale: optional dict {genotype_id: (w_alpha, w_beta)} for
    per-genotype lambda scaling (from G-matrix neighbor weights).
    topt_bounds: optional (lo, hi) tuple. When provided, Topt is fitted as 3rd parameter.
    lambda_t: penalty strength for Topt toward G-matrix neighbor target.
    obs_perturbation_sd: observation noise std for perturbed restarts.
    perturb_maxiter: DA iterations for perturbed restarts (0 = auto).
    photo_B_bounds: optional (lo, hi) tuple. When provided, B is fitted per genotype.
    photo_Pc_bounds: optional (lo, hi) tuple. When provided, Pc is fitted per genotype.
    gblup_photo_targets: dict {genotype_id: (B_pred, Pc_pred)} or None.
    lambda_bphoto: penalty strength for B toward GBLUP target.
    lambda_pc: penalty strength for Pc toward GBLUP target.
    photo_a_bounds: optional (lo, hi) tuple for logistic3 a parameter.
    photo_b_bounds: optional (lo, hi) tuple for logistic3 b parameter.
    photo_d_bounds: optional (lo, hi) tuple for logistic3 d parameter.
    gblup_photo3_targets: dict {genotype_id: (a_pred, b_pred, d_pred)} or None.
    lambda_photo_a/b/d: penalty strength for logistic3 params toward GBLUP targets.
    """
    tasks = []
    fit_photo = photo_B_bounds is not None and photo_Pc_bounds is not None
    fit_photo3 = (photo_a_bounds is not None and photo_b_bounds is not None
                  and photo_d_bounds is not None)
    for g in genotypes:
        plantings = list(plantings_of_g_fn(g))
        use_penalty = False
        alpha_target = 0.0
        beta_target = 0.0
        topt_target_val = 0.0
        photo_B_target_val = 0.0
        photo_Pc_target_val = 0.0
        photo_a_target_val = 0.0
        photo_b_target_val = 0.0
        photo_d_target_val = 0.0
        if gblup_targets and g in gblup_targets:
            use_penalty = True
            tgt = gblup_targets[g]
            alpha_target, beta_target = tgt[0], tgt[1]
            if len(tgt) >= 3:
                topt_target_val = tgt[2]
        if gblup_photo_targets and g in gblup_photo_targets:
            use_penalty = True
            ptgt = gblup_photo_targets[g]
            photo_B_target_val, photo_Pc_target_val = ptgt[0], ptgt[1]
        if gblup_photo3_targets and g in gblup_photo3_targets:
            use_penalty = True
            p3tgt = gblup_photo3_targets[g]
            photo_a_target_val, photo_b_target_val, photo_d_target_val = p3tgt[0], p3tgt[1], p3tgt[2]

        # Per-genotype lambda scaling (from G-matrix kinship weights)
        eff_lambda_a = lambda_a
        eff_lambda_b = lambda_b
        eff_lambda_t = lambda_t
        eff_lambda_bphoto = lambda_bphoto
        eff_lambda_pc = lambda_pc
        eff_lambda_photo_a = lambda_photo_a
        eff_lambda_photo_b = lambda_photo_b
        eff_lambda_photo_d = lambda_photo_d
        if lambda_scale and g in lambda_scale:
            wa, wb = lambda_scale[g]
            eff_lambda_a = lambda_a * wa
            eff_lambda_b = lambda_b * wb
            eff_lambda_t = lambda_t * wa  # use same kinship weight for topt
            eff_lambda_bphoto = lambda_bphoto * wa
            eff_lambda_pc = lambda_pc * wa
            eff_lambda_photo_a = lambda_photo_a * wa
            eff_lambda_photo_b = lambda_photo_b * wa
            eff_lambda_photo_d = lambda_photo_d * wa

        tasks.append({
            "g": g,
            "T_dict_g": T_dict[g],
            "DAP_obs_g": DAP_obs_dict[g],
            "planting_list": plantings,
            "Tb": args_Tb, "Topt": args_Topt, "Tc": args_Tc,
            "maxiter": maxiter, "local_max_iter": local_max_iter,
            "seed": seed,
            "alpha_bounds": tuple(alpha_bounds),
            "beta_bounds": tuple(beta_bounds),
            "n_restarts": n_restarts,
            "normalize_loss": normalize_loss,
            "top_k": top_k,
            "max_posterior_samples": max_posterior_samples,
            "predict_plantings": predict_plantings,
            "DL_dict_g": DL_dict.get(g) if DL_dict else None,
            "photo_B": photo_B, "photo_Pc": photo_Pc,
            "photo_type": photo_type, "photo_model": photo_model,
            "use_penalty": use_penalty,
            "alpha_target": alpha_target,
            "beta_target": beta_target,
            "lambda_a": eff_lambda_a,
            "lambda_b": eff_lambda_b,
            "topt_bounds": tuple(topt_bounds) if topt_bounds is not None else None,
            "topt_target": topt_target_val,
            "lambda_t": eff_lambda_t,
            "photo_B_bounds": tuple(photo_B_bounds) if photo_B_bounds is not None else None,
            "photo_Pc_bounds": tuple(photo_Pc_bounds) if photo_Pc_bounds is not None else None,
            "photo_B_target": photo_B_target_val,
            "photo_Pc_target": photo_Pc_target_val,
            "lambda_bphoto": eff_lambda_bphoto,
            "lambda_pc": eff_lambda_pc,
            # 3-param logistic (Messina) photoperiod
            "photo_a_bounds": tuple(photo_a_bounds) if photo_a_bounds is not None else None,
            "photo_b_bounds": tuple(photo_b_bounds) if photo_b_bounds is not None else None,
            "photo_d_bounds": tuple(photo_d_bounds) if photo_d_bounds is not None else None,
            "photo_a_target": photo_a_target_val,
            "photo_b_target": photo_b_target_val,
            "photo_d_target": photo_d_target_val,
            "lambda_photo_a": eff_lambda_photo_a,
            "lambda_photo_b": eff_lambda_photo_b,
            "lambda_photo_d": eff_lambda_photo_d,
            # Linear-plateau (Grimm 1993) photoperiod — fixed pop-level
            "photo_n_min": photo_n_min,
            "photo_n_opt": photo_n_opt,
            "obs_perturbation_sd": obs_perturbation_sd,
            "perturb_maxiter": perturb_maxiter,
        })
    return tasks


def _fullloo_one_fold(task):
    """Worker for one (genotype, fold) pair of Full LOO variance.

    Fits (alpha, beta) on K-1 plantings via DA, computes H* from the K-1
    plantings, and predicts fractional DAP for the held-out planting.
    """
    g = task["g"]
    held_out = task["held_out_planting"]
    train_subset = task["planting_list"]  # K-1 plantings

    T_dict_local = {g: task["T_dict_g"]}
    DAP_obs_local = {g: task["DAP_obs_g"]}
    plantings_fn = lambda _gid, _pl=train_subset: _pl

    DL_dict_local = None
    if task.get("DL_dict_g") is not None:
        DL_dict_local = {g: task["DL_dict_g"]}

    try:
        posterior = topk_dual_annealing(
            g=g,
            temp_base=task["Tb"],
            temp_optimal=task["Topt"],
            temp_critical=task["Tc"],
            plantings_of_g_fn=plantings_fn,
            T_dict=T_dict_local,
            DAP_obs_dict=DAP_obs_local,
            alpha_bounds=task.get("alpha_bounds", (0.5, 8.0)),
            beta_bounds=task.get("beta_bounds", (0.5, 8.0)),
            top_k=task.get("top_k", 100),
            maxiter=task["maxiter"],
            local_max_iter=task.get("local_max_iter", 150),
            seed=task.get("seed"),
            n_restarts=task.get("n_restarts", 1),
            normalize_loss=task.get("normalize_loss", False),
            DL_dict=DL_dict_local,
            photo_B=task.get("photo_B"),
            photo_Pc=task.get("photo_Pc"),
            photo_type=task.get("photo_type", "short_day"),
            photo_model=task.get("photo_model", "sinclair"),
            photo_n_min=task.get("photo_n_min"),
            photo_n_opt=task.get("photo_n_opt"),
        )

        alpha_hat = posterior["alpha_map"]
        beta_hat = posterior["beta_map"]

        # Compute H* from K-1 training plantings using the fold's fitted params
        H_vals = []
        for e in train_subset:
            if e not in DAP_obs_local[g]:
                continue
            dl_e = None
            if DL_dict_local and g in DL_dict_local and e in DL_dict_local[g]:
                dl_e = DL_dict_local[g][e]
            rates = daily_development_rate(
                T_dict_local[g][e], alpha=alpha_hat, beta=beta_hat,
                temp_base=task["Tb"], temp_optimal=task["Topt"],
                temp_critical=task["Tc"],
                daylengths=dl_e, photo_B=task.get("photo_B"),
                photo_Pc=task.get("photo_Pc"),
                photo_type=task.get("photo_type", "short_day"),
                photo_model=task.get("photo_model", "sinclair"),
                photo_n_min=task.get("photo_n_min"),
                photo_n_opt=task.get("photo_n_opt"),
            )
            dap = int(DAP_obs_local[g][e])
            dap_cut = max(0, min(dap, len(rates)))
            H_vals.append(float(np.sum(rates[:dap_cut])))

        if not H_vals:
            return {"g": g, "fold": held_out, "pred_dap": None,
                    "obs_dap": None, "error": "no H_vals from K-1 subset"}
        target_H = float(np.mean(H_vals))
        if target_H <= 0:
            return {"g": g, "fold": held_out, "pred_dap": None,
                    "obs_dap": None, "error": "target_H <= 0"}

        # Predict DAP for held-out planting
        dl_ho = None
        if DL_dict_local and g in DL_dict_local and held_out in DL_dict_local[g]:
            dl_ho = DL_dict_local[g][held_out]
        pred_dap = predict_flowering_dap(
            temps=T_dict_local[g][held_out],
            target_H=target_H,
            alpha=alpha_hat, beta=beta_hat,
            temp_base=task["Tb"], temp_optimal=task["Topt"],
            temp_critical=task["Tc"],
            daylengths=dl_ho, photo_B=task.get("photo_B"),
            photo_Pc=task.get("photo_Pc"),
            photo_type=task.get("photo_type", "short_day"),
            photo_model=task.get("photo_model", "sinclair"),
            fractional=True,
            photo_n_min=task.get("photo_n_min"),
            photo_n_opt=task.get("photo_n_opt"),
        )

        obs_dap = float(DAP_obs_local[g][held_out])
        return {"g": g, "fold": held_out, "pred_dap": float(pred_dap),
                "obs_dap": obs_dap, "error": None}
    except Exception as exc:
        return {"g": g, "fold": held_out, "pred_dap": None,
                "obs_dap": None, "error": str(exc)}


def _build_fullloo_tasks(
    genotypes, T_dict, DAP_obs_dict, DL_dict, plantings_of_g_fn,
    args_Tb, args_Topt, args_Tc,
    alpha_bounds, beta_bounds, maxiter, local_max_iter, seed, n_restarts,
    normalize_loss, top_k,
    photo_B, photo_Pc, photo_type="short_day", photo_model="sinclair",
    min_folds=3,
):
    """Build (genotype, fold) task dicts for Full LOO variance."""
    tasks = []
    for g in genotypes:
        plantings = list(plantings_of_g_fn(g))
        if len(plantings) < min_folds:
            continue
        for fold_idx, held_out in enumerate(plantings):
            train_subset = [p for p in plantings if p != held_out]
            tasks.append({
                "g": g,
                "held_out_planting": held_out,
                "planting_list": train_subset,
                "T_dict_g": T_dict[g],
                "DAP_obs_g": DAP_obs_dict[g],
                "DL_dict_g": DL_dict.get(g) if DL_dict else None,
                "Tb": args_Tb, "Topt": args_Topt, "Tc": args_Tc,
                "maxiter": maxiter, "local_max_iter": local_max_iter,
                "seed": (seed or 42) + fold_idx,
                "alpha_bounds": tuple(alpha_bounds),
                "beta_bounds": tuple(beta_bounds),
                "n_restarts": n_restarts,
                "normalize_loss": normalize_loss,
                "top_k": top_k,
                "photo_B": photo_B, "photo_Pc": photo_Pc,
                "photo_type": photo_type, "photo_model": photo_model,
            })
    return tasks


def _compute_loo_vars(
    fit_results, args,
    T_dict, DAP_obs_dict, DL_dict, plantings_of_g_fn,
    train_planting_names,
    photo_B, photo_Pc, photo_type, photo_model,
    workers,
):
    """Compute LOO variance per genotype (cheap or Full LOO depending on args).

    Returns dict {genotype_id: loo_var_float}.
    """
    from collections import defaultdict

    loo_vars = {}
    use_loo = args.loo_variance or getattr(args, "full_loo", False)
    if not use_loo:
        return loo_vars

    if getattr(args, "full_loo", False):
        # Full LOO: re-fit (alpha, beta) per fold via parallel DA
        loo_maxiter = getattr(args, "full_loo_maxiter", None) or args.maxiter
        successful_gids = [fr["id"] for fr in fit_results if fr["error"] is None]
        tasks = _build_fullloo_tasks(
            genotypes=successful_gids,
            T_dict=T_dict, DAP_obs_dict=DAP_obs_dict, DL_dict=DL_dict,
            plantings_of_g_fn=plantings_of_g_fn,
            args_Tb=args.Tb, args_Topt=args.Topt, args_Tc=args.Tc,
            alpha_bounds=args.alpha_bounds, beta_bounds=args.beta_bounds,
            maxiter=loo_maxiter,
            local_max_iter=args.local_max_iter,
            seed=args.seed, n_restarts=args.n_restarts,
            normalize_loss=args.normalize_loss,
            top_k=args.top_k,
            photo_B=photo_B, photo_Pc=photo_Pc,
            photo_type=photo_type, photo_model=photo_model,
        )
        print(f"  Full LOO: {len(tasks)} (genotype, fold) tasks, maxiter={loo_maxiter}")
        fold_results = _run_fitting(tasks, workers, worker_fn=_fullloo_one_fold, verbose=False)

        # Group by genotype, compute MSE of residuals
        n_ok = sum(1 for fr in fold_results if fr["error"] is None)
        n_fail = len(fold_results) - n_ok
        print(f"  Full LOO fitting: {n_ok} succeeded, {n_fail} failed")
        residuals_by_g = defaultdict(list)
        for fr in fold_results:
            if fr["error"] is None and fr["pred_dap"] is not None and fr["pred_dap"] > 0:
                residuals_by_g[fr["g"]].append(fr["pred_dap"] - fr["obs_dap"])
        for gid, resids in residuals_by_g.items():
            if len(resids) >= 3:
                arr = np.array(resids)
                loo_vars[gid] = float(np.mean(arr ** 2))
    else:
        # Cheap LOO: same fitted (alpha, beta), only recompute H* per fold
        from cgm_wgp.mech_fit import loo_planting_variance
        for fr in fit_results:
            if fr["error"] is not None:
                continue
            gid = fr["id"]
            loo_v = loo_planting_variance(
                g=gid, alpha=fr["alpha"], beta=fr["beta"],
                temp_base=args.Tb, temp_optimal=args.Topt, temp_critical=args.Tc,
                T_dict=T_dict, DAP_obs_dict=DAP_obs_dict,
                training_plantings=train_planting_names,
                DL_dict=DL_dict, photo_B=photo_B, photo_Pc=photo_Pc,
                photo_type=photo_type,
            )
            if loo_v is not None:
                loo_vars[gid] = loo_v

    if loo_vars:
        vals = list(loo_vars.values())
        n_fallback = sum(1 for fr in fit_results if fr["error"] is None) - len(loo_vars)
        label = "Full LOO" if getattr(args, "full_loo", False) else "LOO"
        print(f"  {label} variance: min={min(vals):.1f}, median={np.median(vals):.1f}, "
              f"max={max(vals):.1f}, n={len(vals)}, fallback_to_global={n_fallback}")

    return loo_vars


def _run_fitting(tasks, workers, worker_fn=_topk_one_genotype, verbose=True):
    """Run parallel or sequential genotype fitting."""
    if workers == 1:
        fit_results = []
        for i, task in enumerate(tasks, start=1):
            g = task["g"]
            if verbose:
                print(f"[{i}/{len(tasks)}] Genotype: {g}")
            result = worker_fn(task)
            if verbose:
                if result["error"]:
                    print(f"  ! Fit failed for genotype {g}: {result['error']}", file=sys.stderr)
                else:
                    extra = ""
                    if "alpha_std" in result and not math.isnan(result.get("alpha_std", float("nan"))):
                        extra = f" | posterior: n={result['n_accepted']}, alpha_std={result['alpha_std']:.4f}, beta_std={result['beta_std']:.4f}"
                    print(f"  -> alpha={result['alpha']:.6f} beta={result['beta']:.6f} loss={result['loss']:.6e}{extra}")
            fit_results.append(result)
    else:
        n_workers = workers if workers > 0 else max(1, multiprocessing.cpu_count() - 2)
        if verbose:
            print(f"Fitting {len(tasks)} genotypes across {n_workers} workers...")
        with multiprocessing.Pool(n_workers) as pool:
            fit_results = pool.map(worker_fn, tasks)
        if verbose:
            n_ok = sum(1 for r in fit_results if r["error"] is None)
            n_fail = len(fit_results) - n_ok
            print(f"Completed: {n_ok} succeeded, {n_fail} failed")
    return fit_results


def _compute_training_rmse(
    fit_results, T_dict, DAP_obs_dict, DL_dict, plantings_of_g_fn,
    photo_B, photo_Pc, args_Tb, args_Topt, args_Tc, photo_type="short_day",
    photo_model="sinclair",
):
    """Compute training RMSE: predict DAP on training plantings using mean H*."""
    all_errors = []
    for result in fit_results:
        if result["error"] is not None:
            continue
        gid = result["id"]
        a_hat = result["alpha"]
        b_hat = result["beta"]
        if not (np.isfinite(a_hat) and np.isfinite(b_hat)):
            continue

        train_pl = list(plantings_of_g_fn(gid))
        if not train_pl:
            continue

        # Compute H* = mean cumulative progress across training plantings
        H_vals = []
        for e in train_pl:
            if gid not in T_dict or e not in T_dict[gid]:
                continue
            temps_e = np.asarray(T_dict[gid][e], dtype=float)
            dap_e = int(DAP_obs_dict[gid][e])
            dl_e = None
            if DL_dict and gid in DL_dict and e in DL_dict[gid]:
                dl_e = DL_dict[gid][e]
            rates = daily_development_rate(
                temps_e, alpha=a_hat, beta=b_hat,
                temp_base=args_Tb, temp_optimal=args_Topt, temp_critical=args_Tc,
                daylengths=dl_e, photo_B=photo_B, photo_Pc=photo_Pc,
                photo_type=photo_type, photo_model=photo_model,
            )
            dap_cut = max(0, min(dap_e, len(rates)))
            H_vals.append(float(np.sum(rates[:dap_cut])))

        if not H_vals:
            continue
        target_H = float(np.mean(H_vals))
        if target_H <= 0:
            continue

        # Predict back on training plantings
        for e in train_pl:
            if gid not in T_dict or e not in T_dict[gid]:
                continue
            temps_e = np.asarray(T_dict[gid][e], dtype=float)
            dap_obs = int(DAP_obs_dict[gid][e])
            dl_e = None
            if DL_dict and gid in DL_dict and e in DL_dict[gid]:
                dl_e = DL_dict[gid][e]
            predicted_dap = predict_flowering_dap(
                temps=temps_e, target_H=target_H,
                alpha=a_hat, beta=b_hat,
                temp_base=args_Tb, temp_optimal=args_Topt, temp_critical=args_Tc,
                daylengths=dl_e, photo_B=photo_B, photo_Pc=photo_Pc,
                photo_type=photo_type, photo_model=photo_model,
            )
            if predicted_dap > 0:
                all_errors.append(predicted_dap - dap_obs)

    if all_errors:
        return float(np.sqrt(np.mean(np.square(all_errors))))
    return 1e9


def _compute_gblup_loo_mech_mse(
    fit_results, evd_path, T_dict, DAP_obs_dict, DL_dict, plantings_of_g_fn,
    photo_B, photo_Pc, args_Tb, args_Topt, args_Tc,
    photo_type="short_day", photo_model="sinclair",
    photo_results=None,
):
    """Compute honest out-of-sample mech MSE and bias via leave-one-genotype-out GBLUP.

    For each training genotype, predict its (alpha, beta) from the other N-1
    training genotypes using GBLUP, then run the mech simulation with those
    predicted params and collect residuals against observed DAP. This matches
    the actual error regime of unseen genotypes (for `new_varieties` CV), and
    is much more honest than in-sample `training_mse`.

    Returns dict with {"mse", "bias", "n"} or None if GBLUP-LOO can't run.
    """
    from cgm_wgp.gblup import gblup_predict_params
    from pathlib import Path as _P

    if not _P(evd_path).exists():
        return None  # caller falls back to training_mse

    # Build fitted_params dict from current fit_results
    fitted_params = {}
    for r in fit_results:
        if r.get("error") is not None:
            continue
        a, b = r.get("alpha"), r.get("beta")
        loss = r.get("loss", 1e9)
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if not np.isfinite(loss) or loss >= 1e8:
            continue  # sentinel fits
        fitted_params[r["id"]] = (a, b)

    if len(fitted_params) < 3:
        return None

    # LOO-GBLUP: each fitted genotype predicted from the other N-1
    try:
        param_preds = gblup_predict_params(evd_path, fitted_params, loo=True)
    except Exception:
        return None

    # For each genotype, simulate mech with LOO-predicted (alpha, beta)
    # against its training plantings and compute residuals.
    all_errors = []
    for gid, orig in fitted_params.items():
        pred = param_preds.get(gid)
        if pred is None:
            continue
        loo_a, loo_b = float(pred[0]), float(pred[1])
        if not (np.isfinite(loo_a) and np.isfinite(loo_b)):
            continue
        if gid not in T_dict:
            continue

        # Per-genotype photo params (if fitted per-gid)
        _pr = photo_results.get(gid) if photo_results else None
        _pa = _pr.get("photo_a") if _pr else None
        _pb = _pr.get("photo_b") if _pr else None
        _pd = _pr.get("photo_d") if _pr else None
        _pnmin = _pr.get("photo_n_min") if _pr else None
        _pnopt = _pr.get("photo_n_opt") if _pr else None

        train_pl = list(plantings_of_g_fn(gid))
        # Compute H* for this genotype from OBSERVED data using LOO-predicted
        # (alpha, beta). This mirrors how we'd compute H* for an unseen gid.
        H_vals = []
        for e in train_pl:
            if e not in T_dict[gid]:
                continue
            temps_e = np.asarray(T_dict[gid][e], dtype=float)
            dap_e = int(DAP_obs_dict[gid][e])
            dl_e = None
            if DL_dict and gid in DL_dict and e in DL_dict[gid]:
                dl_e = DL_dict[gid][e]
            rates = daily_development_rate(
                temps_e, alpha=loo_a, beta=loo_b,
                temp_base=args_Tb, temp_optimal=args_Topt, temp_critical=args_Tc,
                daylengths=dl_e, photo_B=photo_B, photo_Pc=photo_Pc,
                photo_type=photo_type, photo_model=photo_model,
                photo_a=_pa, photo_b=_pb, photo_d=_pd,
                photo_n_min=_pnmin, photo_n_opt=_pnopt,
            )
            dap_cut = max(0, min(dap_e, len(rates)))
            H_vals.append(float(np.sum(rates[:dap_cut])))

        if not H_vals:
            continue
        target_H = float(np.mean(H_vals))
        if target_H <= 0:
            continue

        # Predict back on training plantings using LOO-predicted params
        for e in train_pl:
            if e not in T_dict[gid]:
                continue
            temps_e = np.asarray(T_dict[gid][e], dtype=float)
            dap_obs = int(DAP_obs_dict[gid][e])
            dl_e = None
            if DL_dict and gid in DL_dict and e in DL_dict[gid]:
                dl_e = DL_dict[gid][e]
            predicted_dap = predict_flowering_dap(
                temps=temps_e, target_H=target_H,
                alpha=loo_a, beta=loo_b,
                temp_base=args_Tb, temp_optimal=args_Topt, temp_critical=args_Tc,
                daylengths=dl_e, photo_B=photo_B, photo_Pc=photo_Pc,
                photo_type=photo_type, photo_model=photo_model,
                extrapolate=True,
                photo_a=_pa, photo_b=_pb, photo_d=_pd,
                photo_n_min=_pnmin, photo_n_opt=_pnopt,
            )
            if predicted_dap > 0:
                all_errors.append(float(predicted_dap - dap_obs))

    if not all_errors:
        return None

    mse = float(np.mean(np.square(all_errors)))
    bias = float(np.mean(all_errors))
    return {"mse": mse, "bias": bias, "n": len(all_errors)}


def _compute_env_holdout_mech_mse(
    fit_results, T_dict, DAP_obs_dict, DL_dict, plantings_of_g_fn,
    train_planting_names, photo_B, photo_Pc,
    args_Tb, args_Topt, args_Tc,
    photo_type="short_day", photo_model="sinclair",
    photo_results=None,
):
    """Compute honest env-extrapolation mech MSE via leave-one-env-out.

    For each training env `e`, hold it out and re-derive target_H per genotype
    from the OTHER training envs (using the fitted (α, β) as-is). Then predict
    DAP on the held-out env and collect residuals. Aggregates across all
    (genotype, held_out_env) pairs.

    This captures the "extrapolate to a new env" error regime that in-sample
    training_mse and genotype-LOO both underestimate. Used as a floor for
    mech discrepancy_var when the predict env differs from training envs.

    Degenerate cases:
      - Single training env: cannot hold out. Returns None.
      - Genotype with <2 plantings: cannot form a holdout leave-one-out. Skipped.

    Returns dict with {"mse", "bias", "n"} or None if the calc can't run.

    FUTURE: enhance with weather-distance inflation. Currently this is an
    in-sample env-LOO estimate; if the actual predict env is weather-OOD
    (e.g., tropical predict from temperate training), the true error will
    exceed env_loo_mech_mse. A proper fix would compute Mahalanobis distance
    in (Tmean, DL_mean, DL_range) space between predict env and training
    centroid, and inflate variance proportionally. See plan
    wild-sniffing-turing.md for design notes.
    """
    if train_planting_names is None or len(train_planting_names) < 2:
        return None

    all_errors = []
    for row in fit_results:
        if row.get("error") is not None:
            continue
        gid = row["id"]
        a_hat = row.get("alpha")
        b_hat = row.get("beta")
        if not (isinstance(a_hat, (int, float)) and np.isfinite(a_hat)
                and isinstance(b_hat, (int, float)) and np.isfinite(b_hat)):
            continue

        # Per-genotype photo params (if fitted per-gid)
        _pr = photo_results.get(gid) if photo_results else None
        _pa = _pr.get("photo_a") if _pr else None
        _pb = _pr.get("photo_b") if _pr else None
        _pd = _pr.get("photo_d") if _pr else None
        _pnmin = _pr.get("photo_n_min") if _pr else None
        _pnopt = _pr.get("photo_n_opt") if _pr else None

        if gid not in T_dict:
            continue
        train_pl = list(plantings_of_g_fn(gid))
        if len(train_pl) < 2:
            continue  # can't leave-one-out with fewer than 2 envs

        # Precompute cum progress at obs_dap per env using fitted (α, β)
        H_by_env = {}
        for e in train_pl:
            if e not in T_dict[gid] or e not in DAP_obs_dict.get(gid, {}):
                continue
            temps_e = np.asarray(T_dict[gid][e], dtype=float)
            dap_e = int(DAP_obs_dict[gid][e])
            dl_e = None
            if DL_dict and gid in DL_dict and e in DL_dict[gid]:
                dl_e = DL_dict[gid][e]
            rates = daily_development_rate(
                temps_e, alpha=a_hat, beta=b_hat,
                temp_base=args_Tb, temp_optimal=args_Topt, temp_critical=args_Tc,
                daylengths=dl_e, photo_B=photo_B, photo_Pc=photo_Pc,
                photo_type=photo_type, photo_model=photo_model,
                photo_a=_pa, photo_b=_pb, photo_d=_pd,
                photo_n_min=_pnmin, photo_n_opt=_pnopt,
            )
            dap_cut = max(0, min(dap_e, len(rates)))
            H_by_env[e] = float(np.sum(rates[:dap_cut]))

        if len(H_by_env) < 2:
            continue

        # For each env, hold it out, compute target_H from remaining, predict
        env_list = list(H_by_env.keys())
        for held_out in env_list:
            other = [e for e in env_list if e != held_out]
            target_H = float(np.mean([H_by_env[e] for e in other]))
            if target_H <= 0:
                continue
            temps_ho = np.asarray(T_dict[gid][held_out], dtype=float)
            dap_obs = int(DAP_obs_dict[gid][held_out])
            dl_ho = None
            if DL_dict and gid in DL_dict and held_out in DL_dict[gid]:
                dl_ho = DL_dict[gid][held_out]
            pred_dap = predict_flowering_dap(
                temps=temps_ho, target_H=target_H,
                alpha=a_hat, beta=b_hat,
                temp_base=args_Tb, temp_optimal=args_Topt, temp_critical=args_Tc,
                daylengths=dl_ho, photo_B=photo_B, photo_Pc=photo_Pc,
                photo_type=photo_type, photo_model=photo_model,
                extrapolate=True,
                photo_a=_pa, photo_b=_pb, photo_d=_pd,
                photo_n_min=_pnmin, photo_n_opt=_pnopt,
            )
            if pred_dap > 0:
                all_errors.append(float(pred_dap - dap_obs))

    if not all_errors:
        return None

    mse = float(np.mean(np.square(all_errors)))
    bias = float(np.mean(all_errors))
    return {"mse": mse, "bias": bias, "n": len(all_errors)}


def _derive_rn_gblup_alpha_beta_targets(
    *,
    genotypes,
    rn_dap_by_cell,
    T_dict,
    DL_dict,
    plantings_of_g_fn,
    args,
    photo_kwargs=None,
    workers: int = 1,
    verbose: bool = False,
):
    """Derive per-genotype (α, β) targets from RN-GBLUP DAP predictions.

    For each genotype, build a synthetic `DAP_obs_dict` where observed DAPs
    come from RN-GBLUP predictions at training environments. Then fit
    (α, β) using the existing Phase 1 mech fitter via `_build_topk_tasks`
    with `top_k=1, n_restarts=1` and no penalty. The resulting (α, β) per
    genotype is what Phase 1 mech would produce if it were fitting to
    RN-GBLUP's G×E-aware predictions instead of raw observations.

    These (α_rn, β_rn) are used as `gblup_targets` in the Phase 1 penalty
    machinery — replacing the current main-effects-only GBLUP smoothing
    with a G×E-aware signal derived from RN-GBLUP.

    Skips:
      - Genotypes with < 2 RN-GBLUP envs (need variance across envs for the
        Phase 1 variance loss to have signal).
      - Genotypes whose fit failed (error is not None) or returned non-finite
        (α, β).

    Returns dict {gid: (alpha_hat, beta_hat)} or {} on total failure.
    """
    if not rn_dap_by_cell:
        return {}

    # Build synthetic DAP_obs_dict for training envs only, from RN-GBLUP
    synthetic_DAP_obs = {}
    for g in genotypes:
        g_upper = str(g).strip().upper()
        train_envs = list(plantings_of_g_fn(g))
        obs_for_g = {}
        for e in train_envs:
            key = (g_upper, str(e).strip())
            if key in rn_dap_by_cell:
                val = rn_dap_by_cell[key]
                if val is not None and np.isfinite(val) and val > 0:
                    obs_for_g[e] = int(round(val))
        if len(obs_for_g) >= 2:
            synthetic_DAP_obs[g] = obs_for_g

    if not synthetic_DAP_obs:
        return {}

    # Build tasks with no penalty, no perturbation, top_k=1, n_restarts=1.
    # This is the cheapest possible Phase 1 fit — just find the MAP.
    target_genotypes = list(synthetic_DAP_obs.keys())

    tasks = _build_topk_tasks(
        genotypes=target_genotypes,
        T_dict=T_dict,
        DAP_obs_dict=synthetic_DAP_obs,
        DL_dict=DL_dict,
        plantings_of_g_fn=plantings_of_g_fn,
        args_Tb=args.Tb, args_Topt=args.Topt, args_Tc=args.Tc,
        alpha_bounds=args.alpha_bounds,
        beta_bounds=args.beta_bounds,
        maxiter=max(50, args.maxiter // 4),  # cheaper than full Phase 1
        local_max_iter=args.local_max_iter,
        seed=args.seed,
        n_restarts=1,
        normalize_loss=args.normalize_loss,
        top_k=1,
        max_posterior_samples=1,
        predict_plantings=None,
        photo_B=None, photo_Pc=None,
        photo_type="short_day", photo_model="sinclair",
        gblup_targets=None, lambda_a=0.0, lambda_b=0.0,
        lambda_scale=None,
        topt_bounds=None, lambda_t=0.0,
        obs_perturbation_sd=0.0,
        perturb_maxiter=0,
        photo_B_bounds=None, photo_Pc_bounds=None,
        photo_a_bounds=None, photo_b_bounds=None, photo_d_bounds=None,
        photo_n_min=(photo_kwargs or {}).get("photo_n_min"),
        photo_n_opt=(photo_kwargs or {}).get("photo_n_opt"),
    )

    fit_rows = _run_fitting(
        tasks, workers=workers, worker_fn=_topk_one_genotype, verbose=verbose,
    )

    rn_targets = {}
    for r in fit_rows:
        if r.get("error") is not None:
            continue
        a = r.get("alpha")
        b = r.get("beta")
        if a is None or b is None:
            continue
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        loss = r.get("loss", 1e9)
        if not np.isfinite(loss) or loss >= 1e8:
            continue  # sentinel fit
        rn_targets[r["id"]] = (float(a), float(b))

    return rn_targets


def _run_jarquin_cached(
    *,
    pheno_df,
    weather_source,
    evd_path,
    train_plantings,
    predict_plantings,
    use_photo,
    latitude_map,
    test_pheno_df,
    jarquin_output_path: str | None = None,
):
    """Run RN-GBLUP (Jarquin 2014) once and return rows + info + by-cell map.

    Wraps cgm_wgp.jarquin_loo.jarquin_predict_dap so the same call can be
    made early (pre-Phase-1, for coupling targets and cardinal fitting) and
    its results cached for later ensemble reuse. Also optionally writes
    the per-genotype predictions CSV when jarquin_output_path is provided.

    Returns a dict with keys:
      - 'rows': list[dict] — raw jarquin prediction rows (empty on failure)
      - 'info': dict — VC components and pred_var_map
      - 'by_cell': dict[(gid_upper, env), float] — predicted DAP lookup
      - 'available': bool — True iff rows has at least one valid cell
      - 'error': str | None — exception message if call failed
    """
    result = {
        "rows": [],
        "info": {},
        "by_cell": {},
        "available": False,
        "error": None,
    }
    try:
        from cgm_wgp.jarquin_loo import jarquin_predict_dap as _jarquin_predict_dap
        rows, info = _jarquin_predict_dap(
            pheno_df=pheno_df,
            weather_source=weather_source,
            evd_path=evd_path,
            train_plantings=train_plantings,
            predict_plantings=predict_plantings,
            use_photo=use_photo,
            latitude_map=latitude_map,
            test_pheno_df=test_pheno_df,
        )
        result["rows"] = rows or []
        result["info"] = info or {}
        # Build (gid_upper, env) -> predicted_dap lookup for coupling targets
        by_cell = {}
        for r in result["rows"]:
            pred = r.get("predicted_dap")
            if pred not in (None, "NA"):
                try:
                    by_cell[(str(r["id"]).strip().upper(),
                             str(r["planting"]).strip())] = float(pred)
                except (TypeError, ValueError):
                    pass
        result["by_cell"] = by_cell
        result["available"] = bool(result["rows"])
        # Write predictions CSV if caller asked for it
        if jarquin_output_path and result["rows"]:
            with open(jarquin_output_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=[
                    "id", "planting", "observed_dap", "predicted_dap", "error", "pred_var",
                ])
                writer.writeheader()
                for prow in result["rows"]:
                    writer.writerow(prow)
    except Exception as exc:
        result["error"] = str(exc)
    return result




def _compute_param_shift(prev_params: dict, curr_params: dict) -> tuple:
    """Compute max absolute shift in alpha, beta[, topt][, B, Pc] across genotypes.

    Returns (max_alpha_shift, max_beta_shift, max_topt_shift, max_B_shift, max_Pc_shift).
    Missing parameters default to 0.0.
    """
    alpha_shifts = []
    beta_shifts = []
    topt_shifts = []
    B_shifts = []
    Pc_shifts = []
    for gid in curr_params:
        if gid in prev_params:
            curr, prev = curr_params[gid], prev_params[gid]
            alpha_shifts.append(abs(curr[0] - prev[0]))
            beta_shifts.append(abs(curr[1] - prev[1]))
            if len(curr) > 2 and len(prev) > 2:
                topt_shifts.append(abs(curr[2] - prev[2]))
            # B, Pc are always last 2 elements when present (len >= 4)
            if len(curr) >= 4 and len(prev) >= 4:
                B_shifts.append(abs(curr[-2] - prev[-2]))
                Pc_shifts.append(abs(curr[-1] - prev[-1]))
    if not alpha_shifts:
        return (float("inf"), float("inf"), float("inf"), float("inf"), float("inf"))
    return (
        max(alpha_shifts),
        max(beta_shifts),
        max(topt_shifts) if topt_shifts else 0.0,
        max(B_shifts) if B_shifts else 0.0,
        max(Pc_shifts) if Pc_shifts else 0.0,
    )




def main():
    args = parse_args()

    # Auto-create run output folder if --out and --run-dir not explicitly provided
    if args.run_dir:
        # Structured mode: set --out to the params/fitted.csv path for compatibility
        args.out = get_output_path(Path(args.run_dir), "params", "fitted.csv")
        print(f"Run output folder (structured): {args.run_dir}")
    elif args.out is None:
        location = extract_location_from_weather(args.weather)
        run_dir = create_run_dir(crop=args.crop, location=location,
                                 label=getattr(args, "run_label", None))
        args.run_dir = str(run_dir)
        args.out = get_output_path(run_dir, "params", "fitted.csv")
        print(f"Run output folder: {run_dir}")

    # Parse latitude map if provided
    latitude_map = None
    if args.latitude_map:
        import json
        latitude_map = json.loads(args.latitude_map)
        print(f"Latitude map: {latitude_map}")

    # Resolve --fit-topt and --topt-bounds
    if getattr(args, "fit_topt", False) and args.topt_bounds is None:
        args.topt_bounds = [args.Tb + 5.0, args.Tc - 5.0]
        print(f"Topt fitting enabled: bounds [{args.topt_bounds[0]:.1f}, {args.topt_bounds[1]:.1f}]")
    elif getattr(args, "fit_topt", False):
        print(f"Topt fitting enabled: bounds [{args.topt_bounds[0]:.1f}, {args.topt_bounds[1]:.1f}]")
    if not getattr(args, "fit_topt", False):
        args.topt_bounds = None  # ensure None when not fitting

    # Parse photoperiod parameters
    photo_B = None
    photo_Pc = None
    photo_type = args.photo_type
    photo_model = args.photo_model or "logistic3"

    # ── Single canonical mech+photo path ────────────────────────────
    # The ONLY supported way to combine photoperiod and thermal in the
    # mechanistic model is:
    #   Phase 1 coupling    → temp-only fit of (α, β)
    #   Phase 2 posterior   → temp-only, perturbed restarts
    #   Bolted-on photo     → fit_photo_only() (ZERO thermal) for (a, b, d)
    #   Recompute           → cardinal_beta(T) × logistic3(DL) for held-out
    # This requires photo_model=="logistic3". The legacy "sinclair" and
    # "logistic" paths JOINTLY couple α/β and photo B/Pc inside the Phase 1
    # objective, which contradicts the canonical architecture. They are
    # rejected here so a pulled config using the legacy path fails loudly
    # instead of silently running the wrong fit. "linear_plateau" is kept
    # available but the recommended path is logistic3.
    _LEGACY_JOINT_MODELS = ("sinclair", "logistic")
    if photo_model in _LEGACY_JOINT_MODELS:
        raise ValueError(
            f"photo_model='{photo_model}' is a legacy joint-coupling path "
            f"and is no longer supported in the mech+photo pipeline. The "
            f"canonical mech+photo architecture is photo_model='logistic3' "
            f"(temp-only α/β fit → pure photo-only shape fit → multiply at "
            f"recompute). Set photoperiod.model: logistic3 in your config "
            f"(with a_bounds/b_bounds/d_bounds) or pass --photo-model "
            f"logistic3."
        )
    # Logistic3 photoperiod initial values (population-level, used when not fitting per-genotype)
    photo_a_init = getattr(args, 'photo_a', None)
    photo_b_init = getattr(args, 'photo_b', None)
    photo_d_init = getattr(args, 'photo_d', None)
    if photo_model == "logistic3":
        if photo_a_init is not None and photo_b_init is not None and photo_d_init is not None:
            print(f"Photoperiod (logistic3): a={photo_a_init}, b={photo_b_init}, d={photo_d_init}")
        elif photo_a_init is not None or photo_b_init is not None or photo_d_init is not None:
            print("Warning: All of --photo-a, --photo-b, --photo-d must be provided for logistic3. Ignoring.",
                  file=sys.stderr)
            photo_a_init = None
            photo_b_init = None
            photo_d_init = None
    elif photo_B is not None and photo_Pc is not None:
        print(f"Photoperiod ({photo_model}): B={photo_B}, Pc={photo_Pc}")
    elif photo_B is not None or photo_Pc is not None:
        print("Warning: Both --photo-B and --photo-Pc must be provided for photoperiod. Ignoring.", file=sys.stderr)
        photo_B = None
        photo_Pc = None

    pheno_obs, T_dict, DAP_obs_dict, plantings_of_g_fn, DL_dict = load_and_build_dicts(
        phenotypes_csv=args.phenotypes,
        weather_csv=args.weather,
        latitude_map=latitude_map,
    )
    if DL_dict is not None and (photo_B is not None or photo_model == "logistic3"):
        n_dl = sum(len(v) for v in DL_dict.values())
        print(f"Daylength data loaded for {n_dl} genotype-planting combinations")

    # Load held-out test phenotypes for validation (balanced_training_subset CV scheme).
    # When provided, observed DAPs for validation/metrics come from this file instead
    # of pheno_obs. Fitting still uses pheno_obs (which contains only training rows in
    # this mode). Backward-compatible: when --test-phenotypes is absent, test_pheno_df
    # is None and every downstream consumer falls back to pheno_obs.
    test_pheno_df = None
    test_obs_by_cell = None  # dict: (id_str, planting_str) -> observed DAP (int)
    if getattr(args, 'test_phenotypes', None):
        test_pheno_df = pd.read_csv(args.test_phenotypes)
        # Keep only uncensored observations with a valid ft
        if "censored" in test_pheno_df.columns:
            test_pheno_df = test_pheno_df[test_pheno_df["censored"] == 0].copy()
        test_pheno_df = test_pheno_df.dropna(subset=["ft"]).copy()
        # Normalize types for lookups
        test_pheno_df["id"] = test_pheno_df["id"].astype(str)
        test_pheno_df["Planting"] = test_pheno_df["Planting"].astype(str)
        test_pheno_df["ft"] = test_pheno_df["ft"].astype(float)
        # Parse dates (jarquin needs them)
        for _col in ("Start_Date", "End_Date"):
            if _col in test_pheno_df.columns:
                test_pheno_df[_col] = pd.to_datetime(test_pheno_df[_col], errors="coerce")
        # Average reps per (id, Planting) cell so each test cell has one observed DAP
        test_pheno_df = (
            test_pheno_df
            .groupby(["id", "Planting"], as_index=False)
            .agg({
                "ft": "mean",
                **({"Start_Date": "first"} if "Start_Date" in test_pheno_df.columns else {}),
                **({"End_Date": "first"} if "End_Date" in test_pheno_df.columns else {}),
            })
        )
        test_obs_by_cell = {
            (str(r["id"]).upper().strip(), str(r["Planting"]).strip()): float(r["ft"])
            for _, r in test_pheno_df.iterrows()
        }
        n_test_envs = test_pheno_df["Planting"].nunique()
        print(f"Test phenotypes loaded: {len(test_pheno_df)} held-out cells "
              f"across {n_test_envs} environments "
              f"(used for validation, not fitting)")

    # Build reference weather per planting (for unseen genotypes)
    # Weather is location-specific, so all genotypes at same planting share temps
    ref_temps = {}   # planting -> temps array
    ref_dl = {}      # planting -> daylength array
    for _gid in T_dict:
        for _planting in T_dict[_gid]:
            if _planting not in ref_temps:
                ref_temps[_planting] = T_dict[_gid][_planting]
                if DL_dict and _gid in DL_dict and _planting in DL_dict[_gid]:
                    ref_dl[_planting] = DL_dict[_gid][_planting]

    # Load weather for predict plantings not yet in ref_temps (happens in
    # cv00_double_novelty where the predict env is completely held out and no
    # training genotype loads its weather). Weather is location-specific and
    # contains no phenotype information, so loading it is not data leakage.
    if args.predict_plantings:
        _pred_pls = parse_planting_names(args.predict_plantings)
        _weather_path = Path(args.weather)

        # Helper to slice weather by [start, end] date window for single_location
        def _slice_weather(_wdf, start, end, latitude=None):
            """Return (temps_series, daylength_array) sliced to [start, end]."""
            _sub = _wdf.loc[start:end]
            if _sub.empty:
                return None, None
            _tmean = ((_sub["tmx"] + _sub["tmn"]) / 2.0)
            _dl = None
            if "DAYLhr" in _sub.columns:
                _dl = _sub["DAYLhr"].values.astype(float)
            elif latitude is not None:
                from cgm_wgp.mech_fit import compute_daylength
                _doys = _sub.index.dayofyear.values
                _dl = np.array([compute_daylength(int(d), float(latitude)) for d in _doys])
            return _tmean, _dl

        for _pl in _pred_pls:
            if _pl in ref_temps:
                continue
            if _weather_path.is_dir():
                # Multi-location dir: each planting has its own weather.csv
                _wf = _weather_path / str(_pl) / "weather.csv"
                if _wf.exists():
                    _wdf = pd.read_csv(_wf)
                    _wdf["Period"] = pd.to_datetime(_wdf["Period"])
                    _wdf = _wdf.set_index("Period").sort_index()
                    _tmean = ((_wdf["tmx"] + _wdf["tmn"]) / 2.0)
                    ref_temps[_pl] = _tmean
                    if latitude_map and _pl in latitude_map:
                        from cgm_wgp.mech_fit import compute_daylength
                        _doys = _wdf.index.dayofyear.values
                        _lat = latitude_map[_pl]
                        ref_dl[_pl] = np.array([compute_daylength(int(d), _lat) for d in _doys])
                    print(f"  Loaded weather for held-out predict planting '{_pl}' "
                          f"({len(_tmean)} days)")
            elif _weather_path.is_file() and test_pheno_df is not None:
                # Single_location: one weather CSV, but each planting has its
                # own date window. Extract Start_Date / End_Date from test
                # phenotypes (they aren't observations, just metadata).
                _pl_rows = test_pheno_df[test_pheno_df["Planting"].astype(str) == str(_pl)]
                if _pl_rows.empty or "Start_Date" not in _pl_rows.columns:
                    continue
                _start = pd.to_datetime(_pl_rows["Start_Date"].iloc[0])
                _end = pd.to_datetime(_pl_rows["End_Date"].iloc[0])
                if pd.isna(_start) or pd.isna(_end):
                    continue
                # Load and cache single-location weather (once)
                if not hasattr(_slice_weather, "_cache"):
                    _wdf = pd.read_csv(_weather_path)
                    _wdf["Period"] = pd.to_datetime(_wdf["Period"])
                    _wdf = _wdf.set_index("Period").sort_index()
                    _slice_weather._cache = _wdf
                _lat = latitude_map.get(_pl) if latitude_map else None
                _tmean, _dl = _slice_weather(_slice_weather._cache, _start, _end, _lat)
                if _tmean is None or _tmean.empty:
                    continue
                ref_temps[_pl] = _tmean
                if _dl is not None:
                    ref_dl[_pl] = _dl
                print(f"  Loaded weather for held-out predict planting '{_pl}' "
                      f"(single_location, {len(_tmean)} days, "
                      f"{_start.date()} → {_end.date()})")

    # Augment T_dict / DL_dict with test env weather so mech prediction loops
    # can predict at envs where the genotype was held out. Weather is
    # env-specific (identical across genotypes at the same env), so we can
    # copy from ref_temps / ref_dl without introducing data leakage.
    # Only runs when test_pheno_df is provided (balanced_training_subset mode).
    if test_pheno_df is not None:
        needed_test_envs = set(test_pheno_df["Planting"].astype(str).unique())
        n_augmented = 0
        for _gid in list(T_dict.keys()):
            for _planting in needed_test_envs:
                if _planting not in T_dict[_gid] and _planting in ref_temps:
                    T_dict[_gid][_planting] = ref_temps[_planting]
                    n_augmented += 1
                    if DL_dict is not None and _gid in DL_dict and _planting in ref_dl:
                        DL_dict[_gid][_planting] = ref_dl[_planting]
        if n_augmented > 0:
            print(f"Augmented T_dict with {n_augmented} (genotype, test-env) cells "
                  f"using ref_temps (weather-only, no observations leaked)")

    # Augment T_dict / DL_dict with entries for UNSEEN test genotypes
    # (genotypes held out from training entirely — new_varieties and
    # cv00_double_novelty CV schemes). Without this, Joint CGM-WGP's
    # predict_new_genotypes loop skips these gids because it requires
    # `gmat_gid in T_dict` to get weather. Weather is env-specific so
    # copying ref_temps[planting] into a new gid slot is a zero-
    # information operation — no observations leaked. Same safety
    # argument as the per-planting augmentation above.
    if test_pheno_df is not None:
        existing_gids = set(T_dict.keys())
        test_gids = set(test_pheno_df["id"].astype(str).unique())
        new_gids = test_gids - existing_gids
        if new_gids:
            needed_test_envs = set(test_pheno_df["Planting"].astype(str).unique())
            n_cells_added = 0
            for _gid in new_gids:
                T_dict[_gid] = {}
                if DL_dict is not None:
                    DL_dict[_gid] = {}
                for _planting in needed_test_envs:
                    if _planting in ref_temps:
                        T_dict[_gid][_planting] = ref_temps[_planting]
                        n_cells_added += 1
                        if DL_dict is not None and _planting in ref_dl:
                            DL_dict[_gid][_planting] = ref_dl[_planting]
            print(f"Augmented T_dict with {len(new_gids)} unseen-genotype entries "
                  f"({n_cells_added} cells) from ref_temps "
                  f"(weather-only, no observations leaked)")

    # Parse train/predict planting specifications
    train_planting_names = None
    predict_planting_names = None
    if args.train_plantings:
        train_planting_names = parse_planting_names(args.train_plantings)
        print(f"Training plantings restricted to: {train_planting_names}")
    if args.predict_plantings:
        predict_planting_names = parse_planting_names(args.predict_plantings)
        print(f"Prediction plantings: {predict_planting_names}")

    # If training plantings are specified, wrap plantings_of_g_fn to filter
    if train_planting_names is not None:
        _orig_plantings_fn = plantings_of_g_fn
        def plantings_of_g_fn(gid):
            all_p = _orig_plantings_fn(gid)
            return [p for p in all_p if p in train_planting_names]

    # Build list of candidate genotypes from phenotypes file (preserve order)
    try:
        ids_arr = list(pheno_obs["id"].astype(str).unique())
    except Exception:
        # fallback: use keys from DAP_obs_dict
        ids_arr = list(map(str, DAP_obs_dict.keys()))

    if args.genotype:
        genotypes = [args.genotype]
    else:
        # keep only genotypes for which DAP observations exist after filtering
        genotypes = [g for g in ids_arr if g in DAP_obs_dict]
        if len(genotypes) == 0:
            raise ValueError("No genotypes found to fit (after filtering). Check phenotypes / filtering logic.")

    if args.exclude_genotypes:
        exclude_set = {g.strip() for g in args.exclude_genotypes.split(",")}
        before = len(genotypes)
        genotypes = [g for g in genotypes if g not in exclude_set]
        print(f"Excluded {before - len(genotypes)} genotype(s): {exclude_set}")

    print(f"Fitting {len(genotypes)} genotype(s). Output will be written to: {args.out}\n")

    # ── Posterior sampling (top-K dual annealing) ─────────────────────
    photo_results = None  # set by sequential photo coupling in top-K path
    # ── Top-K dual annealing with two-phase GBLUP coupling ──────────
    # Phase 1: Coupling loop (deterministic, converges)
    #   Mech MAP → GBLUP BLUPs on (α,β) → re-fit with penalty → repeat
    # Phase 2: Posterior generation (stochastic, for ensemble weighting)
    #   100 perturbed restarts with converged coupling → posterior variance
    # Single-round MAP fit (the published configuration). The multi-round
    # GBLUP-coupling/convergence machinery has been retired.
    n_rounds = 1
    convergence_only = False
    lambda_max = 0.5
    _convergence_tol = 0.01

    # Resolve EVD path for GBLUP param prediction
    _evd_path = args.evd_path or str(get_evd_path(crop=args.crop))
    _evd_available = Path(_evd_path).exists() and n_rounds > 1

    gblup_targets = None
    _lambda_scale = None
    prev_fitted_params = None
    fit_results = None
    convergence_history = []  # per-round summary for logging
    iter_detail_path = _output_path(args, "_iteration_detail.csv")
    iter_detail_header_written = False
    _topt_bounds = getattr(args, 'topt_bounds', None)

    # ── Pre-Phase-1 RN-GBLUP call (cached for Phase 1 targets + ensemble) ──
    # Run RN-GBLUP once, early, with predict_plantings = train ∪ predict
    # so that:
    #   - Phase 1 coupling targets get per-genotype predictions at ALL
    #     training envs (needed by _derive_rn_gblup_alpha_beta_targets,
    #     which requires ≥2 envs per genotype for the variance loss to
    #     have signal).
    #   - The ensemble block downstream gets predictions at predict envs.
    # Caller of the RN helper may not need predict_rows at train envs,
    # but jarquin doesn't cost materially more for the expanded set.
    _train_pl_for_rn = (
        train_planting_names if train_planting_names
        else list(set(pheno_obs["Planting"]))
    )
    _pred_pl_for_rn = predict_planting_names or []
    _combined_pred_pl = list(dict.fromkeys(
        list(_train_pl_for_rn) + list(_pred_pl_for_rn)
    ))  # preserve order, dedupe
    # NOTE: jarquin_output_path is deliberately None here — we don't
    # want the expanded (train ∪ predict) prediction rows written to
    # disk or passed to the ensemble block unfiltered. We'll write
    # jarquin.csv with predict-env-only rows after filtering below.
    _rn_cache = _run_jarquin_cached(
        pheno_df=pheno_obs,
        weather_source=args.weather,
        evd_path=_evd_path,
        train_plantings=_train_pl_for_rn,
        predict_plantings=_combined_pred_pl,
        use_photo=getattr(args, "rn_photoperiod", False),
        latitude_map=latitude_map,
        test_pheno_df=test_pheno_df,
        jarquin_output_path=None,
    )

    # Filter the cached rows/info DOWNSTREAM consumers (ensemble,
    # jarquin.csv output) see to ONLY predict envs. `rn_dap_by_cell`
    # keeps the full train ∪ predict coverage for the inversion.
    _predict_env_set = set(str(e).strip() for e in (_pred_pl_for_rn or []))
    if _rn_cache["available"] and _predict_env_set:
        _filtered_rows = [
            r for r in _rn_cache["rows"]
            if str(r.get("planting", "")).strip() in _predict_env_set
        ]
        # Also filter pred_var_map in jq_info to predict envs only so
        # the ensemble variance lookup doesn't see training-env keys.
        _pvm = _rn_cache["info"].get("pred_var_map", {})
        _filtered_pvm = {
            key: val for key, val in _pvm.items()
            if len(key) == 2 and str(key[1]).strip() in _predict_env_set
        }
        _rn_cache["rows"] = _filtered_rows
        _rn_cache["info"] = dict(_rn_cache["info"])
        _rn_cache["info"]["pred_var_map"] = _filtered_pvm
        # `by_cell` intentionally retains ALL (train + predict) cells
        # for Phase 1 coupling targets.
        # Write jarquin.csv with ONLY predict-env rows
        if _filtered_rows:
            _jq_out = _output_path(args, "_jarquin_predictions.csv")
            try:
                with open(_jq_out, "w", newline="") as _fh:
                    _w = csv.DictWriter(_fh, fieldnames=[
                        "id", "planting", "observed_dap", "predicted_dap", "error", "pred_var",
                    ])
                    _w.writeheader()
                    for _prow in _filtered_rows:
                        _w.writerow(_prow)
            except Exception as _exc:
                print(f"  Warning: failed to write {_jq_out}: {_exc}", file=sys.stderr)
    if _rn_cache["error"]:
        print(f"  Pre-Phase-1 RN-GBLUP failed: {_rn_cache['error']}", file=sys.stderr)
        print(f"  Ensemble requires RN-GBLUP. Ensemble block will retry or skip.",
              file=sys.stderr)
    elif _rn_cache["available"]:
        print(f"  Pre-Phase-1 RN-GBLUP: {len(_rn_cache['by_cell'])} cells predicted")

    # ── Derive RN-GBLUP (α, β) targets for Phase 1 coupling ──────────
    # Instead of smoothing (α, β) with a G-only GBLUP (main effects only),
    # use RN-GBLUP's G×E-aware per-cell predictions as synthetic observed
    # DAPs and refit mech against those. The resulting (α_rn, β_rn) per
    # genotype is fed to the Phase 1 penalty loop via gblup_targets.
    # When this succeeds, it replaces the G-only gblup_predict_params path.
    # When it fails or is unavailable, Phase 1 falls back to the existing
    # main-effects GBLUP smoothing.
    gblup_targets_rn_initial = None
    if _rn_cache["available"] and n_rounds > 1:
        try:
            gblup_targets_rn_initial = _derive_rn_gblup_alpha_beta_targets(
                genotypes=genotypes,
                rn_dap_by_cell=_rn_cache["by_cell"],
                T_dict=T_dict,
                DL_dict=DL_dict,
                plantings_of_g_fn=plantings_of_g_fn,
                args=args,
                photo_kwargs=None,  # Phase 1 is temp-only
                workers=args.workers,
                verbose=False,
            )
            n_targets = len(gblup_targets_rn_initial) if gblup_targets_rn_initial else 0
            min_targets = max(2, len(genotypes) // 3)  # need ≥33% coverage
            if n_targets >= min_targets:
                print(f"  RN-GBLUP → (α, β) inversion: {n_targets} targets "
                      f"(of {len(genotypes)} genotypes)")
            else:
                print(f"  RN-GBLUP → (α, β) inversion: only {n_targets} targets "
                      f"(need ≥{min_targets}), falling back to main-effects GBLUP",
                      file=sys.stderr)
                gblup_targets_rn_initial = None
        except Exception as exc:
            print(f"  RN-GBLUP → (α, β) inversion failed: {exc}", file=sys.stderr)
            gblup_targets_rn_initial = None

    # Per-genotype photoperiod fitting
    _photo_B_bounds = None
    _photo_Pc_bounds = None
    _photo_a_bounds = None
    _photo_b_bounds = None
    _photo_d_bounds = None
    gblup_photo_targets = None
    gblup_photo3_targets = None
    # Perturbed observation restarts for posterior diversity (Phase 2)
    obs_perturbation_sd = getattr(args, 'obs_perturbation_sd', 0.0)
    obs_perturbation_sd_auto = getattr(args, 'obs_perturbation_sd_auto', False)

    photo_results = None

    # Linear-plateau photoperiod params. Set by the bolted-on photo stage
    # AFTER coupling has converged. Currently None at coupling time — the
    # plumbing is wired up so future photo-aware coupling refinement can
    # use them, but the coupling loop runs temp-only.
    _pop_n_min = None
    _pop_n_opt = None

    # ── Phase 1: Coupling loop (deterministic) ──────────────────────
    # n_restarts=1, top_k=1, no perturbation — just find MAP per genotype
    coupling_lambda = lambda_max  # fixed, not ramping

    for round_idx in range(n_rounds):
        if gblup_targets:
            target_label = f"GBLUP BLUP targets for {len(gblup_targets)} genotypes"
        else:
            target_label = "no penalty (independent fitting)"

        print(f"\n{'='*60}")
        print(f"  PHASE 1 — COUPLING ROUND {round_idx + 1}/{n_rounds}")
        print(f"  Lambda = {coupling_lambda:.4f}  |  {target_label}")
        print(f"{'='*60}")

        _has_photo_targets = gblup_photo_targets is not None and len(gblup_photo_targets) > 0
        _has_photo3_targets = gblup_photo3_targets is not None and len(gblup_photo3_targets) > 0
        tasks = _build_topk_tasks(
            genotypes=genotypes,
            T_dict=T_dict, DAP_obs_dict=DAP_obs_dict, DL_dict=DL_dict,
            plantings_of_g_fn=plantings_of_g_fn,
            args_Tb=args.Tb, args_Topt=args.Topt, args_Tc=args.Tc,
            alpha_bounds=args.alpha_bounds,
            beta_bounds=args.beta_bounds,
            maxiter=args.maxiter,
            local_max_iter=args.local_max_iter,
            seed=args.seed,
            n_restarts=1,           # deterministic: single restart
            normalize_loss=args.normalize_loss,
            top_k=1,                # deterministic: just MAP
            max_posterior_samples=args.max_posterior_samples,
            predict_plantings=None,  # no predictions during coupling
            photo_B=photo_B, photo_Pc=photo_Pc, photo_type=photo_type,
            photo_model=photo_model,
            gblup_targets=gblup_targets,
            lambda_a=coupling_lambda if gblup_targets else 0.0,
            lambda_b=coupling_lambda if gblup_targets else 0.0,
            lambda_scale=_lambda_scale,
            topt_bounds=_topt_bounds,
            lambda_t=coupling_lambda if gblup_targets else 0.0,
            obs_perturbation_sd=0.0,  # no perturbation during coupling
            perturb_maxiter=0,
            photo_B_bounds=_photo_B_bounds,
            photo_Pc_bounds=_photo_Pc_bounds,
            gblup_photo_targets=gblup_photo_targets,
            lambda_bphoto=coupling_lambda if _has_photo_targets else 0.0,
            lambda_pc=coupling_lambda if _has_photo_targets else 0.0,
            # 3-param logistic (Messina) photoperiod
            photo_a_bounds=_photo_a_bounds,
            photo_b_bounds=_photo_b_bounds,
            photo_d_bounds=_photo_d_bounds,
            gblup_photo3_targets=gblup_photo3_targets,
            lambda_photo_a=coupling_lambda if _has_photo3_targets else 0.0,
            lambda_photo_b=coupling_lambda if _has_photo3_targets else 0.0,
            lambda_photo_d=coupling_lambda if _has_photo3_targets else 0.0,
            # Linear-plateau (Grimm 1993) photoperiod — pre-fit pop params
            photo_n_min=_pop_n_min,
            photo_n_opt=_pop_n_opt,
        )

        fit_results = _run_fitting(tasks, args.workers, worker_fn=_topk_one_genotype, verbose=True)
        if args.workers != 1:
            for result in fit_results:
                g_id = result["id"]
                if result["error"]:
                    print(f"  ! {g_id}: FAILED ({result['error']})")
                else:
                    topt_str = f" topt={result['topt']:.2f}" if "topt" in result and not math.isnan(result.get("topt", float("nan"))) else ""
                    photo_str = f" B={result['photo_B']:.2f} Pc={result['photo_Pc']:.2f}" if "photo_B" in result and not math.isnan(result.get("photo_B", float("nan"))) else ""
                    print(f"  {g_id}: alpha={result['alpha']:.6f} beta={result['beta']:.6f}{topt_str}{photo_str} loss={result['loss']:.6e}")

        # Extract fitted params for GBLUP and convergence check
        fitted_params = {}
        for r in fit_results:
            if r["error"] is None and np.isfinite(r["alpha"]) and np.isfinite(r["beta"]):
                params_tuple = [r["alpha"], r["beta"]]
                if "topt" in r and not math.isnan(r.get("topt", float("nan"))):
                    params_tuple.append(r["topt"])
                if "photo_B" in r and not math.isnan(r.get("photo_B", float("nan"))):
                    params_tuple.extend([r["photo_B"], r["photo_Pc"]])
                fitted_params[r["id"]] = tuple(params_tuple)

        # Write per-round detail CSV
        if n_rounds > 1:
            _has_topt_col = _topt_bounds is not None
            _has_photo_col = _photo_B_bounds is not None
            detail_fields = ["round", "genotype", "alpha", "beta"]
            if _has_topt_col:
                detail_fields.append("topt")
            if _has_photo_col:
                detail_fields.extend(["photo_B", "photo_Pc"])
            detail_fields += ["loss", "alpha_std", "beta_std"]
            if _has_topt_col:
                detail_fields.append("topt_std")
            if _has_photo_col:
                detail_fields.extend(["photo_B_std", "photo_Pc_std"])
            detail_fields += ["n_accepted", "lambda"]
            mode = "w" if not iter_detail_header_written else "a"
            with open(iter_detail_path, mode, newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=detail_fields)
                if not iter_detail_header_written:
                    writer.writeheader()
                    iter_detail_header_written = True
                for r in fit_results:
                    if r["error"] is None:
                        a_std = r.get("alpha_std", float("nan"))
                        b_std = r.get("beta_std", float("nan"))
                        row = {
                            "round": round_idx + 1,
                            "genotype": r["id"],
                            "alpha": f"{r['alpha']:.6f}",
                            "beta": f"{r['beta']:.6f}",
                            "loss": f"{r['loss']:.6e}" if r.get("loss") is not None else "",
                            "alpha_std": f"{a_std:.4f}" if isinstance(a_std, (int, float)) and np.isfinite(a_std) else "",
                            "beta_std": f"{b_std:.4f}" if isinstance(b_std, (int, float)) and np.isfinite(b_std) else "",
                            "n_accepted": r.get("n_accepted", ""),
                            "lambda": f"{coupling_lambda:.4f}",
                        }
                        if _has_topt_col:
                            t_val = r.get("topt", float("nan"))
                            t_std = r.get("topt_std", float("nan"))
                            row["topt"] = f"{t_val:.4f}" if isinstance(t_val, (int, float)) and np.isfinite(t_val) else ""
                            row["topt_std"] = f"{t_std:.4f}" if isinstance(t_std, (int, float)) and np.isfinite(t_std) else ""
                        if _has_photo_col:
                            bv = r.get("photo_B", float("nan"))
                            pcv = r.get("photo_Pc", float("nan"))
                            bs = r.get("photo_B_std", float("nan"))
                            pcs = r.get("photo_Pc_std", float("nan"))
                            row["photo_B"] = f"{bv:.4f}" if isinstance(bv, (int, float)) and np.isfinite(bv) else ""
                            row["photo_Pc"] = f"{pcv:.4f}" if isinstance(pcv, (int, float)) and np.isfinite(pcv) else ""
                            row["photo_B_std"] = f"{bs:.4f}" if isinstance(bs, (int, float)) and np.isfinite(bs) else ""
                            row["photo_Pc_std"] = f"{pcs:.4f}" if isinstance(pcs, (int, float)) and np.isfinite(pcs) else ""
                        writer.writerow(row)

        # Auto-estimate obs_perturbation_sd after Round 1 (for Phase 2)
        if round_idx == 0 and obs_perturbation_sd_auto:
            genotype_rmses = []
            for r in fit_results:
                if r["error"] is None and r.get("loss") is not None and np.isfinite(r["loss"]):
                    n_train = r.get("n_training_plantings", 1)
                    if n_train > 0:
                        rmse = math.sqrt(r["loss"] / n_train)
                        genotype_rmses.append(rmse)
            if genotype_rmses:
                obs_perturbation_sd = float(np.median(genotype_rmses))
                print(f"\n  Auto-estimated obs_perturbation_sd = {obs_perturbation_sd:.2f} days "
                      f"(from Round 1 training RMSE, n={len(genotype_rmses)} genotypes)")
            else:
                print(f"\n  Warning: Could not auto-estimate obs_perturbation_sd (no valid Round 1 results)")

        # Build round summary for convergence history
        round_summary = {
            "round": round_idx + 1,
            "n_genotypes": len(fitted_params),
            "mean_alpha": f"{np.mean([p[0] for p in fitted_params.values()]):.4f}" if fitted_params else "",
            "mean_beta": f"{np.mean([p[1] for p in fitted_params.values()]):.4f}" if fitted_params else "",
            "std_alpha": f"{np.std([p[0] for p in fitted_params.values()]):.4f}" if fitted_params else "",
            "std_beta": f"{np.std([p[1] for p in fitted_params.values()]):.4f}" if fitted_params else "",
            "lambda": f"{coupling_lambda:.4f}",
            "max_alpha_shift": "",
            "max_beta_shift": "",
            "converged": "no",
        }
        if _topt_bounds is not None and fitted_params:
            topt_vals_round = [p[2] for p in fitted_params.values() if len(p) > 2 and np.isfinite(p[2])]
            if topt_vals_round:
                round_summary["mean_topt"] = f"{np.mean(topt_vals_round):.4f}"
                round_summary["std_topt"] = f"{np.std(topt_vals_round):.4f}"
            round_summary["max_topt_shift"] = ""
        if _photo_B_bounds is not None and fitted_params:
            B_vals_round = [p[-2] for p in fitted_params.values() if len(p) >= 4]
            Pc_vals_round = [p[-1] for p in fitted_params.values() if len(p) >= 4]
            if B_vals_round:
                round_summary["mean_photo_B"] = f"{np.mean(B_vals_round):.4f}"
                round_summary["mean_photo_Pc"] = f"{np.mean(Pc_vals_round):.4f}"
                round_summary["std_photo_B"] = f"{np.std(B_vals_round):.4f}"
                round_summary["std_photo_Pc"] = f"{np.std(Pc_vals_round):.4f}"
            round_summary["max_B_shift"] = ""
            round_summary["max_Pc_shift"] = ""

        # Check convergence (from round 2 onward — round 1 has no penalty)
        if round_idx > 0 and prev_fitted_params:
            param_shifts = _compute_param_shift(prev_fitted_params, fitted_params)
            max_alpha_shift, max_beta_shift = param_shifts[0], param_shifts[1]
            max_topt_shift = param_shifts[2]
            max_B_shift = param_shifts[3]
            max_Pc_shift = param_shifts[4]
            topt_shift_str = f", max topt shift = {max_topt_shift:.6f}" if _topt_bounds is not None else ""
            photo_shift_str = f", max B shift = {max_B_shift:.4f}, max Pc shift = {max_Pc_shift:.4f}" if _photo_B_bounds is not None else ""
            print(f"\n  Convergence: max alpha shift = {max_alpha_shift:.6f}, max beta shift = {max_beta_shift:.6f}{topt_shift_str}{photo_shift_str}")
            round_summary["max_alpha_shift"] = f"{max_alpha_shift:.6f}"
            round_summary["max_beta_shift"] = f"{max_beta_shift:.6f}"
            if _topt_bounds is not None:
                round_summary["max_topt_shift"] = f"{max_topt_shift:.6f}"
            if _photo_B_bounds is not None:
                round_summary["max_B_shift"] = f"{max_B_shift:.6f}"
                round_summary["max_Pc_shift"] = f"{max_Pc_shift:.6f}"
            all_converged = max_alpha_shift < _convergence_tol and max_beta_shift < _convergence_tol
            if _topt_bounds is not None:
                all_converged = all_converged and max_topt_shift < _convergence_tol
            if _photo_B_bounds is not None:
                all_converged = all_converged and max_B_shift < _convergence_tol and max_Pc_shift < _convergence_tol
            if all_converged:
                round_summary["converged"] = "yes"
                print(f"  CONVERGED at round {round_idx + 1}")
                convergence_history.append(round_summary)
                break

        convergence_history.append(round_summary)
        prev_fitted_params = dict(fitted_params)

        # GBLUP smoothing on fitted params for next round.
        # Preferred path: RN-GBLUP-derived targets (G×E-aware). These
        # are computed once pre-Phase-1 from RN-GBLUP's per-cell DAP
        # predictions and reused each round — they don't depend on the
        # round's fitted_params (which is intentional: the targets are
        # a stationary reference derived from raw data, not the fit).
        # Fallback: main-effects GBLUP via gblup_predict_params().
        if round_idx < n_rounds - 1 and len(fitted_params) >= 2:
            if gblup_targets_rn_initial is not None:
                # RN-GBLUP path: use pre-computed (α, β) targets.
                gblup_targets = dict(gblup_targets_rn_initial)
                # No per-gid λ scaling from PEV — use flat λ for all gids
                _lambda_scale = None
                # Photo targets still need G-only GBLUP when per-gid
                # photoperiod fitting is active (RN-GBLUP's output
                # doesn't give us per-gid photo params).
                if _photo_B_bounds is not None and _evd_available:
                    try:
                        from cgm_wgp.gblup import gblup_predict_params
                        gblup_preds = gblup_predict_params(
                            _evd_path, fitted_params, loo=True, has_photo=True,
                        )
                        gblup_photo_targets = {
                            gid: (vals[2], vals[3]) for gid, vals in gblup_preds.items()
                        }
                    except Exception as exc:
                        print(f"  Warning: photo target GBLUP failed: {exc}",
                              file=sys.stderr)
                        gblup_photo_targets = None
                else:
                    gblup_photo_targets = None
                # Log RN-GBLUP target shifts
                rn_shifts_a = []
                rn_shifts_b = []
                for gid, tgt in gblup_targets.items():
                    if gid in fitted_params:
                        fp = fitted_params[gid]
                        rn_shifts_a.append(abs(tgt[0] - fp[0]))
                        rn_shifts_b.append(abs(tgt[1] - fp[1]))
                if rn_shifts_a:
                    print(f"\n  RN-GBLUP target shifts: "
                          f"alpha median={np.median(rn_shifts_a):.4f} max={max(rn_shifts_a):.4f}, "
                          f"beta median={np.median(rn_shifts_b):.4f} max={max(rn_shifts_b):.4f}")
                print(f"  RN-GBLUP (α, β) targets for {len(gblup_targets)} genotypes "
                      f"(next round lambda = {coupling_lambda:.4f})")
            elif _evd_available:
                # FALLBACK: main-effects G-only GBLUP on fitted (α, β)
                try:
                    from cgm_wgp.gblup import gblup_predict_params
                    _fit_photo_active = _photo_B_bounds is not None
                    gblup_preds = gblup_predict_params(
                        _evd_path, fitted_params, loo=True,
                        has_photo=_fit_photo_active,
                    )
                    gblup_targets = {}
                    gblup_photo_targets = {} if _fit_photo_active else None
                    _lambda_scale = {}
                    pev_alpha_list = []
                    for gid, vals in gblup_preds.items():
                        if _fit_photo_active:
                            gblup_targets[gid] = (vals[0], vals[1])
                            gblup_photo_targets[gid] = (vals[2], vals[3])
                            pev_alpha_list.append(vals[4])
                        else:
                            gblup_targets[gid] = (vals[0], vals[1])
                            pev_alpha_list.append(vals[2])
                    if pev_alpha_list:
                        pev_arr = np.array(pev_alpha_list)
                        inv_pev = 1.0 / np.maximum(pev_arr, 1e-10)
                        inv_pev_norm = inv_pev / inv_pev.mean() if inv_pev.mean() > 0 else np.ones_like(inv_pev)
                        for i, gid in enumerate(gblup_preds.keys()):
                            _lambda_scale[gid] = (float(inv_pev_norm[i]), float(inv_pev_norm[i]))
                    nbr_shifts_a = []
                    nbr_shifts_b = []
                    for gid, tgt in gblup_targets.items():
                        if gid in fitted_params:
                            fp = fitted_params[gid]
                            nbr_shifts_a.append(abs(tgt[0] - fp[0]))
                            nbr_shifts_b.append(abs(tgt[1] - fp[1]))
                    if nbr_shifts_a:
                        print(f"\n  GBLUP LOO target shifts: "
                              f"alpha median={np.median(nbr_shifts_a):.4f} max={max(nbr_shifts_a):.4f}, "
                              f"beta median={np.median(nbr_shifts_b):.4f} max={max(nbr_shifts_b):.4f}")
                    print(f"  GBLUP BLUP targets (main-effects fallback) for "
                          f"{len(gblup_targets)} genotypes "
                          f"(next round lambda = {coupling_lambda:.4f})")
                except Exception as exc:
                    print(f"\n  Warning: GBLUP param smoothing failed: {exc}. "
                          f"Continuing without penalty.", file=sys.stderr)
                    gblup_targets = None
                    gblup_photo_targets = None
                    _lambda_scale = None

    # ── Phase 2: Posterior generation (stochastic) ──────────────────
    # Run perturbed restarts with converged coupling to get posterior
    # for ensemble variance-based weighting
    if obs_perturbation_sd > 0 or obs_perturbation_sd_auto:
        # Auto-adjust n_restarts for posterior generation
        if obs_perturbation_sd > 0 and args.n_restarts < args.top_k:
            print(f"\n  Auto-setting n_restarts={args.top_k} to match top_k "
                  f"(was {args.n_restarts}, need 1 MAP per restart for posterior diversity)")
            args.n_restarts = args.top_k

        print(f"\n{'='*60}")
        print(f"  PHASE 2 — POSTERIOR GENERATION")
        print(f"  {args.n_restarts} perturbed restarts, top_k={args.top_k}")
        print(f"  obs_perturbation_sd = {obs_perturbation_sd:.2f}")
        if gblup_targets:
            print(f"  Coupling: lambda={coupling_lambda:.4f} with GBLUP BLUP targets")
        print(f"{'='*60}")

        _has_photo_targets_p2 = gblup_photo_targets is not None and len(gblup_photo_targets) > 0
        _has_photo3_targets_p2 = gblup_photo3_targets is not None and len(gblup_photo3_targets) > 0
        posterior_tasks = _build_topk_tasks(
            genotypes=genotypes,
            T_dict=T_dict, DAP_obs_dict=DAP_obs_dict, DL_dict=DL_dict,
            plantings_of_g_fn=plantings_of_g_fn,
            args_Tb=args.Tb, args_Topt=args.Topt, args_Tc=args.Tc,
            alpha_bounds=args.alpha_bounds,
            beta_bounds=args.beta_bounds,
            maxiter=args.maxiter,
            local_max_iter=args.local_max_iter,
            seed=args.seed,
            n_restarts=args.n_restarts,
            normalize_loss=args.normalize_loss,
            top_k=args.top_k,
            max_posterior_samples=args.max_posterior_samples,
            predict_plantings=predict_planting_names,
            photo_B=photo_B, photo_Pc=photo_Pc, photo_type=photo_type,
            photo_model=photo_model,
            gblup_targets=gblup_targets,
            lambda_a=coupling_lambda if gblup_targets else 0.0,
            lambda_b=coupling_lambda if gblup_targets else 0.0,
            lambda_scale=_lambda_scale,
            topt_bounds=_topt_bounds,
            lambda_t=coupling_lambda if gblup_targets else 0.0,
            obs_perturbation_sd=obs_perturbation_sd,
            perturb_maxiter=getattr(args, 'perturb_maxiter', 0),
            photo_B_bounds=_photo_B_bounds,
            photo_Pc_bounds=_photo_Pc_bounds,
            gblup_photo_targets=gblup_photo_targets,
            lambda_bphoto=coupling_lambda if _has_photo_targets_p2 else 0.0,
            lambda_pc=coupling_lambda if _has_photo_targets_p2 else 0.0,
            # 3-param logistic (Messina) photoperiod
            photo_a_bounds=_photo_a_bounds,
            photo_b_bounds=_photo_b_bounds,
            photo_d_bounds=_photo_d_bounds,
            gblup_photo3_targets=gblup_photo3_targets,
            lambda_photo_a=coupling_lambda if _has_photo3_targets_p2 else 0.0,
            lambda_photo_b=coupling_lambda if _has_photo3_targets_p2 else 0.0,
            lambda_photo_d=coupling_lambda if _has_photo3_targets_p2 else 0.0,
            # Linear-plateau (Grimm 1993) photoperiod — pre-fit pop params
            photo_n_min=_pop_n_min,
            photo_n_opt=_pop_n_opt,
        )

        fit_results = _run_fitting(posterior_tasks, args.workers,
                                   worker_fn=_topk_one_genotype, verbose=True)
        if args.workers != 1:
            for result in fit_results:
                g_id = result["id"]
                if result["error"]:
                    print(f"  ! {g_id}: FAILED ({result['error']})")
                else:
                    extra = ""
                    if not math.isnan(result.get("alpha_std", float("nan"))):
                        extra = f" | top-K: n={result['n_accepted']}, alpha_std={result['alpha_std']:.4f}, beta_std={result['beta_std']:.4f}"
                    print(f"  {g_id}: alpha={result['alpha']:.6f} beta={result['beta']:.6f} loss={result['loss']:.6e}{extra}")

        # Write Phase 2 detail to iteration detail CSV
        if n_rounds > 1:
            _has_topt_col = _topt_bounds is not None
            detail_fields = ["round", "genotype", "alpha", "beta"]
            if _has_topt_col:
                detail_fields.append("topt")
            detail_fields += ["loss", "alpha_std", "beta_std"]
            if _has_topt_col:
                detail_fields.append("topt_std")
            detail_fields += ["n_accepted", "lambda"]
            with open(iter_detail_path, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=detail_fields)
                for r in fit_results:
                    if r["error"] is None:
                        a_std = r.get("alpha_std", float("nan"))
                        b_std = r.get("beta_std", float("nan"))
                        row = {
                            "round": "posterior",
                            "genotype": r["id"],
                            "alpha": f"{r['alpha']:.6f}",
                            "beta": f"{r['beta']:.6f}",
                            "loss": f"{r['loss']:.6e}" if r.get("loss") is not None else "",
                            "alpha_std": f"{a_std:.4f}" if isinstance(a_std, (int, float)) and np.isfinite(a_std) else "",
                            "beta_std": f"{b_std:.4f}" if isinstance(b_std, (int, float)) and np.isfinite(b_std) else "",
                            "n_accepted": r.get("n_accepted", ""),
                            "lambda": f"{coupling_lambda:.4f}",
                        }
                        if _has_topt_col:
                            t_val = r.get("topt", float("nan"))
                            t_std = r.get("topt_std", float("nan"))
                            row["topt"] = f"{t_val:.4f}" if isinstance(t_val, (int, float)) and np.isfinite(t_val) else ""
                            row["topt_std"] = f"{t_std:.4f}" if isinstance(t_std, (int, float)) and np.isfinite(t_std) else ""
                        writer.writerow(row)
    else:
        # No perturbation — use Phase 1 results directly but need predictions
        if predict_planting_names:
            print(f"\n  Running prediction pass (no posterior perturbation)...")
            _has_photo_targets_pred = gblup_photo_targets is not None and len(gblup_photo_targets) > 0
            pred_tasks = _build_topk_tasks(
                genotypes=genotypes,
                T_dict=T_dict, DAP_obs_dict=DAP_obs_dict, DL_dict=DL_dict,
                plantings_of_g_fn=plantings_of_g_fn,
                args_Tb=args.Tb, args_Topt=args.Topt, args_Tc=args.Tc,
                alpha_bounds=args.alpha_bounds,
                beta_bounds=args.beta_bounds,
                maxiter=args.maxiter,
                local_max_iter=args.local_max_iter,
                seed=args.seed,
                n_restarts=args.n_restarts,
                normalize_loss=args.normalize_loss,
                top_k=args.top_k,
                max_posterior_samples=args.max_posterior_samples,
                predict_plantings=predict_planting_names,
                photo_B=photo_B, photo_Pc=photo_Pc, photo_type=photo_type,
                photo_model=photo_model,
                gblup_targets=gblup_targets,
                lambda_a=coupling_lambda if gblup_targets else 0.0,
                lambda_b=coupling_lambda if gblup_targets else 0.0,
                lambda_scale=_lambda_scale,
                topt_bounds=_topt_bounds,
                lambda_t=coupling_lambda if gblup_targets else 0.0,
                obs_perturbation_sd=0.0,
                perturb_maxiter=0,
                photo_B_bounds=_photo_B_bounds,
                photo_Pc_bounds=_photo_Pc_bounds,
                gblup_photo_targets=gblup_photo_targets,
                lambda_bphoto=coupling_lambda if _has_photo_targets_pred else 0.0,
                lambda_pc=coupling_lambda if _has_photo_targets_pred else 0.0,
            )
            fit_results = _run_fitting(pred_tasks, args.workers,
                                       worker_fn=_topk_one_genotype, verbose=True)

    # Write convergence history CSV
    if n_rounds > 1 and convergence_history:
        conv_out = _output_path(args, "_convergence_history.csv")
        conv_fields = ["round", "n_genotypes", "mean_alpha", "mean_beta"]
        if _topt_bounds is not None:
            conv_fields.append("mean_topt")
        conv_fields += ["std_alpha", "std_beta"]
        if _topt_bounds is not None:
            conv_fields.append("std_topt")
        conv_fields += ["lambda", "max_alpha_shift", "max_beta_shift"]
        if _topt_bounds is not None:
            conv_fields.append("max_topt_shift")
        if _photo_B_bounds is not None:
            conv_fields += ["mean_photo_B", "std_photo_B", "max_B_shift",
                            "mean_photo_Pc", "std_photo_Pc", "max_Pc_shift"]
        conv_fields.append("converged")
        with open(conv_out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=conv_fields)
            writer.writeheader()
            writer.writerows(convergence_history)
        print(f"\n  Wrote convergence history ({len(convergence_history)} rounds) to {conv_out}")
        print(f"  Wrote per-round detail to {iter_detail_path}")

    # ── Linear-plateau photoperiod fitting (Grimm et al. 1993) ────────
    # Runs AFTER temperature coupling and Phase 2 so we have real
    # per-genotype alpha/beta. Fits population (n_min, n_opt) with the
    # converged thermal model, then recomputes posterior predictive DAP
    # per (genotype, env). When photo prediction fails (F(N) ≈ 0 for
    # the entire prediction window — typical when held-out env DL is
    # outside training range), the temp-only Phase 2 prediction is
    # preserved as an implicit fallback. Gated on
    # photo_model == "linear_plateau" — all other paths unchanged.
    # --mech-no-photo skips the bolted-on stage entirely (temp-only mech)
    _mech_no_photo = getattr(args, "mech_no_photo", False)
    if _mech_no_photo:
        print("\n--mech-no-photo: skipping bolted-on photoperiod stage (temp-only mech)")
    if (photo_model in ("linear_plateau", "logistic3")
            and DL_dict and fit_results
            and not _mech_no_photo):
        print("\n" + "="*60)
        if photo_model == "linear_plateau":
            print("  PHOTOPERIOD STAGE: Linear-Plateau (Grimm 1993)")
        else:
            print("  PHOTOPERIOD STAGE: Logistic3 (Messina, DAP-error loss)")
        print("="*60)

        # Extract fitted alpha/beta from temp-only results
        fitted_alpha = {}
        fitted_beta = {}
        for r in fit_results:
            gid = r["id"]
            a = r.get("alpha")
            b = r.get("beta")
            if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                fitted_alpha[gid] = float(a)
                fitted_beta[gid] = float(b)

        # Dual-threshold mode flag — set inside the logistic3 branch
        # only. Tracked here so the multiplicative recompute (which
        # follows both branches) can be skipped if dual-threshold
        # already wrote the predictions.
        _dual_threshold_applied = False

        # Branch on photo_model: fit either linear_plateau (n_min, n_opt) or
        # logistic3 (a, b, d) with DAP-error loss. Both use the same
        # downstream recompute/fallback logic.
        pop_photo = None
        photo_kwargs = None   # kwargs to pass to daily_development_rate/predict_flowering_dap
        if photo_model == "linear_plateau":
            _n_min_bounds = tuple(getattr(args, 'photo_n_min_bounds', None) or (6.0, 14.0))
            _n_opt_bounds = tuple(getattr(args, 'photo_n_opt_bounds', None) or (8.0, 16.0))
            print(f"  N_min bounds: {_n_min_bounds} (night length hours)")
            print(f"  N_opt bounds: {_n_opt_bounds} (night length hours)")
            from cgm_wgp.mech_fit import fit_population_photo_linplat
            pop_photo = fit_population_photo_linplat(
                genotypes=list(fitted_alpha.keys()),
                fitted_alpha=fitted_alpha,
                fitted_beta=fitted_beta,
                temp_base=args.Tb, temp_optimal=args.Topt, temp_critical=args.Tc,
                plantings_of_g_fn=plantings_of_g_fn,
                T_dict=T_dict, DL_dict=DL_dict, DAP_obs_dict=DAP_obs_dict,
                n_min_bounds=_n_min_bounds,
                n_opt_bounds=_n_opt_bounds,
                maxiter=args.maxiter,
                seed=args.seed,
            )
        else:  # logistic3
            _a_bounds = tuple(getattr(args, 'photo_a_bounds', None) or (0.1, 5.0))
            _b_bounds = tuple(getattr(args, 'photo_b_bounds', None) or (-5.0, 5.0))
            _d_bounds = tuple(getattr(args, 'photo_d_bounds', None) or (10.0, 18.0))
            print(f"  a bounds: {_a_bounds}")
            print(f"  b bounds: {_b_bounds}")
            print(f"  d bounds: {_d_bounds} (daylength hours)")
            print(f"  photo_type: {photo_type} (applies sign constraint on b)")
            # Constrain b sign based on photo_type so the optimizer picks
            # the physiologically correct direction (short-day plants
            # want long nights → b > 0; long-day plants want short
            # nights → b < 0). Without this, a tropical-only training
            # set has degenerate freedom to pick either direction, and
            # the wrong choice extrapolates catastrophically to extreme
            # latitudes.
            _b_lo, _b_hi = float(_b_bounds[0]), float(_b_bounds[1])
            if photo_type == "short_day":
                _b_lo = max(_b_lo, 0.0)
            elif photo_type == "long_day":
                _b_hi = min(_b_hi, 0.0)
            if _b_lo >= _b_hi:
                _b_lo, _b_hi = (0.0, 0.01) if photo_type == "short_day" else (-0.01, 0.0)
            _b_bounds_eff = [_b_lo, _b_hi]

            # Strategy: run photo-only (no temperature) to find the
            # logistic3 shape that best explains flowering variance
            # WITHOUT thermal confounding. Photo-only's adaptive per-
            # genotype H gives a smoother loss landscape and avoids
            # the degenerate-steep solutions that the thermal-coupled
            # dap-error loss finds. Then use those (a, b, d) with
            # thermal during prediction, with target_H recomputed per
            # genotype from training cells (cum(thermal × F_photo)).
            #
            # This decouples the SHAPE fit (photo-only, no thermal) from
            # the THRESHOLD fit (mech+photo recompute), which eliminates
            # the fragile fixed-threshold fit while preserving the mech
            # infrastructure for prediction.
            from cgm_wgp.mech_fit import (
                fit_photo_only,
                photo_rates_from_model,
            )
            _photo_only_fit = fit_photo_only(
                genotypes=list(fitted_alpha.keys()),
                plantings_of_g_fn=plantings_of_g_fn,
                DL_dict=DL_dict,
                DAP_obs_dict=DAP_obs_dict,
                photo_model="logistic3",
                shape_bounds=[list(_a_bounds), _b_bounds_eff, list(_d_bounds)],
                shape_param_names=["a", "b", "d"],
                photo_type=photo_type,
                maxiter=args.maxiter,
                seed=args.seed,
            )
            _shape = _photo_only_fit["shape_params"]
            # Compute temp-only baseline SSE for the improvement metric
            _temp_only_sse = 0.0
            _n_count = 0
            for _g in fitted_alpha.keys():
                _a_g = fitted_alpha.get(_g)
                _b_g = fitted_beta.get(_g)
                if _a_g is None or _b_g is None:
                    continue
                _threshold = 1.0 / _a_g if _a_g > 0 else 1.0
                _sse = 0.0
                _n_valid = 0
                for _e in plantings_of_g_fn(_g):
                    try:
                        _temps_e = np.asarray(T_dict[_g][_e], dtype=float)
                        _dap_obs = int(DAP_obs_dict[_g][_e])
                    except Exception:
                        continue
                    from cgm_wgp.mech_fit import (
                        cardinal_beta as _cardinal_beta,
                        simulate_dap_to_threshold as _sim,
                    )
                    _thermal = _cardinal_beta(_temps_e, args.Tb, args.Topt, args.Tc, _a_g, _b_g)
                    _ones = np.ones(len(_thermal))
                    _dap_pred = _sim(_thermal, _ones, _threshold)
                    _sse += (_dap_pred - _dap_obs) ** 2
                    _n_valid += 1
                if _n_valid > 0:
                    _temp_only_sse += _sse / _n_valid
                    _n_count += 1
            _temp_only_loss = _temp_only_sse / max(1, _n_count)

            # Build a pop_photo dict that mirrors the linear_plateau
            # return structure so downstream logic is unchanged.
            pop_photo = {
                "photo_a": float(_shape["a"]),
                "photo_b": float(_shape["b"]),
                "photo_d": float(_shape["d"]),
                "loss": float(_photo_only_fit["loss"]),
                "temp_only_loss": float(_temp_only_loss),
                # Use photo-only loss as the "improvement" signal:
                # photo-only's loss is already an SSE in days², directly
                # comparable to the temp-only SSE.
                "improvement": float(_temp_only_loss - _photo_only_fit["loss"]),
            }
            print(f"  Population logistic3 photoperiod (Messina, via photo-only fit):")
            print(f"    a={pop_photo['photo_a']:.4f}, b={pop_photo['photo_b']:.4f}, d={pop_photo['photo_d']:.3f}h")
            print(f"    Floor: 1/(1+a) = {1.0/(1.0+pop_photo['photo_a']):.4f}")
            print(f"    Photo-only loss (SSE/n): {pop_photo['loss']:.2f}  |  "
                  f"Temp-only baseline: {pop_photo['temp_only_loss']:.2f}")
            if pop_photo['improvement'] > 0.5:
                print(f"    -> Photoperiod IMPROVES fit (delta = {pop_photo['improvement']:.2f})")
            else:
                print(f"    -> Photoperiod provides no signal (delta = {pop_photo['improvement']:.2f})")
            for _dl in [10, 12, 14, 16]:
                _gp = photo_rates_from_model([_dl], "logistic3", _shape, photo_type)[0]
                print(f"    DL={_dl}h: F(DL) = {_gp:.4f}")

            # ── Dual-threshold mode (--mech-dual-threshold) ────────
            # Replaces the multiplicative cardinal_beta(T) × F_photo(DL)
            # recompute with two INDEPENDENT state variables and a max()
            # combiner. Thermal threshold H_thermal_g comes from the
            # temp-only Phase 1 fit; photo threshold H_photo_g comes
            # from fit_photo_only's adaptive per-genotype H. Predict
            # pred_dap = max(day_thermal_done, day_photo_done) so a
            # plant only flowers when BOTH developmental clocks have
            # completed. No multiplicative substitution → no variance-
            # loss degeneracy.
            if (getattr(args, "mech_dual_threshold", False)
                    and pop_photo.get("improvement", 0.0) > 0.5):
                from cgm_wgp.mech_fit import (
                    cardinal_beta as _cb_dt,
                )
                print("\n  ── Dual-threshold mech+photo mode ──")
                # Per-genotype thermal threshold from Phase 1 fit:
                # mean of cum(cardinal_beta(T)) at observed DAP across
                # this genotype's training cells.
                _H_thermal: Dict[str, float] = {}
                for _gid, _a_g in fitted_alpha.items():
                    _b_g = fitted_beta.get(_gid)
                    if _a_g is None or _b_g is None or _gid not in T_dict:
                        continue
                    _h_vals: List[float] = []
                    for _e in plantings_of_g_fn(_gid):
                        try:
                            _temps_e = np.asarray(T_dict[_gid][_e], dtype=float)
                            _dap_e = int(DAP_obs_dict[_gid][_e])
                        except Exception:
                            continue
                        _thermal = _cb_dt(_temps_e, args.Tb, args.Topt,
                                          args.Tc, _a_g, _b_g)
                        _cut = max(0, min(_dap_e, _thermal.size))
                        _h_vals.append(float(np.sum(_thermal[:_cut])))
                    if _h_vals:
                        _H_thermal[_gid] = float(np.mean(_h_vals))
                # Per-genotype photo threshold: reuse fit_photo_only's output
                _H_photo: Dict[str, float] = dict(
                    _photo_only_fit.get("h_per_genotype", {})
                )
                print(f"  H_thermal computed for {len(_H_thermal)} genotypes")
                print(f"  H_photo computed for {len(_H_photo)} genotypes")
                # Recompute predictions for held-out cells via max()
                if predict_planting_names:
                    n_dt_ok = 0
                    n_dt_skip = 0
                    n_t_binding = 0  # day_thermal > day_photo
                    n_p_binding = 0  # day_photo > day_thermal
                    for r in fit_results:
                        if r.get("error") is not None:
                            continue
                        _gid = r["id"]
                        _a_hat = r.get("alpha")
                        _b_hat = r.get("beta")
                        if not (_a_hat is not None and _b_hat is not None
                                and np.isfinite(_a_hat) and np.isfinite(_b_hat)):
                            continue
                        if _gid not in _H_thermal or _gid not in _H_photo:
                            n_dt_skip += 1
                            continue
                        _new_pp: Dict[str, Dict[str, float]] = {}
                        for _pred_e in predict_planting_names:
                            if _gid not in T_dict or _pred_e not in T_dict[_gid]:
                                continue
                            _temps_pred = np.asarray(T_dict[_gid][_pred_e], dtype=float)
                            _dl_pred = (DL_dict.get(_gid, {}).get(_pred_e)
                                        if DL_dict else None)
                            if _dl_pred is None:
                                continue
                            _dl_pred = np.asarray(_dl_pred, dtype=float)
                            # Independent integrals: thermal vs photo
                            _thermal_pred = _cb_dt(_temps_pred, args.Tb,
                                                   args.Topt, args.Tc,
                                                   _a_hat, _b_hat)
                            _photo_pred = photo_rates_from_model(
                                _dl_pred, "logistic3", _shape, photo_type,
                            )
                            _n_pred = min(len(_thermal_pred), len(_photo_pred))
                            if _n_pred == 0:
                                continue
                            _cum_t = np.cumsum(_thermal_pred[:_n_pred])
                            _cum_p = np.cumsum(_photo_pred[:_n_pred])
                            _idx_t = int(np.searchsorted(
                                _cum_t, _H_thermal[_gid], side="left"))
                            _idx_p = int(np.searchsorted(
                                _cum_p, _H_photo[_gid], side="left"))
                            # If never reached → use the weather window
                            # length (treats "infinitely late" as
                            # bounded by what we can observe). max()
                            # then picks the slower channel.
                            _day_t = _idx_t + 1 if _idx_t < _n_pred else _n_pred
                            _day_p = _idx_p + 1 if _idx_p < _n_pred else _n_pred
                            _pred_dap = max(_day_t, _day_p)
                            if _day_t > _day_p:
                                n_t_binding += 1
                            else:
                                n_p_binding += 1
                            _new_pp[_pred_e] = {
                                "mean": float(_pred_dap),
                                "std": 0.0,
                                "median": float(_pred_dap),
                                "samples": [float(_pred_dap)],
                            }
                        if _new_pp:
                            r["posterior_predictive"] = _new_pp
                            n_dt_ok += 1
                    print(f"  Dual-threshold recomputed for {n_dt_ok} genotypes "
                          f"(skipped {n_dt_skip})")
                    print(f"  Binding channel: thermal={n_t_binding}, "
                          f"photo={n_p_binding} cells")
                _dual_threshold_applied = True

        # Apply population photo params uniformly — but only if there's a real improvement
        # over temp-only. Threshold: at least 0.5 day² improvement in SSE/n.
        if not _dual_threshold_applied and pop_photo.get("improvement", 0.0) > 0.5:
            photo_results = {}
            if photo_model == "linear_plateau":
                for g in fitted_alpha.keys():
                    photo_results[g] = {
                        "photo_n_min": pop_photo["photo_n_min"],
                        "photo_n_opt": pop_photo["photo_n_opt"],
                    }
                _pop_n_min = pop_photo["photo_n_min"]
                _pop_n_opt = pop_photo["photo_n_opt"]
                photo_kwargs = {
                    "photo_model": "linear_plateau",
                    "photo_n_min": pop_photo["photo_n_min"],
                    "photo_n_opt": pop_photo["photo_n_opt"],
                }
            else:  # logistic3
                for g in fitted_alpha.keys():
                    photo_results[g] = {
                        "photo_a": pop_photo["photo_a"],
                        "photo_b": pop_photo["photo_b"],
                        "photo_d": pop_photo["photo_d"],
                    }
                photo_kwargs = {
                    "photo_model": "logistic3",
                    "photo_a": pop_photo["photo_a"],
                    "photo_b": pop_photo["photo_b"],
                    "photo_d": pop_photo["photo_d"],
                }
            print(f"  Photoperiod signal detected (improvement={pop_photo['improvement']:.2f}), applying to predictions")

            # Re-compute posterior_predictive DAP for each genotype with photo applied.
            # Note: this only OVERWRITES if photo prediction succeeds. When F(N) is
            # ~0 for the entire prediction window (e.g. ND held-out from tropical
            # training), pred_dap == -1 and the temp-only Phase 2 prediction is
            # preserved. This is the implicit fallback that keeps Mech alive at
            # extreme latitudes. For logistic3 this should essentially never trigger
            # (smooth sigmoid never reaches 0), but the gate is preserved as a safety
            # net identical to the linear_plateau path.
            if predict_planting_names:
                print(f"  Recomputing posterior predictive DAP with photoperiod for {len(predict_planting_names)} prediction plantings: {predict_planting_names}")
                n_with_pred = sum(1 for g in T_dict if any(p in T_dict[g] for p in predict_planting_names))
                print(f"  Genotypes with at least one prediction planting in T_dict: {n_with_pred}/{len(T_dict)}")
                n_recomputed = 0
                n_skipped_error = 0
                n_skipped_finite = 0
                n_skipped_no_train = 0
                n_skipped_no_pred = 0
                for r in fit_results:
                    if r.get("error") is not None:
                        n_skipped_error += 1
                        continue
                    gid = r["id"]
                    a_hat = r.get("alpha")
                    b_hat = r.get("beta")
                    if not (a_hat is not None and b_hat is not None
                            and np.isfinite(a_hat) and np.isfinite(b_hat)):
                        n_skipped_finite += 1
                        continue
                    t_hat = r.get("topt", args.Topt)
                    # Compute target H from training plantings WITH photo
                    train_pl = list(plantings_of_g_fn(gid))
                    H_vals = []
                    for e in train_pl:
                        try:
                            temps_e = np.asarray(T_dict[gid][e], dtype=float)
                            dap_e = int(DAP_obs_dict[gid][e])
                            dl_e = DL_dict.get(gid, {}).get(e) if DL_dict else None
                            rates = daily_development_rate(
                                temps_e, alpha=a_hat, beta=b_hat,
                                temp_base=args.Tb, temp_optimal=t_hat, temp_critical=args.Tc,
                                daylengths=dl_e,
                                **photo_kwargs,
                            )
                            dap_cut = max(0, min(dap_e, len(rates)))
                            H_vals.append(float(np.sum(rates[:dap_cut])))
                        except Exception:
                            continue
                    if not H_vals:
                        n_skipped_no_train += 1
                        continue
                    target_H = float(np.mean(H_vals))
                    # Predict for each held-out planting WITH photo
                    new_pp = {}
                    for pred_e in predict_planting_names:
                        if gid not in T_dict or pred_e not in T_dict[gid]:
                            continue
                        temps_pred = np.asarray(T_dict[gid][pred_e], dtype=float)
                        dl_pred = DL_dict.get(gid, {}).get(pred_e) if DL_dict else None
                        try:
                            pred_dap = predict_flowering_dap(
                                temps=temps_pred, target_H=target_H,
                                alpha=a_hat, beta=b_hat,
                                temp_base=args.Tb, temp_optimal=t_hat, temp_critical=args.Tc,
                                daylengths=dl_pred,
                                extrapolate=True,
                                **photo_kwargs,
                            )
                            if pred_dap > 0:
                                new_pp[pred_e] = {
                                    "mean": float(pred_dap),
                                    "std": 0.0,
                                    "median": float(pred_dap),
                                    "samples": [float(pred_dap)],
                                }
                        except Exception:
                            continue
                    if new_pp:
                        r["posterior_predictive"] = new_pp
                        n_recomputed += 1
                    else:
                        n_skipped_no_pred += 1
                print(f"  Recomputed posterior predictive for {n_recomputed} genotypes")
                print(f"  Skipped: error={n_skipped_error}, non-finite={n_skipped_finite}, no_train={n_skipped_no_train}, no_pred={n_skipped_no_pred}")
        else:
            print(f"  No photoperiod signal (improvement={pop_photo.get('improvement', 0):.2f}), skipping — temp-only predictions")

    # Training residual MSE for ensemble variance calibration (Kennedy & O'Hagan 2001)
    _train_rmse = _compute_training_rmse(
        fit_results, T_dict, DAP_obs_dict, DL_dict, plantings_of_g_fn,
        photo_B=photo_B, photo_Pc=photo_Pc,
        args_Tb=args.Tb, args_Topt=args.Topt, args_Tc=args.Tc,
        photo_type=photo_type,
    )
    training_mse = _train_rmse ** 2
    print(f"Training residual MSE: {training_mse:.2f} days² (RMSE={_train_rmse:.2f})")

    # Honest out-of-sample mech MSE + bias via leave-one-genotype-out GBLUP.
    # See companion block earlier in the file for rationale.
    _evd_for_loo = args.evd_path or str(get_evd_path(crop=args.crop))
    _gblup_loo = _compute_gblup_loo_mech_mse(
        fit_results, _evd_for_loo, T_dict, DAP_obs_dict, DL_dict,
        plantings_of_g_fn, photo_B, photo_Pc,
        args.Tb, args.Topt, args.Tc,
        photo_type=photo_type, photo_model=photo_model,
        photo_results=photo_results if 'photo_results' in dir() else None,
    )
    if _gblup_loo is not None:
        gblup_loo_mse = _gblup_loo["mse"]
        gblup_loo_bias = _gblup_loo["bias"]
        print(f"GBLUP-LOO mech MSE: {gblup_loo_mse:.2f} days² "
              f"(vs in-sample training_mse={training_mse:.2f}), "
              f"bias={gblup_loo_bias:+.2f} days (n={_gblup_loo['n']})")
    else:
        print(f"GBLUP-LOO mech MSE: unavailable, using training_mse={training_mse:.2f}")
        gblup_loo_mse = training_mse
        gblup_loo_bias = 0.0

    # Env-holdout mech MSE — see companion block at ~line 2870.
    # FUTURE: add weather-distance inflation for OOD predict envs.
    _env_loo = _compute_env_holdout_mech_mse(
        fit_results, T_dict, DAP_obs_dict, DL_dict, plantings_of_g_fn,
        train_planting_names, photo_B, photo_Pc,
        args.Tb, args.Topt, args.Tc,
        photo_type=photo_type, photo_model=photo_model,
        photo_results=photo_results if 'photo_results' in dir() else None,
    )
    if _env_loo is not None:
        env_holdout_mech_mse = _env_loo["mse"]
        print(f"Env-holdout mech MSE: {env_holdout_mech_mse:.2f} days² "
              f"(env-extrapolation floor, n={_env_loo['n']})")
    else:
        env_holdout_mech_mse = 0.0
        print(f"Env-holdout mech MSE: unavailable (<2 training envs)")

    # LOO-planting variance per genotype (replaces global training_mse when --loo-variance)
    loo_vars = _compute_loo_vars(
        fit_results=fit_results, args=args,
        T_dict=T_dict, DAP_obs_dict=DAP_obs_dict, DL_dict=DL_dict,
        plantings_of_g_fn=plantings_of_g_fn,
        train_planting_names=train_planting_names,
        photo_B=photo_B, photo_Pc=photo_Pc, photo_type=photo_type,
        photo_model=photo_model, workers=args.workers,
    )

    # ── Phase 3: Build results and write CSVs ────────────────────────
    results = []
    for r in fit_results:
        row = {
            "id": r["id"], "alpha": r["alpha"], "beta": r["beta"],
            "loss": r.get("loss", ""),
            "alpha_mean": r.get("alpha_mean", ""),
            "alpha_std": r.get("alpha_std", ""),
            "beta_mean": r.get("beta_mean", ""),
            "beta_std": r.get("beta_std", ""),
            "n_accepted": r.get("n_accepted", ""),
            "n_training_plantings": r.get("n_training_plantings", ""),
        }
        if "topt" in r and not math.isnan(r.get("topt", float("nan"))):
            row["topt"] = r["topt"]
            row["topt_mean"] = r.get("topt_mean", "")
            row["topt_std"] = r.get("topt_std", "")
        results.append(row)

    # Write run configuration JSON for reproducibility
    import json as _json
    run_config = {
        "alpha_bounds": args.alpha_bounds,
        "beta_bounds": args.beta_bounds,
        "topt_bounds": getattr(args, "topt_bounds", None),
        "fit_topt": getattr(args, "fit_topt", False),
        "Tb": args.Tb, "Topt": args.Topt, "Tc": args.Tc,
        "iterative_rounds": 1,
        "convergence_tol": _convergence_tol,
        "top_k": args.top_k,
        "maxiter": args.maxiter,
        "n_restarts": args.n_restarts,
        "seed": args.seed,
        "workers": args.workers,
        "normalize_loss": args.normalize_loss,
        "loo_variance": args.loo_variance,
        "full_loo": getattr(args, "full_loo", False),
        "full_loo_maxiter": getattr(args, "full_loo_maxiter", None),
        "n_genotypes": len(results),
        "train_plantings": train_planting_names if train_planting_names else [],
        "predict_plantings": predict_planting_names if predict_planting_names else [],
    }
    run_config["lambda_max"] = lambda_max
    config_out = _output_path(args, "_run_config.json")
    with open(config_out, "w") as fh:
        _json.dump(run_config, fh, indent=2)
    print(f"Wrote run config to {config_out}")

    # write CSV (alpha/beta with posterior statistics)
    progress_out = _output_path(args, "_progress.csv")

    fieldnames = ["id", "alpha", "beta", "loss",
                  "alpha_mean", "alpha_std", "beta_mean", "beta_std",
                  "n_accepted", "n_training_plantings"]
    # Add topt columns if any result has them
    if results and "topt" in results[0]:
        fieldnames = ["id", "alpha", "beta", "topt", "loss",
                      "alpha_mean", "alpha_std", "beta_mean", "beta_std",
                      "topt_mean", "topt_std",
                      "n_accepted", "n_training_plantings"]
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # build and write per-planting cumulative progress
    progress_rows = []
    for row in results:
        gid = row["id"]
        a_hat = row["alpha"]
        b_hat = row["beta"]
        t_hat_prog = row.get("topt", args.Topt)
        # skip genotypes with NaN params
        if not (isinstance(a_hat, (int, float)) and np.isfinite(a_hat)
                and isinstance(b_hat, (int, float)) and np.isfinite(b_hat)):
            continue
        try:
            prog_map = cumulative_progress_by_planting(
                g=gid,
                alpha=a_hat,
                beta=b_hat,
                temp_base=args.Tb,
                temp_optimal=t_hat_prog,
                temp_critical=args.Tc,
                plantings_of_g_fn=plantings_of_g_fn,
                T_dict=T_dict,
                DAP_obs_dict=DAP_obs_dict,
                DL_dict=DL_dict,
                photo_B=photo_B,
                photo_Pc=photo_Pc,
                photo_type=photo_type,
            )
            if prog_map is None:
                continue
            if isinstance(prog_map, dict):
                items = prog_map.items()
            else:
                items = list(prog_map)
            for planting_id, prog in items:
                progress_rows.append({"id": gid, "planting": planting_id, "cumulative_progress": float(prog)})
        except Exception:
            continue

    if progress_rows:
        with open(progress_out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["id", "planting", "cumulative_progress"])
            writer.writeheader()
            for prow in progress_rows:
                writer.writerow(prow)

    # If debug requested, write per-day rates (date,temp,rate) for each genotype/planting
    # ── Raw Mechanistic (zero penalty, single MAP) ─────────────────────
    # Pure process model: cardinal-beta thermal time + photoperiod, no GBLUP
    # coupling. Written to raw_mechanistic.csv for comparison plots.
    _raw_mech_results = None
    if predict_planting_names:
        print(f"\n  Fitting raw mechanistic (no GBLUP penalty)...")
        _raw_tasks = _build_topk_tasks(
            genotypes=genotypes,
            T_dict=T_dict, DAP_obs_dict=DAP_obs_dict, DL_dict=DL_dict,
            plantings_of_g_fn=plantings_of_g_fn,
            args_Tb=args.Tb, args_Topt=args.Topt, args_Tc=args.Tc,
            alpha_bounds=args.alpha_bounds, beta_bounds=args.beta_bounds,
            maxiter=args.maxiter, local_max_iter=args.local_max_iter,
            seed=args.seed, n_restarts=1, normalize_loss=args.normalize_loss,
            top_k=1, max_posterior_samples=0,
            predict_plantings=None,
            photo_B=photo_B, photo_Pc=photo_Pc, photo_type=photo_type,
            photo_model=photo_model,
            gblup_targets=None, lambda_a=0.0, lambda_b=0.0,
            lambda_scale=None,
            topt_bounds=getattr(args, '_topt_bounds', None),
            lambda_t=0.0,
            obs_perturbation_sd=0.0, perturb_maxiter=0,
            photo_B_bounds=getattr(args, 'photo_B_bounds', None),
            photo_Pc_bounds=getattr(args, 'photo_Pc_bounds', None),
            gblup_photo_targets=None, lambda_bphoto=0.0, lambda_pc=0.0,
            photo_a_bounds=getattr(args, 'photo_a_bounds', None),
            photo_b_bounds=getattr(args, 'photo_b_bounds', None),
            photo_d_bounds=getattr(args, 'photo_d_bounds', None),
            gblup_photo3_targets=None,
            lambda_photo_a=0.0, lambda_photo_b=0.0, lambda_photo_d=0.0,
            photo_n_min=None, photo_n_opt=None,
        )
        _raw_mech_results = _run_fitting(_raw_tasks, args.workers,
                                          worker_fn=_topk_one_genotype, verbose=False)

    # Helper: get daylength array for a genotype/planting pair
    def _get_dl(gid, planting):
        if DL_dict is not None and gid in DL_dict and planting in DL_dict[gid]:
            return DL_dict[gid][planting]
        return None

    # ── Prediction on held-out plantings ──────────────────────────────────
    if predict_planting_names:
        predict_out = _output_path(args, "_predictions.csv")
        pred_rows = []

        # Use posterior predictive DAP from ABC results
        for r in fit_results:
            gid = r["id"]
            if r["error"] is not None:
                continue
            a_hat = r["alpha"]
            b_hat = r["beta"]
            t_hat = r.get("topt", args.Topt)  # per-genotype Topt if fitted
            if not (np.isfinite(a_hat) and np.isfinite(b_hat)):
                continue

            pp = r.get("posterior_predictive", {})

            for pred_e in predict_planting_names:
                pp_e = pp.get(pred_e)
                if pp_e is not None:
                    predicted_dap = round(pp_e["mean"])
                    mech_std = pp_e["std"]
                else:
                    # Fallback: point estimate using MAP alpha/beta
                    train_plantings = list(plantings_of_g_fn(gid))
                    if len(train_plantings) == 0:
                        continue
                    # Per-genotype photo params from sequential fitting
                    _pr = photo_results.get(gid) if photo_results else None
                    _p_a = _pr.get("photo_a") if _pr else None
                    _p_b = _pr.get("photo_b") if _pr else None
                    _p_d = _pr.get("photo_d") if _pr else None
                    _p_nmin = _pr.get("photo_n_min") if _pr else None
                    _p_nopt = _pr.get("photo_n_opt") if _pr else None
                    H_vals = []
                    for e in train_plantings:
                        try:
                            temps_e = np.asarray(T_dict[gid][e], dtype=float)
                            dap_e = int(DAP_obs_dict[gid][e])
                            rates = daily_development_rate(
                                temps_e, alpha=a_hat, beta=b_hat,
                                temp_base=args.Tb, temp_optimal=t_hat, temp_critical=args.Tc,
                                daylengths=_get_dl(gid, e), photo_B=photo_B, photo_Pc=photo_Pc,
                                photo_type=photo_type, photo_model=photo_model,
                                photo_a=_p_a, photo_b=_p_b, photo_d=_p_d,
                                photo_n_min=_p_nmin, photo_n_opt=_p_nopt,
                            )
                            dap_cut = max(0, min(dap_e, len(rates)))
                            H_vals.append(float(np.sum(rates[:dap_cut])))
                        except Exception:
                            continue
                    if len(H_vals) == 0:
                        continue
                    target_H = float(np.mean(H_vals))
                    if gid not in T_dict or pred_e not in T_dict[gid]:
                        continue
                    temps_pred = np.asarray(T_dict[gid][pred_e], dtype=float)
                    predicted_dap = predict_flowering_dap(
                        temps=temps_pred, target_H=target_H,
                        alpha=a_hat, beta=b_hat,
                        temp_base=args.Tb, temp_optimal=t_hat, temp_critical=args.Tc,
                        daylengths=_get_dl(gid, pred_e), photo_B=photo_B, photo_Pc=photo_Pc,
                        photo_model=photo_model,
                        extrapolate=True,
                        photo_a=_p_a, photo_b=_p_b, photo_d=_p_d,
                        photo_n_min=_p_nmin, photo_n_opt=_p_nopt,
                    )
                    mech_std = r.get("alpha_std", 0.0) * 10  # rough fallback uncertainty

                # Observed DAP: prefer test_obs_by_cell (held-out cells from
                # --test-phenotypes) when present, otherwise fall back to
                # DAP_obs_dict built from the training phenotypes file.
                if test_obs_by_cell is not None:
                    observed_dap = test_obs_by_cell.get((str(gid).upper().strip(), str(pred_e).strip()))
                    if observed_dap is None:
                        # Cell is not in the held-out set -> skip emitting a
                        # validation row for it. (It was in training, so
                        # reporting metrics on it would be leakage.)
                        continue
                    observed_dap = int(round(observed_dap))
                else:
                    observed_dap = DAP_obs_dict.get(gid, {}).get(pred_e, None)
                error = (predicted_dap - observed_dap) if (observed_dap is not None and predicted_dap > 0) else None

                pred_row = {
                    "id": gid,
                    "planting": pred_e,
                    "observed_dap": observed_dap if observed_dap is not None else "NA",
                    "predicted_dap": predicted_dap if predicted_dap > 0 else "NA",
                    "error": f"{error:.1f}" if error is not None else "NA",
                    "mech_dap_mean": f"{pp_e['mean']:.1f}" if pp_e else "NA",
                    "mech_dap_std": f"{mech_std:.2f}" if mech_std is not None else "NA",
                    "alpha": f"{a_hat:.6f}",
                    "beta": f"{b_hat:.6f}",
                }
                if "topt" in r and not math.isnan(r.get("topt", float("nan"))):
                    pred_row["topt"] = f"{t_hat:.4f}"
                pred_rows.append(pred_row)

        # ── Raw mechanistic predictions (population-mean α, β) ──────────────
        # Single (α̅, β̅) derived as the mean across successful per-genotype
        # training fits; applied uniformly to every genotype. Target thermal
        # time is the mean accumulated development across training cells under
        # (α̅, β̅). Population-level photoperiod settings (photo_B, photo_Pc)
        # are retained; per-genotype photo_a/b/d are intentionally dropped so
        # that predictions do not distinguish genotypes within an environment.
        # When --test-phenotypes is provided, predictions are emitted only for
        # held-out cells (matching the CV test set used by the other methods).
        _raw_pred_rows = []
        if _raw_mech_results and predict_planting_names:
            _raw_alphas, _raw_betas, _raw_topts = [], [], []
            for r in _raw_mech_results:
                if r.get("error") is not None:
                    continue
                a = r.get("alpha"); b = r.get("beta")
                if a is None or b is None:
                    continue
                if not (np.isfinite(a) and np.isfinite(b)):
                    continue
                _raw_alphas.append(float(a))
                _raw_betas.append(float(b))
                _t = r.get("topt", args.Topt)
                if _t is not None and np.isfinite(_t):
                    _raw_topts.append(float(_t))

            if _raw_alphas and _raw_betas:
                alpha_bar = float(np.mean(_raw_alphas))
                beta_bar = float(np.mean(_raw_betas))
                topt_bar = float(np.mean(_raw_topts)) if _raw_topts else float(args.Topt)

                # Population target_H: mean accumulated development at
                # observed DAP across all training (gid, planting) cells
                # simulated with (α̅, β̅).
                _H_vals = []
                for _gid in genotypes:
                    for _e in plantings_of_g_fn(_gid):
                        if _gid not in T_dict or _e not in T_dict[_gid]:
                            continue
                        _obs = DAP_obs_dict.get(_gid, {}).get(_e)
                        if _obs is None:
                            continue
                        try:
                            _temps = np.asarray(T_dict[_gid][_e], dtype=float)
                            _rates = daily_development_rate(
                                _temps, alpha=alpha_bar, beta=beta_bar,
                                temp_base=args.Tb, temp_optimal=topt_bar,
                                temp_critical=args.Tc,
                                daylengths=_get_dl(_gid, _e),
                                photo_B=photo_B, photo_Pc=photo_Pc,
                                photo_type=photo_type, photo_model=photo_model,
                                photo_a=None, photo_b=None, photo_d=None,
                                photo_n_min=None, photo_n_opt=None,
                            )
                            _cut = max(0, min(int(_obs), len(_rates)))
                            if _cut > 0:
                                _H_vals.append(float(np.sum(_rates[:_cut])))
                        except Exception:
                            continue

                if _H_vals:
                    target_H_bar = float(np.mean(_H_vals))

                    # Iterate over training genotypes × predict plantings.
                    # When --test-phenotypes is present, keep only cells that
                    # appear in the held-out test set (same test fold as the
                    # other methods). Otherwise fall back to training cells.
                    for gid in genotypes:
                        for pred_e in predict_planting_names:
                            if gid not in T_dict or pred_e not in T_dict[gid]:
                                continue
                            try:
                                temps_pred = np.asarray(T_dict[gid][pred_e], dtype=float)
                                predicted_dap = predict_flowering_dap(
                                    temps=temps_pred, target_H=target_H_bar,
                                    alpha=alpha_bar, beta=beta_bar,
                                    temp_base=args.Tb, temp_optimal=topt_bar,
                                    temp_critical=args.Tc,
                                    daylengths=_get_dl(gid, pred_e),
                                    photo_B=photo_B, photo_Pc=photo_Pc,
                                    photo_model=photo_model, extrapolate=True,
                                    photo_a=None, photo_b=None, photo_d=None,
                                    photo_n_min=None, photo_n_opt=None,
                                )
                            except Exception:
                                continue

                            if test_obs_by_cell is not None:
                                _obs_val = test_obs_by_cell.get(
                                    (str(gid).upper().strip(), str(pred_e).strip()))
                                if _obs_val is None:
                                    # Not a held-out test cell → skip
                                    # (prevents reporting metrics on training obs).
                                    continue
                                observed_dap = int(round(float(_obs_val)))
                            else:
                                _obs_val = DAP_obs_dict.get(gid, {}).get(pred_e)
                                if _obs_val is None:
                                    continue
                                observed_dap = int(_obs_val)

                            error = (predicted_dap - observed_dap) if predicted_dap > 0 else None
                            _raw_pred_rows.append({
                                "id": gid, "planting": pred_e,
                                "observed_dap": observed_dap,
                                "predicted_dap": predicted_dap if predicted_dap > 0 else "NA",
                                "error": f"{error:.1f}" if error is not None else "NA",
                            })

            if _raw_pred_rows:
                _raw_out = _output_path(args, "_raw_mech_predictions.csv")
                _raw_fields = ["id", "planting", "observed_dap", "predicted_dap", "error"]
                with open(_raw_out, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=_raw_fields)
                    w.writeheader()
                    w.writerows(_raw_pred_rows)
                _raw_errors = [float(r["error"]) for r in _raw_pred_rows if r["error"] != "NA"]
                if _raw_errors:
                    _raw_mae = float(np.mean(np.abs(_raw_errors)))
                    print(f"    Raw Mech (pop-mean α, β)  MAE = {_raw_mae:.2f} "
                          f"(n={len(_raw_errors)})")

        # ── Write posterior predictive DAP per genotype/planting ──
        if fit_results and any(r.get("posterior_predictive") for r in fit_results if r["error"] is None):
            pp_out = _output_path(args, "_posterior_predictions.csv")
            pp_rows = []
            for r in fit_results:
                if r["error"] is not None:
                    continue
                gid = r["id"]
                pp = r.get("posterior_predictive", {})
                for pred_e, pp_e in pp.items():
                    pp_rows.append({
                        "id": gid,
                        "planting": pred_e,
                        "mech_dap_mean": f"{pp_e['mean']:.2f}",
                        "mech_dap_std": f"{pp_e['std']:.2f}",
                        "mech_dap_median": f"{pp_e['median']:.1f}",
                        "n_samples": len(pp_e.get("samples", [])),
                    })
            if pp_rows:
                with open(pp_out, "w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=[
                        "id", "planting", "mech_dap_mean", "mech_dap_std",
                        "mech_dap_median", "n_samples",
                    ])
                    writer.writeheader()
                    for prow in pp_rows:
                        writer.writerow(prow)
                print(f"Wrote posterior predictive DAPs to {pp_out}")

        # ── GBLUP-predicted (alpha, beta) for unseen genotypes ──────────────
        # GBLUP on fitted shape parameters, then thermal time integration for DAP.
        # Works even when training/prediction genotype panels are fully disjoint
        # (e.g., cross-RM soybean) because alpha/beta are genotype-specific traits.
        if not args.no_gblup_dap:
            try:
                _evd_path = args.evd_path or str(get_evd_path(crop=args.crop))
                if Path(_evd_path).exists():
                    from cgm_wgp.gblup import gblup_predict_params

                    # Build fitted params dict and population H*
                    fitted_params = {}
                    fitted_gids_set = set()
                    H_star_vals = []
                    _topt_map = {}  # per-genotype fitted Topt
                    for r in fit_results:
                        if r["error"] is not None:
                            continue
                        gid = r["id"]
                        a, b = r["alpha"], r["beta"]
                        t_g = r.get("topt", args.Topt)
                        if not (np.isfinite(a) and np.isfinite(b)):
                            continue
                        # Skip sentinel fits (no real training data)
                        loss = r.get("loss", 1e9)
                        if not np.isfinite(loss) or loss >= 1e8:
                            continue
                        fitted_params[gid] = (a, b)
                        fitted_gids_set.add(gid)
                        _topt_map[gid] = t_g
                        # Compute H* for this genotype
                        _pr_g = photo_results.get(gid) if photo_results else None
                        _pa_g = _pr_g.get("photo_a") if _pr_g else None
                        _pb_g = _pr_g.get("photo_b") if _pr_g else None
                        _pd_g = _pr_g.get("photo_d") if _pr_g else None
                        _pnmin_g = _pr_g.get("photo_n_min") if _pr_g else None
                        _pnopt_g = _pr_g.get("photo_n_opt") if _pr_g else None
                        train_pl = list(plantings_of_g_fn(gid))
                        H_vals = []
                        for e in train_pl:
                            if gid not in T_dict or e not in T_dict[gid]:
                                continue
                            temps_e = np.asarray(T_dict[gid][e], dtype=float)
                            dap_e = int(DAP_obs_dict[gid][e])
                            dl_e = None
                            if DL_dict and gid in DL_dict and e in DL_dict[gid]:
                                dl_e = DL_dict[gid][e]
                            rates = daily_development_rate(
                                temps_e, alpha=a, beta=b,
                                temp_base=args.Tb, temp_optimal=t_g, temp_critical=args.Tc,
                                daylengths=dl_e, photo_B=photo_B, photo_Pc=photo_Pc,
                                photo_type=photo_type, photo_model=photo_model,
                                photo_a=_pa_g, photo_b=_pb_g, photo_d=_pd_g,
                                photo_n_min=_pnmin_g, photo_n_opt=_pnopt_g,
                            )
                            dap_cut = max(0, min(dap_e, len(rates)))
                            H_vals.append(float(np.sum(rates[:dap_cut])))
                        if H_vals:
                            H_star_vals.append(float(np.mean(H_vals)))

                    if len(fitted_params) >= 2 and H_star_vals:
                        pop_H_star = float(np.mean(H_star_vals))
                        pop_topt = float(np.mean(list(_topt_map.values()))) if _topt_map else args.Topt

                        # GBLUP-predict (alpha, beta) for all G-matrix genotypes
                        param_preds = gblup_predict_params(_evd_path, fitted_params)

                        # Population-average photo params for proxy genotypes
                        _pop_pa, _pop_pb, _pop_pd = None, None, None
                        _pop_pnmin, _pop_pnopt = None, None
                        if photo_results:
                            _pa_list = [v.get("photo_a") for v in photo_results.values() if v.get("photo_a") is not None and np.isfinite(v.get("photo_a"))]
                            _pb_list = [v.get("photo_b") for v in photo_results.values() if v.get("photo_b") is not None and np.isfinite(v.get("photo_b"))]
                            _pd_list = [v.get("photo_d") for v in photo_results.values() if v.get("photo_d") is not None and np.isfinite(v.get("photo_d"))]
                            if _pa_list:
                                _pop_pa = float(np.mean(_pa_list))
                                _pop_pb = float(np.mean(_pb_list))
                                _pop_pd = float(np.mean(_pd_list))
                            # Linear-plateau params (same value across all genotypes when population-fit)
                            _pnmin_list = [v.get("photo_n_min") for v in photo_results.values() if v.get("photo_n_min") is not None]
                            _pnopt_list = [v.get("photo_n_opt") for v in photo_results.values() if v.get("photo_n_opt") is not None]
                            if _pnmin_list:
                                _pop_pnmin = float(np.mean(_pnmin_list))
                                _pop_pnopt = float(np.mean(_pnopt_list))

                        n_proxy = 0
                        for proxy_gid, (pred_a, pred_b, pev_a, pev_b) in param_preds.items():
                            if proxy_gid in fitted_gids_set:
                                continue  # already have direct posterior predictive
                            # Use per-genotype photo if available, else population average
                            _pr_proxy = photo_results.get(proxy_gid) if photo_results else None
                            _pxy_a = _pr_proxy.get("photo_a") if _pr_proxy else _pop_pa
                            _pxy_b = _pr_proxy.get("photo_b") if _pr_proxy else _pop_pb
                            _pxy_d = _pr_proxy.get("photo_d") if _pr_proxy else _pop_pd
                            _pxy_nmin = _pr_proxy.get("photo_n_min") if _pr_proxy else _pop_pnmin
                            _pxy_nopt = _pr_proxy.get("photo_n_opt") if _pr_proxy else _pop_pnopt

                            for pred_e in predict_planting_names:
                                temps_e = ref_temps.get(pred_e)
                                if temps_e is None:
                                    continue
                                dl_e = ref_dl.get(pred_e)
                                proxy_dap = predict_flowering_dap(
                                    temps=np.asarray(temps_e, dtype=float),
                                    target_H=pop_H_star,
                                    alpha=pred_a, beta=pred_b,
                                    temp_base=args.Tb, temp_optimal=pop_topt,
                                    temp_critical=args.Tc,
                                    daylengths=dl_e, photo_B=photo_B, photo_Pc=photo_Pc,
                                    photo_type=photo_type, photo_model=photo_model,
                                    extrapolate=True,
                                    photo_a=_pxy_a, photo_b=_pxy_b, photo_d=_pxy_d,
                                    photo_n_min=_pxy_nmin, photo_n_opt=_pxy_nopt,
                                )
                                # Observed DAP: use test cells when provided
                                if test_obs_by_cell is not None:
                                    observed_dap = test_obs_by_cell.get(
                                        (str(proxy_gid).upper().strip(), str(pred_e).strip())
                                    )
                                    if observed_dap is not None:
                                        observed_dap = int(round(observed_dap))
                                    # Don't skip when obs is missing — emit
                                    # prediction with NA obs so ensemble can use it
                                else:
                                    observed_dap = DAP_obs_dict.get(proxy_gid, {}).get(pred_e)
                                error = (proxy_dap - observed_dap) if (observed_dap is not None and proxy_dap > 0) else None
                                # Uncertainty: use max of GBLUP-LOO (unseen-
                                # genotype error) and env-holdout (unseen-env
                                # error). Both are honest out-of-sample
                                # estimates; the max handles both new_varieties
                                # (gid-holdout dominant) and cv00_double_novelty
                                # (env-holdout dominant).
                                proxy_std = float(np.sqrt(max(
                                    gblup_loo_mse,
                                    env_holdout_mech_mse,
                                )))
                                pred_rows.append({
                                    "id": proxy_gid, "planting": pred_e,
                                    "observed_dap": observed_dap if observed_dap is not None else "NA",
                                    "predicted_dap": proxy_dap if proxy_dap > 0 else "NA",
                                    "error": f"{error:.1f}" if error is not None else "NA",
                                    "mech_dap_mean": f"{proxy_dap:.1f}" if proxy_dap > 0 else "NA",
                                    "mech_dap_std": f"{proxy_std:.2f}",
                                    "alpha": f"{pred_a:.6f}", "beta": f"{pred_b:.6f}",
                                })
                                n_proxy += 1
                        if n_proxy > 0:
                            print(f"  GBLUP-predicted params: added {n_proxy} mechanistic predictions "
                                  f"for unseen genotypes (pop_H*={pop_H_star:.4f})")
            except Exception as exc:
                print(f"  Warning: GBLUP param prediction failed: {exc}", file=sys.stderr)

        # ── LOO-env training back-prediction for ensemble weighting ────────
        mech_train_errors = []
        mech_bias_per_gid = {}  # gid -> list of signed LOO residuals
        for row in results:
            gid = row["id"]
            a_hat = row["alpha"]
            b_hat = row["beta"]
            t_hat_loo = row.get("topt", args.Topt) if isinstance(row, dict) else args.Topt
            if not (isinstance(a_hat, (int, float)) and np.isfinite(a_hat)
                    and isinstance(b_hat, (int, float)) and np.isfinite(b_hat)):
                continue
            # Per-genotype photo params from sequential fitting
            _pr_loo = photo_results.get(gid) if photo_results else None
            _pa_loo = _pr_loo.get("photo_a") if _pr_loo else None
            _pb_loo = _pr_loo.get("photo_b") if _pr_loo else None
            _pd_loo = _pr_loo.get("photo_d") if _pr_loo else None
            _pnmin_loo = _pr_loo.get("photo_n_min") if _pr_loo else None
            _pnopt_loo = _pr_loo.get("photo_n_opt") if _pr_loo else None
            train_pl = list(plantings_of_g_fn(gid))
            if len(train_pl) < 3:
                continue  # need >=3 so LOO leaves >=2 for target_H
            H_by_env = {}
            for e in train_pl:
                try:
                    temps_e = np.asarray(T_dict[gid][e], dtype=float)
                    dap_e = int(DAP_obs_dict[gid][e])
                    rates = daily_development_rate(
                        temps_e, alpha=a_hat, beta=b_hat,
                        temp_base=args.Tb, temp_optimal=t_hat_loo, temp_critical=args.Tc,
                        daylengths=_get_dl(gid, e), photo_B=photo_B, photo_Pc=photo_Pc,
                        photo_type=photo_type, photo_model=photo_model,
                        photo_a=_pa_loo, photo_b=_pb_loo, photo_d=_pd_loo,
                        photo_n_min=_pnmin_loo, photo_n_opt=_pnopt_loo,
                    )
                    dap_cut = max(0, min(dap_e, len(rates)))
                    H_by_env[e] = float(np.sum(rates[:dap_cut]))
                except Exception:
                    continue
            if len(H_by_env) < 3:
                continue
            env_list = list(H_by_env.keys())
            for held_out in env_list:
                loo_H = float(np.mean([H_by_env[e] for e in env_list if e != held_out]))
                if loo_H <= 0:
                    continue
                try:
                    temps_ho = np.asarray(T_dict[gid][held_out], dtype=float)
                    predicted_dap = predict_flowering_dap(
                        temps=temps_ho, target_H=loo_H,
                        alpha=a_hat, beta=b_hat,
                        temp_base=args.Tb, temp_optimal=t_hat_loo, temp_critical=args.Tc,
                        daylengths=_get_dl(gid, held_out), photo_B=photo_B, photo_Pc=photo_Pc,
                        photo_type=photo_type, photo_model=photo_model,
                        photo_a=_pa_loo, photo_b=_pb_loo, photo_d=_pd_loo,
                        photo_n_min=_pnmin_loo, photo_n_opt=_pnopt_loo,
                    )
                    observed_dap = DAP_obs_dict.get(gid, {}).get(held_out)
                    if observed_dap is not None and predicted_dap > 0:
                        signed_err = float(predicted_dap - int(observed_dap))
                        mech_train_errors.append(signed_err)
                        mech_bias_per_gid.setdefault(gid, []).append(signed_err)
                except Exception:
                    continue

        # Collapse per-genotype error lists to mean signed bias
        mech_bias_per_gid = {g: float(np.mean(v)) for g, v in mech_bias_per_gid.items() if v}
        _pop_mech_bias = float(np.mean(list(mech_bias_per_gid.values()))) if mech_bias_per_gid else 0.0
        # Unseen genotypes (proxy predictions) have no direct LOO bias —
        # use the GBLUP-LOO global bias from the new helper when available,
        # else fall back to the training-gid mean bias.
        _unseen_bias = gblup_loo_bias if 'gblup_loo_bias' in dir() and gblup_loo_bias else _pop_mech_bias
        if mech_bias_per_gid:
            _biases = np.array(list(mech_bias_per_gid.values()))
            print(f"  Per-genotype mech bias: mean={_pop_mech_bias:+.2f}, "
                  f"median={float(np.median(_biases)):+.2f}, "
                  f"std={float(np.std(_biases)):.2f} "
                  f"(n_genotypes={len(_biases)}); "
                  f"unseen-gid bias={_unseen_bias:+.2f}")

        # Apply bias correction in-place to pred_rows. Use the POPULATION
        # mean bias only — per-genotype bias estimates from LOO on ~5
        # plantings are too noisy to subtract safely (adding noise
        # instead of removing systematic error). Population bias is a
        # single scalar that corrects the crop-wide systematic offset
        # without injecting per-genotype noise.
        _applied_bias = _pop_mech_bias  # scalar, applied uniformly
        _n_corrected = 0
        if abs(_applied_bias) > 1e-9:
            for _r in pred_rows:
                if _r["predicted_dap"] == "NA":
                    continue
                _old_dap = float(_r["predicted_dap"])
                _new_dap = round(_old_dap - _applied_bias)
                _r["predicted_dap"] = int(_new_dap)
                # Recompute error if obs is available
                _obs = _r.get("observed_dap")
                if _obs not in (None, "NA"):
                    _r["error"] = f"{(_new_dap - float(_obs)):.1f}"
                # Also update mech_dap_mean for posterior writeback
                if _r.get("mech_dap_mean") not in ("NA", None):
                    try:
                        _old_mean = float(_r["mech_dap_mean"])
                        _r["mech_dap_mean"] = f"{(_old_mean - _applied_bias):.1f}"
                    except (TypeError, ValueError):
                        pass
                _n_corrected += 1
            print(f"  Bias-corrected {_n_corrected} mechanistic prediction rows "
                  f"(population bias={_applied_bias:+.2f})")

        if pred_rows:
            _pred_fields = ["id", "planting", "observed_dap", "predicted_dap", "error",
                            "mech_dap_mean", "mech_dap_std", "alpha", "beta"]
            if any("topt" in r for r in pred_rows):
                _pred_fields.append("topt")
            with open(predict_out, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=_pred_fields)
                writer.writeheader()
                for prow in pred_rows:
                    writer.writerow(prow)

            # Summary statistics
            errors = [float(r["error"]) for r in pred_rows if r["error"] != "NA"]
            n_failed = sum(1 for r in pred_rows if r["predicted_dap"] == "NA")
            n_total = len(pred_rows)
            if errors:
                abs_errors = np.abs(errors)
                mae = float(np.mean(abs_errors))
                rmse = float(np.sqrt(np.mean(np.square(errors))))
                bias = float(np.mean(errors))
                median_err = float(np.median(errors))

                print(f"\n{'=' * 60}")
                print(f"  MECHANISTIC (ABC) PREDICTION SUMMARY")
                print(f"{'=' * 60}")
                print(f"  Output: {predict_out}")
                print(f"  Total predictions: {n_total}  |  Evaluated: {len(errors)}  |  Failed (no convergence): {n_failed}")
                print()

                print(f"  Overall Metrics:")
                print(f"    MAE  = {mae:.1f} days    (mean absolute error)")
                print(f"    RMSE = {rmse:.1f} days    (root mean square error)")
                print(f"    Bias = {bias:+.1f} days    ({'predictions tend early' if bias < 0 else 'predictions tend late' if bias > 0 else 'no systematic direction'})")
                print(f"    Median error = {median_err:+.1f} days")
                print()

                print(f"  Accuracy Breakdown:")
                for threshold in [3, 5, 7, 10, 15, 20]:
                    count = int(np.sum(abs_errors <= threshold))
                    pct = count / len(errors) * 100
                    bar = '#' * int(pct / 2)
                    print(f"    Within {threshold:2d} days: {count:4d}/{len(errors)}  ({pct:5.1f}%)  {bar}")
                print()

                # Per-planting breakdown
                plantings_in_pred = sorted(set(r["planting"] for r in pred_rows))
                if len(plantings_in_pred) > 1:
                    print(f"  Per-Planting Breakdown:")
                    print(f"    {'Planting':<14s} {'N':>4s}  {'MAE':>6s}  {'RMSE':>6s}  {'Bias':>7s}  {'Within5':>8s}")
                    print(f"    {'-'*14} {'-'*4}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*8}")
                    for p in plantings_in_pred:
                        p_errors = [float(r["error"]) for r in pred_rows if r["planting"] == p and r["error"] != "NA"]
                        if p_errors:
                            p_abs = np.abs(p_errors)
                            p_mae = float(np.mean(p_abs))
                            p_rmse = float(np.sqrt(np.mean(np.square(p_errors))))
                            p_bias = float(np.mean(p_errors))
                            p_w5 = np.sum(p_abs <= 5) / len(p_errors) * 100
                            print(f"    {p:<14s} {len(p_errors):4d}  {p_mae:5.1f}d  {p_rmse:5.1f}d  {p_bias:+6.1f}d  {p_w5:6.1f}%")
                    print()

                print(f"  How these metrics work:")
                print(f"    error    = predicted_dap - observed_dap  (per genotype per planting)")
                print(f"    MAE      = mean( |error| )              (average miss size, ignoring direction)")
                print(f"    RMSE     = sqrt( mean( error^2 ) )      (like MAE but large errors count more)")
                print(f"    Bias     = mean( error )                 (+ means late, - means early on average)")
                print(f"{'=' * 60}")

                # Write metrics_summary.csv
                metrics_out = _output_path(args, "_metrics_summary.csv")
                with open(metrics_out, "w", newline="") as mf:
                    mw = csv.writer(mf)
                    mw.writerow(["planting", "n_evaluated", "n_failed",
                                 "mae", "rmse", "bias", "median_error",
                                 "within_3d", "within_5d", "within_7d",
                                 "within_10d", "within_15d", "within_20d"])
                    mw.writerow([
                        "ALL", len(errors), n_failed,
                        round(mae, 2), round(rmse, 2), round(bias, 2), round(median_err, 2),
                        int(np.sum(abs_errors <= 3)), int(np.sum(abs_errors <= 5)),
                        int(np.sum(abs_errors <= 7)), int(np.sum(abs_errors <= 10)),
                        int(np.sum(abs_errors <= 15)), int(np.sum(abs_errors <= 20)),
                    ])
                    for p in sorted(set(r["planting"] for r in pred_rows)):
                        p_errors = np.array([float(r["error"]) for r in pred_rows
                                             if r["planting"] == p and r["error"] != "NA"])
                        p_failed = sum(1 for r in pred_rows
                                       if r["planting"] == p and r["predicted_dap"] == "NA")
                        if len(p_errors) > 0:
                            p_abs = np.abs(p_errors)
                            mw.writerow([
                                p, len(p_errors), p_failed,
                                round(float(np.mean(p_abs)), 2),
                                round(float(np.sqrt(np.mean(p_errors**2))), 2),
                                round(float(np.mean(p_errors)), 2),
                                round(float(np.median(p_errors)), 2),
                                int(np.sum(p_abs <= 3)), int(np.sum(p_abs <= 5)),
                                int(np.sum(p_abs <= 7)), int(np.sum(p_abs <= 10)),
                                int(np.sum(p_abs <= 15)), int(np.sum(p_abs <= 20)),
                            ])
                print(f"Wrote metrics summary to {metrics_out}")
            else:
                print(f"\nPredictions written to: {predict_out} (no observed values for comparison)")

        # ── Diagnostic: mech per-env predicted mean (for stdout print only) ──
        # NOT passed to GBLUP — that would make GBLUP depend on mech.
        mech_planting_means = {}
        for pred_e in predict_planting_names:
            daps = [int(r["predicted_dap"]) for r in pred_rows
                    if r["planting"] == pred_e and r["predicted_dap"] != "NA"]
            if daps:
                mech_planting_means[pred_e] = float(np.mean(daps))

        # ── GBLUP DAP prediction (independent genomic baseline) ──
        # Must NOT use mechanistic predictions as env means — GBLUP is the
        # baseline we compare mech/ensemble against. Pass pred_planting_means=None
        # so gblup_predict_dap() computes per-env means from training data
        # only (falling back to overall training mean when the predict env
        # isn't in training, e.g. cv00_double_novelty / geographic LOO).
        # This keeps GBLUP's performance fully independent of mech.
        gblup_dap_rows = []
        gblup_info = {}
        evd_path = args.evd_path or str(get_evd_path(crop=args.crop))
        if not args.no_gblup_dap:
            if Path(evd_path).exists():
                try:
                    from cgm_wgp.gblup import gblup_predict_dap

                    gblup_dap_rows, gblup_info = gblup_predict_dap(
                        evd_path=evd_path,
                        pheno_df=pheno_obs,
                        train_plantings=train_planting_names if train_planting_names else list(set(pheno_obs["Planting"])),
                        predict_plantings=predict_planting_names,
                        rscript_path=None,
                        pred_planting_means=None,  # independent: use train envs
                        test_pheno_df=test_pheno_df,
                    )

                    if gblup_dap_rows:
                        gblup_dap_out = _output_path(args, "_gblup_dap_predictions.csv")
                        with open(gblup_dap_out, "w", newline="") as fh:
                            writer = csv.DictWriter(fh, fieldnames=[
                                "id", "planting", "observed_dap", "predicted_dap", "error",
                                "breeding_value", "pred_var",
                            ])
                            writer.writeheader()
                            for prow in gblup_dap_rows:
                                writer.writerow(prow)

                        # GBLUP DAP metrics summary
                        gblup_errors = [float(r["error"]) for r in gblup_dap_rows if r["error"] != "NA"]
                        gblup_n_failed = sum(1 for r in gblup_dap_rows if r["predicted_dap"] == "NA")
                        if gblup_errors:
                            gblup_abs_errors = np.abs(gblup_errors)
                            gblup_mae = float(np.mean(gblup_abs_errors))
                            gblup_rmse = float(np.sqrt(np.mean(np.square(gblup_errors))))
                            gblup_bias = float(np.mean(gblup_errors))
                            gblup_median_err = float(np.median(gblup_errors))

                            print(f"\n{'=' * 60}")
                            print(f"  GBLUP DAP PREDICTION SUMMARY")
                            print(f"{'=' * 60}")
                            print(f"  Output: {gblup_dap_out}")
                            print(f"  Lambda (REML): {gblup_info['lambda_est']:.4f}")
                            print(f"  G-matrix genotypes: {gblup_info['n_gmatrix']}  |  Matched to phenotypes: {gblup_info['n_matched']}")
                            print(f"  Total predictions: {len(gblup_dap_rows)}  |  Evaluated: {len(gblup_errors)}  |  Failed: {gblup_n_failed}")
                            print()
                            print(f"  Overall Metrics:")
                            print(f"    MAE  = {gblup_mae:.1f} days")
                            print(f"    RMSE = {gblup_rmse:.1f} days")
                            print(f"    Bias = {gblup_bias:+.1f} days")
                            print(f"    Median error = {gblup_median_err:+.1f} days")
                            print()
                            print(f"  Accuracy Breakdown:")
                            for threshold in [3, 5, 7, 10, 15, 20]:
                                count = int(np.sum(gblup_abs_errors <= threshold))
                                pct = count / len(gblup_errors) * 100
                                bar = '#' * int(pct / 2)
                                print(f"    Within {threshold:2d} days: {count:4d}/{len(gblup_errors)}  ({pct:5.1f}%)  {bar}")
                            print(f"{'=' * 60}")

                            # Write GBLUP DAP metrics_summary.csv
                            gblup_metrics_out = _output_path(args, "_gblup_dap_metrics_summary.csv")
                            with open(gblup_metrics_out, "w", newline="") as mf:
                                mw = csv.writer(mf)
                                mw.writerow(["planting", "n_evaluated", "n_failed",
                                             "mae", "rmse", "bias", "median_error",
                                             "within_3d", "within_5d", "within_7d",
                                             "within_10d", "within_15d", "within_20d"])
                                mw.writerow([
                                    "ALL", len(gblup_errors), gblup_n_failed,
                                    round(gblup_mae, 2), round(gblup_rmse, 2), round(gblup_bias, 2), round(gblup_median_err, 2),
                                    int(np.sum(gblup_abs_errors <= 3)), int(np.sum(gblup_abs_errors <= 5)),
                                    int(np.sum(gblup_abs_errors <= 7)), int(np.sum(gblup_abs_errors <= 10)),
                                    int(np.sum(gblup_abs_errors <= 15)), int(np.sum(gblup_abs_errors <= 20)),
                                ])
                                for p_name in sorted(set(r["planting"] for r in gblup_dap_rows)):
                                    p_errors = np.array([float(r["error"]) for r in gblup_dap_rows
                                                         if r["planting"] == p_name and r["error"] != "NA"])
                                    p_failed = sum(1 for r in gblup_dap_rows
                                                   if r["planting"] == p_name and r["predicted_dap"] == "NA")
                                    if len(p_errors) > 0:
                                        p_abs = np.abs(p_errors)
                                        mw.writerow([
                                            p_name, len(p_errors), p_failed,
                                            round(float(np.mean(p_abs)), 2),
                                            round(float(np.sqrt(np.mean(p_errors**2))), 2),
                                            round(float(np.mean(p_errors)), 2),
                                            round(float(np.median(p_errors)), 2),
                                            int(np.sum(p_abs <= 3)), int(np.sum(p_abs <= 5)),
                                            int(np.sum(p_abs <= 7)), int(np.sum(p_abs <= 10)),
                                            int(np.sum(p_abs <= 15)), int(np.sum(p_abs <= 20)),
                                        ])
                            print(f"Wrote GBLUP DAP metrics summary to {gblup_metrics_out}")
                        else:
                            print(f"\nGBLUP DAP predictions written to: {gblup_dap_out} (no observed values for comparison)")
                    else:
                        print("\nGBLUP DAP: no predictions generated (no genotypes matched G-matrix)")
                except Exception as exc:
                    print(f"\nWarning: GBLUP DAP prediction failed: {exc}", file=sys.stderr)
            else:
                print(f"\nSkipping GBLUP DAP: EVD file not found at {evd_path}")

        # ── RN-GBLUP (Jarquin 2014) comparison method ────────────────────
        # Reuse cached rows/info from the pre-Phase-1 call, which is the
        # only RN-GBLUP invocation per run. Same results feed both coupling
        # targets (Phase 1) and the comparison table (here). If the pre-run
        # failed (e.g. EVD missing), fall back to a fresh call so crops
        # without the early block still work.
        jarquin_rows = []
        jq_info = {}
        rn_ensemble_available = False
        _pre_cache = locals().get("_rn_cache")
        if _pre_cache and _pre_cache.get("available"):
            jarquin_rows = _pre_cache["rows"]
            jq_info = _pre_cache["info"]
            rn_ensemble_available = True
        elif _pre_cache and _pre_cache.get("error"):
            # Pre-run failed; pass through the same error to user
            print(f"  ERROR: RN-GBLUP (Jarquin) failed: {_pre_cache['error']}",
                  file=sys.stderr)
            print(f"  Ensemble requires RN-GBLUP. Skipping ensemble.", file=sys.stderr)
        else:
            # No pre-run happened (e.g. single-shot path); run fresh here
            _late_cache = _run_jarquin_cached(
                pheno_df=pheno_obs,
                weather_source=args.weather,
                evd_path=evd_path,
                train_plantings=(
                    train_planting_names if train_planting_names
                    else list(set(pheno_obs["Planting"]))
                ),
                predict_plantings=predict_planting_names,
                use_photo=getattr(args, "rn_photoperiod", False),
                latitude_map=latitude_map,
                test_pheno_df=test_pheno_df,
                jarquin_output_path=_output_path(args, "_jarquin_predictions.csv"),
            )
            if _late_cache["error"]:
                print(f"  ERROR: RN-GBLUP (Jarquin) failed: {_late_cache['error']}",
                      file=sys.stderr)
                print(f"  Ensemble requires RN-GBLUP. Skipping ensemble.",
                      file=sys.stderr)
            elif _late_cache["available"]:
                jarquin_rows = _late_cache["rows"]
                jq_info = _late_cache["info"]
                rn_ensemble_available = True


        # ── Validation Comparison Table ────────────────────────────────
        def _extract_metrics(rows, error_key="error"):
            """Extract standard metrics from a list of prediction dicts."""
            errs = [float(r[error_key]) for r in rows if r[error_key] != "NA"]
            if not errs:
                return None
            ea = np.abs(errs)
            return {
                "n_evaluated": len(errs),
                "mae": round(float(np.mean(ea)), 2),
                "rmse": round(float(np.sqrt(np.mean(np.square(errs)))), 2),
                "bias": round(float(np.mean(errs)), 2),
                "median_error": round(float(np.median(errs)), 2),
                "within_3d": int(np.sum(ea <= 3)),
                "within_5d": int(np.sum(ea <= 5)),
                "within_7d": int(np.sum(ea <= 7)),
                "within_10d": int(np.sum(ea <= 10)),
            }

        comparison_rows = []

        # 1. Mechanistic (ABC posterior predictive)
        mech_m = _extract_metrics(pred_rows)
        if mech_m:
            mech_centers = ", ".join(f"{e}={mech_planting_means[e]:.1f}"
                                     for e in predict_planting_names if e in mech_planting_means) if mech_planting_means else "N/A"
            comparison_rows.append({"method": "Mechanistic", "center_used": f"{mech_centers} (posterior predictive)", **mech_m})

        # 2-4. GBLUP variants (only if GBLUP was computed)
        if gblup_dap_rows:
            # Compute overall training mean for Naive GBLUP
            train_means = gblup_info.get("train_planting_means", {})
            train_ft = pheno_obs[pheno_obs["Planting"].isin(
                train_planting_names if train_planting_names else list(set(pheno_obs["Planting"]))
            )]["ft"]
            overall_train_mean = float(train_ft.mean()) if len(train_ft) > 0 else 0.0

            # Compute observed prediction planting means for Oracle GBLUP.
            # In balanced_training_subset mode (test_pheno_df is provided),
            # compute means over the FULL observed population at each predict
            # env (training half + held-out half), since "Oracle" assumes we
            # know the true env mean.
            obs_pred_means = {}
            for pp in predict_planting_names:
                pp_ft_train = pheno_obs[pheno_obs["Planting"] == pp]["ft"]
                if test_pheno_df is not None:
                    pp_ft_test = test_pheno_df[test_pheno_df["Planting"] == pp]["ft"]
                    pp_ft = pd.concat([pp_ft_train, pp_ft_test], ignore_index=True)
                else:
                    pp_ft = pp_ft_train
                if len(pp_ft) > 0:
                    obs_pred_means[pp] = float(pp_ft.mean())

            # Naive GBLUP: use training mean as center
            naive_means = {pp: overall_train_mean for pp in predict_planting_names}
            try:
                from cgm_wgp.gblup import gblup_predict_dap
                naive_rows, _ = gblup_predict_dap(
                    evd_path=evd_path,
                    pheno_df=pheno_obs,
                    train_plantings=train_planting_names if train_planting_names else list(set(pheno_obs["Planting"])),
                    predict_plantings=predict_planting_names,
                    pred_planting_means=naive_means,
                    test_pheno_df=test_pheno_df,
                )
                naive_m = _extract_metrics(naive_rows)
                if naive_m:
                    naive_center = f"{overall_train_mean:.1f} (train mean)"
                    comparison_rows.append({"method": "Naive GBLUP", "center_used": naive_center, **naive_m})
            except Exception as exc:
                print(f"  Warning: Naive GBLUP failed: {exc}", file=sys.stderr)

            # RN-GBLUP: Jarquin (2014) Reaction Norm model (computed above)
            if jarquin_rows:
                jarquin_m = _extract_metrics(jarquin_rows)
                if jarquin_m:
                    vc_arr = jq_info["vc"]
                    if len(vc_arr) == 6:
                        sg2, se2, sp2, sge2, sgp2, sr2 = vc_arr
                        vc_note = (
                            f"sg2={sg2:.1f}, se2={se2:.1f}, sp2={sp2:.1f}, "
                            f"sge2={sge2:.2f}, sgp2={sgp2:.2f}, sr2={sr2:.1f}"
                        )
                    else:
                        sg2, se2, sge2, sr2 = vc_arr
                        vc_note = (
                            f"sg2={sg2:.1f}, se2={se2:.1f}, "
                            f"sge2={sge2:.2f}, sr2={sr2:.1f}"
                        )
                    comparison_rows.append({
                        "method": "RN-GBLUP",
                        "center_used": vc_note,
                        **jarquin_m,
                    })


            # Oracle GBLUP: use actual observed prediction planting means
            if obs_pred_means:
                try:
                    oracle_rows, _ = gblup_predict_dap(
                        evd_path=evd_path,
                        pheno_df=pheno_obs,
                        train_plantings=train_planting_names if train_planting_names else list(set(pheno_obs["Planting"])),
                        predict_plantings=predict_planting_names,
                        pred_planting_means=obs_pred_means,
                        test_pheno_df=test_pheno_df,
                    )
                    oracle_m = _extract_metrics(oracle_rows)
                    if oracle_m:
                        oracle_center = ", ".join(f"{e}={obs_pred_means[e]:.1f}" for e in predict_planting_names if e in obs_pred_means)
                        comparison_rows.append({"method": "Oracle GBLUP", "center_used": f"{oracle_center} (observed)", **oracle_m})
                except Exception as exc:
                    print(f"  Warning: Oracle GBLUP failed: {exc}", file=sys.stderr)

        # ── Joint CGM-WGP (Messina/Technow EM+L-BFGS-B) ─────────────────
        # Fits per-genotype (Theta_g, S_g, Pc_g) jointly under a multivariate
        # normal genomic prior using EM with L-BFGS-B for the inner loop.
        # Cardinal temps (Tb, Topt, Tc) are fixed from config. Runs after all
        # other methods so it can't interfere with existing results.
        joint_pred_rows = []
        if getattr(args, "run_joint", False) and predict_planting_names:
            try:
                from cgm_wgp.joint_fit import joint_fit as _joint_fit
                from cgm_wgp.joint_fit import predict_all as _joint_predict_all
                from cgm_wgp.joint_fit import predict_new_genotypes as _joint_predict_new
                from cgm_wgp.gblup import load_evd as _load_evd_joint

                print(f"\n{'─' * 60}")
                print("  Running Joint CGM-WGP (Messina/Technow)...")
                print(f"{'─' * 60}")

                # Load EVD
                _jevd_path = args.evd_path or str(get_evd_path(crop=args.crop))
                V_full, d_full, gmatrix_ids_full = _load_evd_joint(_jevd_path)

                # Build genotype intersection (pheno ∩ G-matrix, case-insensitive)
                _gmat_upper = {gid.upper(): i for i, gid in enumerate(gmatrix_ids_full)}
                _joint_gids = [g for g in genotypes if g.upper() in _gmat_upper]
                _gmat_idx = [_gmat_upper[g.upper()] for g in _joint_gids]
                V_sub = V_full[np.ix_(_gmat_idx, range(V_full.shape[1]))]

                # Filter DAP_obs to training plantings only
                _joint_DAP = {}
                for g in _joint_gids:
                    if g in DAP_obs_dict:
                        _joint_DAP[g] = {
                            e: v for e, v in DAP_obs_dict[g].items()
                            if train_planting_names is None or e in train_planting_names
                        }
                _joint_pfn = lambda g: list(_joint_DAP.get(g, {}).keys())

                _j_Tb = args.joint_Tb if args.joint_Tb is not None else args.Tb
                _j_Topt = args.joint_Topt if args.joint_Topt is not None else args.Topt
                _j_Tc = args.joint_Tc if args.joint_Tc is not None else args.Tc
                _jresult = _joint_fit(
                    T_dict=T_dict,
                    DL_dict=DL_dict,
                    DAP_obs_dict=_joint_DAP,
                    plantings_of_g_fn=_joint_pfn,
                    V=V_sub,
                    d=d_full,
                    genotype_ids=_joint_gids,
                    Tb=_j_Tb, Topt=_j_Topt, Tc=_j_Tc,
                    photo_enabled=bool(args.joint_photo_enabled),
                    photo_direction=str(args.joint_photo_direction),
                    a_fixed=float(args.joint_a_fixed),
                    Theta_bounds=tuple(args.joint_Theta_bounds),
                    S_bounds=tuple(args.joint_S_bounds),
                    Pc_bounds=tuple(args.joint_Pc_bounds),
                    max_em_iters=int(args.joint_max_em_iters),
                    em_tol=float(args.joint_em_tol),
                    lbfgsb_maxiter=int(args.joint_lbfgsb_maxiter),
                    verbose=True,
                )

                # Predict on held-out/predict plantings.
                # Pass FULL DAP_obs_dict (not filtered) for validation
                # lookup — predict_all checks test_obs first (for
                # balanced/new_varieties modes), then falls back to
                # DAP_obs_dict for LOO schemes where the predict env's
                # observed DAPs are in the full dict but not in
                # the training-filtered _joint_DAP.
                joint_pred_rows = _joint_predict_all(
                    _jresult, T_dict, DL_dict, predict_planting_names,
                    DAP_obs_dict, test_obs_by_cell,
                )

                # BLUP-predict for unseen genotypes (new_varieties / cv00)
                _joint_new = _joint_predict_new(
                    _jresult, V_full, d_full, list(gmatrix_ids_full),
                    T_dict, DL_dict, predict_planting_names,
                    DAP_obs_dict, test_obs_by_cell,
                )
                joint_pred_rows.extend(_joint_new)

                # Strip any extra keys (joint_fit.predict_new_genotypes adds
                # a "method" field that our CSV schema doesn't include).
                _joint_fields = ["id", "planting", "observed_dap", "predicted_dap", "error"]
                joint_pred_rows = [
                    {k: r.get(k) for k in _joint_fields} for r in joint_pred_rows
                ]

                # Write predictions CSV
                if joint_pred_rows:
                    _joint_out = _output_path(args, "_joint_predictions.csv")
                    with open(_joint_out, "w", newline="") as fh:
                        w = csv.DictWriter(fh, fieldnames=_joint_fields)
                        w.writeheader()
                        w.writerows(joint_pred_rows)
                    print(f"  Wrote Joint predictions to {_joint_out}")

                # Persist per-genotype fitted (Theta, S, Pc) for downstream
                # analyses (distribution figures, heritability).
                try:
                    from cgm_wgp.joint_fit import write_fitted_csv as _write_joint_fitted
                    _jparams_out = _output_path(args, "_joint_params.csv")
                    _write_joint_fitted(_jresult, _jparams_out)
                except Exception as _jexc:
                    print(f"  [warn] failed to write joint_params.csv: {_jexc}",
                          file=sys.stderr)

                # Add to validation comparison
                joint_m = _extract_metrics(joint_pred_rows)
                if joint_m:
                    comparison_rows.append({
                        "method": "Joint",
                        "center_used": f"EM/L-BFGS-B ({_jresult.n_em_iterations} iters, "
                                       f"SSE={_jresult.sse:.1f})",
                        **joint_m,
                    })
                    print(f"  Joint: MAE={joint_m['mae']:.1f}, RMSE={joint_m['rmse']:.1f}, "
                          f"n={joint_m['n_evaluated']}")

            except Exception as exc:
                print(f"  ERROR: Joint CGM-WGP failed: {exc}", file=sys.stderr)
                import traceback
                traceback.print_exc()

        if comparison_rows:
            # Method descriptions for the key
            method_descriptions = {
                "Mechanistic": "Top-K dual annealing of (alpha, beta) per genotype -> posterior predictive DAP distribution. Uncertainty-aware mechanistic predictions.",
                "Naive GBLUP": "Predicts DAP using training-set grand mean + GBLUP breeding values. Weak baseline: the planting mean is just the average across all training environments.",
                "RN-GBLUP": "Jarquin (2014) Reaction Norm GBLUP: genomic (G), environmental (E_K), and GxE kernels with REML variance components. BLUP prediction in new environment via cross-covariance.",
                "Oracle GBLUP": "Uses observed planting mean from the prediction environment + GBLUP breeding values. Upper bound for GBLUP methods (not a real prediction).",
                "Joint": "Joint CGM-WGP (Messina/Technow 2018): EM+L-BFGS-B fitting per-genotype (Theta, S, Pc) under multivariate normal genomic prior with G-inverse.",
            }

            # Write CSV
            comp_out = _output_path(args, "_validation_comparison.csv")
            train_str = ", ".join(train_planting_names) if train_planting_names else "all"
            pred_str = ", ".join(predict_planting_names) if predict_planting_names else "none"
            comp_fields = ["method", "train_plantings", "predict_plantings",
                           "n_evaluated", "mae", "rmse", "bias", "median_error",
                           "within_3d", "within_5d", "within_7d", "within_10d", "center_used", "description"]
            with open(comp_out, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=comp_fields)
                writer.writeheader()
                for crow in comparison_rows:
                    crow["description"] = method_descriptions.get(crow["method"], "")
                    crow["train_plantings"] = train_str
                    crow["predict_plantings"] = pred_str
                    writer.writerow(crow)

            # Print comparison table to stdout
            print(f"\n{'=' * 84}")
            print(f"  VALIDATION COMPARISON")
            print(f"{'=' * 84}")
            print(f"  {'Method':<24s} {'n':>5s} {'MAE':>7s} {'RMSE':>7s} {'Bias':>7s} {'<=3d':>5s} {'<=5d':>5s} {'<=7d':>5s} {'<=10d':>5s}")
            print(f"  {'-'*24} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*5} {'-'*5} {'-'*5}")
            for crow in comparison_rows:
                print(f"  {crow['method']:<24s} {crow['n_evaluated']:>5d} {crow['mae']:>7.1f} {crow['rmse']:>7.1f} {crow['bias']:>+7.1f}"
                      f" {crow['within_3d']:>5d} {crow['within_5d']:>5d} {crow['within_7d']:>5d} {crow['within_10d']:>5d}")
            print(f"{'=' * 84}")
            print(f"  Wrote validation comparison to {comp_out}")

    print(f"\nDone. Wrote {len(results)} rows to {args.out}")
    if progress_rows:
        print(f"Wrote {len(progress_rows)} planting-rows to {progress_out}")


if __name__ == "__main__":
    main()
