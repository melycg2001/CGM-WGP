from pathlib import Path
import argparse
import subprocess
import sys
import os

# Import path configuration
from cgm_wgp.config import (
    PROJECT_ROOT, DATA_DIR, GMATRIX_DIR, RESULTS_DIR, RESULTS_OUTPUT_DIR,
    get_phenotypes_path, get_evd_path,
    create_run_dir, extract_location_from_weather,
    load_crop_config, finalize_run, get_output_path,
)


def run_cmd(cmd, env=None) -> int:
    print("RUN:", " ".join(map(str, cmd)))
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        print(f"Command failed (exit {res.returncode})", file=sys.stderr)
    return res.returncode


def _build_joint_subcmd_args(args) -> list:
    """Translate the parent's joint_fit settings into CLI flags for main_fit."""
    extra = []
    if getattr(args, "joint_photo_enabled", False):
        extra += ["--joint-photo-enabled"]
    if getattr(args, "joint_Tb", None) is not None:
        extra += ["--joint-Tb", str(args.joint_Tb)]
    if getattr(args, "joint_Topt", None) is not None:
        extra += ["--joint-Topt", str(args.joint_Topt)]
    if getattr(args, "joint_Tc", None) is not None:
        extra += ["--joint-Tc", str(args.joint_Tc)]
    if getattr(args, "joint_a_fixed", None) is not None:
        extra += ["--joint-a-fixed", str(args.joint_a_fixed)]
    if getattr(args, "joint_Theta_bounds", None) is not None:
        extra += ["--joint-Theta-bounds", str(args.joint_Theta_bounds[0]),
                  str(args.joint_Theta_bounds[1])]
    if getattr(args, "joint_S_bounds", None) is not None:
        extra += ["--joint-S-bounds", str(args.joint_S_bounds[0]),
                  str(args.joint_S_bounds[1])]
    if getattr(args, "joint_Pc_bounds", None) is not None:
        extra += ["--joint-Pc-bounds", str(args.joint_Pc_bounds[0]),
                  str(args.joint_Pc_bounds[1])]
    if getattr(args, "joint_max_em_iters", None) is not None:
        extra += ["--joint-max-em-iters", str(args.joint_max_em_iters)]
    if getattr(args, "joint_em_tol", None) is not None:
        extra += ["--joint-em-tol", str(args.joint_em_tol)]
    if getattr(args, "joint_lbfgsb_maxiter", None) is not None:
        extra += ["--joint-lbfgsb-maxiter", str(args.joint_lbfgsb_maxiter)]
    if getattr(args, "joint_photo_direction", None) is not None:
        extra += ["--joint-photo-direction", str(args.joint_photo_direction)]
    return extra


def main():
    p = argparse.ArgumentParser(
        description="Orchestrate cardinal-beta fitting + GBLUP + RN-GBLUP + Joint CGM-WGP"
    )
    p.add_argument("--config", default=None,
                   help="Path to crop YAML config. Populates defaults for --crop, --Tb, --Topt, "
                        "--Tc, --phenotypes, --weather, --evd-path.")
    p.add_argument("--maxiter", type=int, default=2000, help="maxiter for dual_annealing MAP estimation (Phase 1)")
    p.add_argument("--phenotypes", default=None, help="phenotypes CSV path (default: pipeline/1_data/{crop}/phenotypes_dated.csv)")
    p.add_argument("--weather", default=None,
                   help="Path to weather CSV file OR weather directory. "
                        "Single file: e.g. pipeline/1_data/weather/FL_Hasting/weather.csv. "
                        "Directory: e.g. pipeline/1_data/weather/ (loads per-location files).")
    p.add_argument("--genotype", default=None, help="single genotype id (optional)")
    p.add_argument("--Tb", type=float, default=None)
    p.add_argument("--Topt", type=float, default=None)
    p.add_argument("--Tc", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--crop", default=None, help="Crop name for output folder (e.g. 'lettuce')")
    p.add_argument("--workers", type=int, default=0,
                   help="Number of parallel workers for genotype fitting (0 = all cores minus 2, 1 = sequential)")
    p.add_argument("--python", default=sys.executable, help="Python executable to run the scripts")
    p.add_argument("--train-plantings", default=None,
                   help="Comma-separated planting numbers to use for fitting (e.g. '1,2'). Passed through to main_fit_alpha_beta.py.")
    p.add_argument("--predict-plantings", default=None,
                   help="Comma-separated planting numbers to predict flowering on (e.g. '5,6'). Passed through to main_fit_alpha_beta.py.")

    # Posterior sampling (top-K dual annealing)
    p.add_argument("--max-posterior-samples", type=int, default=200,
                   help="Cap on posterior samples for DAP prediction (default: 200)")

    # GBLUP DAP pass-through
    p.add_argument("--evd-path", default=None,
                   help="Path to EVD.rda for GBLUP DAP prediction. Passed through to main_fit_alpha_beta.py.")
    p.add_argument("--no-gblup-dap", action="store_true",
                   help="Skip GBLUP DAP prediction.")
    p.add_argument("--alpha-bounds", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="Alpha parameter bounds (default: 0.5 8.0)")
    p.add_argument("--beta-bounds", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="Beta parameter bounds (default: 0.5 8.0)")
    p.add_argument("--fit-topt", action="store_true", default=False,
                   help="Fit Topt per genotype as 3rd parameter alongside alpha, beta")
    p.add_argument("--topt-bounds", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="Topt parameter bounds when --fit-topt is enabled (default: Tb+5, Tc-5)")
    p.add_argument("--exclude-genotypes", default=None,
                   help="Comma-separated genotype IDs to exclude from fitting and prediction (e.g. 'FA306,FA125'). Passed through.")
    p.add_argument("--n-restarts", type=int, default=1,
                   help="Number of dual_annealing restarts per genotype with different seeds (default: 1)")
    p.add_argument("--normalize-loss", action="store_true",
                   help="Divide loss variance term by number of plantings (normalizes across genotypes)")

    # Photoperiod
    p.add_argument("--photo-model", type=str, default=None,
                   choices=["sinclair", "logistic", "logistic3", "linear_plateau"],
                   help='Photoperiod response model: "sinclair", "logistic", "logistic3" (Messina), or "linear_plateau" (Grimm 1993)')
    p.add_argument("--latitude-map", default=None,
                   help='JSON string mapping locations to latitudes (overrides config)')
    p.add_argument("--photo-type", default=None, choices=["short_day", "long_day"],
                   help="Photoperiod model type: short_day (default) or long_day")
    # 3-param logistic photoperiod (Messina): g(P) = 1/(1 + a*exp(b*(P-d)))
    p.add_argument("--photo-a", type=float, default=None, help="Logistic3 amplitude param a (floor=1/(1+a))")
    p.add_argument("--photo-b", type=float, default=None, help="Logistic3 slope param b (negative=short-day)")
    p.add_argument("--photo-d", type=float, default=None, help="Logistic3 inflection point d (daylength hours)")
    p.add_argument("--photo-a-bounds", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    p.add_argument("--photo-b-bounds", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    p.add_argument("--photo-d-bounds", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    # Linear-plateau (Grimm 1993) photoperiod
    p.add_argument("--photo-n-min-bounds", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="Linear-plateau N_min bounds (night length hours)")
    p.add_argument("--photo-n-opt-bounds", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="Linear-plateau N_opt bounds (night length hours)")

    # RN-GBLUP photoperiod kernel (Jarquín extension: y = µ + g + e + p + ge + gp + ε)
    p.add_argument("--rn-photoperiod", action="store_true",
                   help="Add a photoperiod environmental kernel + G⊙P interaction kernel "
                        "to the RN-GBLUP (Jarquín) model. Default OFF (4-VC byte-identical).")

    p.add_argument("--mech-no-photo", action="store_true",
                   help="Skip the bolted-on photoperiod stage (temp-only mech). For "
                        "the photo/temp/mech diagnostic comparison.")
    p.add_argument("--mech-dual-threshold", action="store_true",
                   help="Use dual-threshold mech+photo coupling (two independent "
                        "state variables, pred_dap = max(day_thermal, day_photo)) "
                        "instead of multiplicative cardinal_beta × F_photo.")

    # Top-K dual annealing
    p.add_argument("--top-k", type=int, default=100,
                   help="Top-K points to keep from dual annealing (default: 100)")
    p.add_argument("--no-loo-variance", action="store_true",
                   help="Disable LOO-planting variance; use global training MSE for ensemble (old behavior)")
    p.add_argument("--full-loo", action="store_true",
                   help="Enable Full LOO: re-fit (alpha, beta) per fold for genuinely independent "
                        "LOO variance estimates. ~5x slower than main fitting.")
    p.add_argument("--full-loo-maxiter", type=int, default=None,
                   help="Override --maxiter for Full LOO fold fits (default: uses --maxiter)")

    # Perturbed observation restarts (posterior diversity)
    p.add_argument("--obs-perturbation-sd", type=float, default=0.0,
                   help="Observation noise std for perturbed restarts (default: 0 = off)")
    p.add_argument("--obs-perturbation-sd-auto", action="store_true",
                   help="Auto-estimate obs perturbation SD from Round 1 training residuals")
    p.add_argument("--perturb-maxiter", type=int, default=0,
                   help="DA iterations for perturbed restarts (r > 0). "
                        "Default: 0 = auto (max(200, maxiter // 10)).")

    # Cross-validation
    p.add_argument("--cv-scheme", default=None,
                   help="CV scheme name from config cv: section (e.g. 'individual', 'seasonal'). "
                        "Runs all folds, combines predictions, generates comparison plots.")

    # Run labeling
    p.add_argument("--run-label", default=None,
                   help="Human-readable label for this run (stored in meta.json and index.json)")

    # Joint CGM-WGP (Messina/Technow)
    p.add_argument("--run-joint", action="store_true",
                   help="Run Joint CGM-WGP (EM+L-BFGS-B) as comparison method")

    # Joint CGM-WGP per-method overrides (read from config.model.joint_fit)
    p.add_argument("--joint-photo-enabled", action="store_true",
                   help="Enable photoperiod g(P) in Joint forward model.")
    p.add_argument("--joint-Tb", type=float, default=None)
    p.add_argument("--joint-Topt", type=float, default=None)
    p.add_argument("--joint-Tc", type=float, default=None)
    p.add_argument("--joint-a-fixed", type=float, default=None)
    p.add_argument("--joint-Theta-bounds", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"))
    p.add_argument("--joint-S-bounds", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"))
    p.add_argument("--joint-Pc-bounds", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"))
    p.add_argument("--joint-max-em-iters", type=int, default=None)
    p.add_argument("--joint-em-tol", type=float, default=None)
    p.add_argument("--joint-lbfgsb-maxiter", type=int, default=None)
    p.add_argument("--joint-photo-direction", choices=["short_day", "long_day"],
                   default=None,
                   help="Photoperiod response direction for Joint additive model.")

    args = p.parse_args()

    # Load config and apply defaults (explicit CLI args override config)
    if args.config:
        cfg = load_crop_config(config_path=args.config)
        model = cfg.get("model", {})
        weather_cfg = cfg.get("weather", {})

        if args.crop is None:
            args.crop = cfg.get("crop", "unknown")
        if args.Tb is None:
            args.Tb = model.get("Tb", 5.0)
        if args.Topt is None:
            args.Topt = model.get("Topt", 20.0)
        if args.Tc is None:
            args.Tc = model.get("Tc", 35.0)
        if args.alpha_bounds is None:
            args.alpha_bounds = model.get("alpha_bounds", [0.5, 8.0])
        if args.beta_bounds is None:
            args.beta_bounds = model.get("beta_bounds", [0.5, 8.0])
        if not args.fit_topt and model.get("fit_topt", False):
            args.fit_topt = True
        if args.topt_bounds is None and model.get("topt_bounds") is not None:
            args.topt_bounds = [float(x) for x in model["topt_bounds"]]
        if args.train_plantings is None and "train_plantings" in model:
            args.train_plantings = ",".join(model["train_plantings"])
        if args.predict_plantings is None and "predict_plantings" in model:
            args.predict_plantings = ",".join(model["predict_plantings"])
        if args.exclude_genotypes is None:
            excl = model.get("exclude_genotypes", [])
            if excl:
                args.exclude_genotypes = ",".join(str(x) for x in excl)

        # Photoperiod from config (only if CLI didn't override)
        photo_cfg = model.get("photoperiod", {})
        if args.photo_type is None:
            args.photo_type = photo_cfg.get("type", "short_day")
        if args.photo_model is None:
            if photo_cfg:
                # Config has a photoperiod block → use its model or default logistic3
                args.photo_model = photo_cfg.get("model", "logistic3")
            else:
                # No photoperiod block in config → default to logistic3 but force
                # --mech-no-photo so the bolted-on stage is skipped. Without an
                # explicit photoperiod: block, the photo fit would run on seasonal
                # DL variation (single-location confounding), producing wrong
                # predictions. The user can override with --photo-model logistic3
                # to explicitly enable photo fitting.
                args.photo_model = "logistic3"
                if not getattr(args, 'mech_no_photo', False):
                    args.mech_no_photo = True
        # Logistic3 params from config
        if args.photo_a is None and photo_cfg.get("a") is not None:
            args.photo_a = float(photo_cfg["a"])
        if args.photo_b is None and photo_cfg.get("b") is not None:
            args.photo_b = float(photo_cfg["b"])
        if args.photo_d is None and photo_cfg.get("d") is not None:
            args.photo_d = float(photo_cfg["d"])
        if args.photo_a_bounds is None and photo_cfg.get("a_bounds"):
            args.photo_a_bounds = [float(x) for x in photo_cfg["a_bounds"]]
        if args.photo_b_bounds is None and photo_cfg.get("b_bounds"):
            args.photo_b_bounds = [float(x) for x in photo_cfg["b_bounds"]]
        if args.photo_d_bounds is None and photo_cfg.get("d_bounds"):
            args.photo_d_bounds = [float(x) for x in photo_cfg["d_bounds"]]
        # Linear-plateau (Grimm 1993) photoperiod bounds
        if not hasattr(args, 'photo_n_min_bounds') or args.photo_n_min_bounds is None:
            args.photo_n_min_bounds = None
        if not hasattr(args, 'photo_n_opt_bounds') or args.photo_n_opt_bounds is None:
            args.photo_n_opt_bounds = None
        if photo_cfg.get("n_min_bounds"):
            args.photo_n_min_bounds = [float(x) for x in photo_cfg["n_min_bounds"]]
        if photo_cfg.get("n_opt_bounds"):
            args.photo_n_opt_bounds = [float(x) for x in photo_cfg["n_opt_bounds"]]

        # RN-GBLUP photoperiod kernel from config
        rn_cfg = model.get("rn_gblup", {})
        if not args.rn_photoperiod and rn_cfg.get("use_photoperiod", False):
            args.rn_photoperiod = True

        # Mech+photo coupling mode from config.
        # photoperiod.coupling: dual_threshold — enables --mech-dual-threshold
        # CLI flag still overrides (only promotes False→True here, never the
        # other way). Absence of the field keeps the previous default
        # (multiplicative daily rates).
        if not getattr(args, 'mech_dual_threshold', False):
            if photo_cfg.get("coupling") == "dual_threshold":
                args.mech_dual_threshold = True

        # Top-K / posterior-diversity config (the surviving "iterative" knobs)
        iter_cfg = model.get("iterative", {})
        if args.top_k == 100 and "top_k" in iter_cfg:
            args.top_k = int(iter_cfg["top_k"])
        if args.obs_perturbation_sd == 0.0 and "obs_perturbation_sd" in iter_cfg:
            args.obs_perturbation_sd = float(iter_cfg["obs_perturbation_sd"])
        if not args.obs_perturbation_sd_auto and iter_cfg.get("obs_perturbation_sd_auto", False):
            args.obs_perturbation_sd_auto = True
        if args.perturb_maxiter == 0 and "perturb_maxiter" in iter_cfg:
            args.perturb_maxiter = int(iter_cfg["perturb_maxiter"])

        # Methods from config (default: all methods enabled)
        # If methods list is present, only listed methods run.
        # If absent, all methods run (backward compatible).
        methods_cfg = model.get("methods")
        if methods_cfg is not None:
            args._methods = [m.lower().strip() for m in methods_cfg]
        else:
            args._methods = None  # means "all"

        # Joint CGM-WGP: enable + per-method param overrides from config
        joint_cfg = model.get("joint_fit", {})
        if not getattr(args, 'run_joint', False) and joint_cfg.get("enabled", False):
            args.run_joint = True
        # Only pull from config when the CLI didn't explicitly set a value.
        if not args.joint_photo_enabled and joint_cfg.get("photo_enabled", False):
            args.joint_photo_enabled = True
        if args.joint_Tb is None and "Tb" in joint_cfg:
            args.joint_Tb = float(joint_cfg["Tb"])
        if args.joint_Topt is None and "Topt" in joint_cfg:
            args.joint_Topt = float(joint_cfg["Topt"])
        if args.joint_Tc is None and "Tc" in joint_cfg:
            args.joint_Tc = float(joint_cfg["Tc"])
        if args.joint_a_fixed is None and "a_fixed" in joint_cfg:
            args.joint_a_fixed = float(joint_cfg["a_fixed"])
        if args.joint_Theta_bounds is None and "Theta_bounds" in joint_cfg:
            args.joint_Theta_bounds = [float(x) for x in joint_cfg["Theta_bounds"]]
        if args.joint_S_bounds is None and "S_bounds" in joint_cfg:
            args.joint_S_bounds = [float(x) for x in joint_cfg["S_bounds"]]
        if args.joint_Pc_bounds is None and "Pc_bounds" in joint_cfg:
            args.joint_Pc_bounds = [float(x) for x in joint_cfg["Pc_bounds"]]
        if args.joint_max_em_iters is None and "max_em_iters" in joint_cfg:
            args.joint_max_em_iters = int(joint_cfg["max_em_iters"])
        if args.joint_em_tol is None and "em_tol" in joint_cfg:
            args.joint_em_tol = float(joint_cfg["em_tol"])
        if args.joint_lbfgsb_maxiter is None and "lbfgsb_maxiter" in joint_cfg:
            args.joint_lbfgsb_maxiter = int(joint_cfg["lbfgsb_maxiter"])
        if args.joint_photo_direction is None:
            # Prefer explicit joint_fit.photo_direction; else fall back to
            # photoperiod.type (short_day/long_day) from the config.
            jdir = joint_cfg.get("photo_direction")
            if jdir is None:
                jdir = model.get("photoperiod", {}).get("type")
            if jdir in ("short_day", "long_day"):
                args.joint_photo_direction = jdir

        # Latitude map from config locations section
        if args.latitude_map is None:
            locations_cfg = cfg.get("locations", {})
            if locations_cfg:
                import json
                lat_map = {}
                for loc_name, loc_info in locations_cfg.items():
                    if isinstance(loc_info, dict) and "latitude" in loc_info:
                        lat_map[loc_name] = float(loc_info["latitude"])
                if lat_map:
                    args.latitude_map = json.dumps(lat_map)

        # Weather: derive from config if not given on CLI
        if args.weather is None:
            w_fmt = weather_cfg.get("format", "")
            w_src = weather_cfg.get("source", "")
            if w_fmt == "single_location":
                src = Path(w_src)
                if not src.is_absolute():
                    src = PROJECT_ROOT / src
                if src.is_dir():
                    args.weather = str(src / "weather.csv")
                else:
                    args.weather = str(src)
            elif w_fmt in ("multi_location_dir", "combined_csv"):
                from cgm_wgp.config import WEATHER_DIR
                args.weather = str(WEATHER_DIR)
            else:
                args.weather = str(w_src)

    # Apply defaults for any remaining None values
    if args.crop is None:
        args.crop = "unknown"
    if args.Tb is None:
        args.Tb = 5.0
    if args.Topt is None:
        args.Topt = 20.0
    if args.Tc is None:
        args.Tc = 35.0
    if args.alpha_bounds is None:
        args.alpha_bounds = [0.5, 8.0]
    if args.beta_bounds is None:
        args.beta_bounds = [0.5, 8.0]
    if args.photo_type is None:
        args.photo_type = "short_day"
    if args.photo_model is None:
        # Default to logistic3 — the only canonical mech+photo path.
        # sinclair/logistic are legacy and will be rejected at runtime.
        args.photo_model = "logistic3"

    if args.weather is None:
        print("Error: --weather is required (or provide --config with weather settings)", file=sys.stderr)
        sys.exit(1)

    # ── Single canonical mech+photo path guard (orchestrator level) ────
    # The only supported mech+photo path is logistic3. sinclair/logistic
    # are legacy joint-coupling paths that run α/β and photo B/Pc in the
    # same dual-annealing objective, which contradicts the architecture
    # documented in TECHNICAL.md. main_fit.py has the same guard, but
    # since the orchestrator historically stripped --photo-model when it
    # equaled the old default ("sinclair"), a silent fallthrough was
    # possible. Fail loudly here so neither the orchestrator nor any
    # subprocess can run the legacy path. Exempt --photo-only (pure
    # photoperiod diagnostic with zero thermal).
    _LEGACY_JOINT_MODELS = ("sinclair", "logistic")
    if (args.photo_model in _LEGACY_JOINT_MODELS
            and not getattr(args, "photo_only", False)):
        print(
            f"Error: photo_model='{args.photo_model}' is a legacy "
            f"joint-coupling path and is no longer supported in the "
            f"mech+photo pipeline. The canonical architecture is "
            f"photo_model='logistic3' (temp-only α/β fit → pure photo-only "
            f"shape fit → multiply at recompute). Set photoperiod.model: "
            f"logistic3 in your config (with a_bounds/b_bounds/d_bounds) "
            f"or pass --photo-model logistic3. (Note: sinclair/logistic "
            f"are still allowed under --photo-only for pure-photoperiod "
            f"diagnostics.)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Post-parse: resolve crop-aware defaults
    if args.phenotypes is None:
        args.phenotypes = str(get_phenotypes_path(crop=args.crop))
    if args.evd_path is None:
        args.evd_path = str(get_evd_path(crop=args.crop))

    env = os.environ.copy()

    # ── CV scheme mode: run all folds, combine, plot, then exit ──
    if args.cv_scheme:
        _run_cv_scheme(args, env)
        return

    # Create timestamped run folder (structured: runs/{crop}/{location}/{ts}/)
    location = extract_location_from_weather(args.weather)
    run_dir = create_run_dir(crop=args.crop, location=location,
                             label=getattr(args, "run_label", None))
    print(f"\n{'='*60}")
    print(f"Run output folder: {run_dir}")
    print(f"{'='*60}\n")

    # ── Build main_fit command ──
    cmd = [
        args.python, "-m", "cgm_wgp.main_fit",
        "--phenotypes", args.phenotypes,
        "--weather", args.weather,
        "--Tb", str(args.Tb),
        "--Topt", str(args.Topt),
        "--Tc", str(args.Tc),
        "--maxiter", str(args.maxiter),
        "--seed", str(args.seed),
        "--run-dir", str(run_dir),
        "--crop", args.crop,
    ]
    if args.run_label:
        cmd += ["--run-label", args.run_label]
    if args.genotype:
        cmd += ["--genotype", args.genotype]

    cmd += ["--max-posterior-samples", str(args.max_posterior_samples)]

    if args.train_plantings:
        cmd += ["--train-plantings", args.train_plantings]
    if args.predict_plantings:
        cmd += ["--predict-plantings", args.predict_plantings]
    cmd += ["--workers", str(args.workers)]

    # GBLUP DAP pass-through
    if args.evd_path:
        cmd += ["--evd-path", args.evd_path]
    if args.no_gblup_dap:
        cmd += ["--no-gblup-dap"]
    if args.alpha_bounds != [0.5, 8.0]:
        cmd += ["--alpha-bounds", str(args.alpha_bounds[0]), str(args.alpha_bounds[1])]
    if args.beta_bounds != [0.5, 8.0]:
        cmd += ["--beta-bounds", str(args.beta_bounds[0]), str(args.beta_bounds[1])]
    if args.fit_topt:
        cmd += ["--fit-topt"]
    if args.topt_bounds is not None:
        cmd += ["--topt-bounds", str(args.topt_bounds[0]), str(args.topt_bounds[1])]
    if args.n_restarts != 1:
        cmd += ["--n-restarts", str(args.n_restarts)]
    if args.normalize_loss:
        cmd += ["--normalize-loss"]
    if args.exclude_genotypes:
        cmd += ["--exclude-genotypes", args.exclude_genotypes]

    # Photoperiod pass-through
    if args.latitude_map is not None:
        cmd += ["--latitude-map", args.latitude_map]
    if args.photo_type != "short_day":
        cmd += ["--photo-type", args.photo_type]
    # Always pass --photo-model through so the subprocess gets the exact
    # value from config/CLI (not the subprocess's own default). This is
    # required for the canonical-path guard in main_fit.py to see the
    # intended model and reject legacy joint-coupling paths.
    cmd += ["--photo-model", args.photo_model]
    # Linear-plateau photoperiod bounds (Grimm 1993)
    if getattr(args, 'photo_n_min_bounds', None):
        cmd += ["--photo-n-min-bounds", str(args.photo_n_min_bounds[0]), str(args.photo_n_min_bounds[1])]
    if getattr(args, 'photo_n_opt_bounds', None):
        cmd += ["--photo-n-opt-bounds", str(args.photo_n_opt_bounds[0]), str(args.photo_n_opt_bounds[1])]
    # Logistic3 photoperiod bounds — pass through whenever photo_model is logistic3
    # so the new bolted-on DAP-error fit can use configured bounds.
    if args.photo_model == "logistic3":
        if getattr(args, 'photo_a_bounds', None):
            cmd += ["--photo-a-bounds", str(args.photo_a_bounds[0]), str(args.photo_a_bounds[1])]
        if getattr(args, 'photo_b_bounds', None):
            cmd += ["--photo-b-bounds", str(args.photo_b_bounds[0]), str(args.photo_b_bounds[1])]
        if getattr(args, 'photo_d_bounds', None):
            cmd += ["--photo-d-bounds", str(args.photo_d_bounds[0]), str(args.photo_d_bounds[1])]
    # RN-GBLUP photoperiod kernel
    if getattr(args, 'rn_photoperiod', False):
        cmd += ["--rn-photoperiod"]
    # Mech+photo coupling
    if getattr(args, 'mech_no_photo', False):
        cmd += ["--mech-no-photo"]
    if getattr(args, 'mech_dual_threshold', False):
        cmd += ["--mech-dual-threshold"]

    # Top-K dual annealing pass-through
    if args.top_k != 100:
        cmd += ["--top-k", str(args.top_k)]
    if args.no_loo_variance:
        cmd += ["--no-loo-variance"]
    if args.full_loo:
        cmd += ["--full-loo"]
    if args.full_loo_maxiter is not None:
        cmd += ["--full-loo-maxiter", str(args.full_loo_maxiter)]
    if args.obs_perturbation_sd > 0:
        cmd += ["--obs-perturbation-sd", str(args.obs_perturbation_sd)]
    if args.obs_perturbation_sd_auto:
        cmd += ["--obs-perturbation-sd-auto"]
    if args.perturb_maxiter > 0:
        cmd += ["--perturb-maxiter", str(args.perturb_maxiter)]
    if getattr(args, 'run_joint', False):
        cmd += ["--run-joint"]
        cmd += _build_joint_subcmd_args(args)

    # ── Run ──
    rc = run_cmd(cmd, env=env)
    if rc != 0:
        sys.exit(rc)

    # ── Auto-generate per-run diagnostic plots ──
    _generate_run_plots(run_dir)

    # ── Finalize: update meta.json with metrics from diagnostics/metrics.csv ──
    _finalize_run_metrics(run_dir)

    print(f"\nWorkflow completed successfully.")
    print(f"All outputs in: {run_dir}")


def _run_cv_scheme(args, env):
    """Run a cross-validation scheme: all folds, combine predictions, generate plots."""
    import pandas as pd
    from cgm_wgp.config import load_crop_config, OUTPUT_DIR, create_run_dir, extract_location_from_weather

    if not args.config:
        print("Error: --cv-scheme requires --config", file=sys.stderr)
        sys.exit(1)

    cfg = load_crop_config(config_path=args.config)
    cv_cfg = cfg.get("cv", {})
    scheme = cv_cfg.get(args.cv_scheme)
    if scheme is None:
        available = list(cv_cfg.keys())
        print(f"Error: CV scheme '{args.cv_scheme}' not found in config. Available: {available}",
              file=sys.stderr)
        sys.exit(1)

    # Detect balanced_training_subset scheme (CV2 Monte Carlo). This scheme
    # partitions (genotype × environment) cells 50/50 per env across `repetitions`
    # independent random splits, rather than holding out whole environments.
    # Each rep becomes a "fold" in the output structure, and cross-rep
    # aggregation happens in the combined step below.
    is_balanced_subset = bool(
        scheme.get("train_fraction") is not None
        and args.cv_scheme not in ("new_varieties", "cv00_double_novelty")
    )
    is_new_varieties = (args.cv_scheme == "new_varieties")
    is_double_novelty = (args.cv_scheme == "cv00_double_novelty")

    # Resolve folds
    if is_new_varieties:
        # Whole-genotype holdout. Randomly selects (1 - train_fraction) of
        # genotypes and removes them from training in EVERY environment.
        # Tests pure genomic generalization: can the G-matrix predict a
        # line that was never phenotyped anywhere?
        train_fraction = float(scheme.get("train_fraction", 0.8))
        repetitions = int(scheme.get("repetitions", 5))
        random_seed = int(scheme.get("random_seed", 42))
        folds = {f"rep_{i+1}": None for i in range(repetitions)}
        print(f"\n  new_varieties CV (whole-genotype holdout):")
        print(f"    train_fraction={train_fraction}, repetitions={repetitions}, "
              f"random_seed={random_seed}")
    elif is_double_novelty:
        # Joint novelty: for each environment, hold out that env entirely
        # AND (1 - train_fraction) of genotypes simultaneously.
        # Test set = unseen genotypes × unseen environment.
        # One fold per environment.
        train_fraction = float(scheme.get("train_fraction", 0.8))
        random_seed = int(scheme.get("random_seed", 42))
        pheno_df = pd.read_csv(args.phenotypes)
        _col = "Planting" if "Planting" in pheno_df.columns else "planting"
        all_dn_envs = sorted(pheno_df[_col].astype(str).unique())
        folds = {}
        for p in all_dn_envs:
            others = [x for x in all_dn_envs if x != p]
            folds[str(p)] = {"predict": [str(p)], "train": [str(x) for x in others]}
        print(f"\n  cv00_double_novelty (unseen genotypes × unseen environment):")
        print(f"    train_fraction={train_fraction}, {len(folds)} folds (one per env), "
              f"random_seed={random_seed}")
    elif is_balanced_subset:
        # Per-rep (gid, env) partitioning happens later (we need pheno_df).
        # Mark folds as a placeholder list of rep names so downstream code
        # can iterate.
        train_fraction = float(scheme.get("train_fraction", 0.5))
        repetitions = int(scheme.get("repetitions", 5))
        random_seed = int(scheme.get("random_seed", 42))
        if train_fraction <= 0 or train_fraction >= 1:
            print(f"Error: balanced_training_subset train_fraction must be in (0, 1), "
                  f"got {train_fraction}", file=sys.stderr)
            sys.exit(1)
        folds = {f"rep_{i+1}": None for i in range(repetitions)}
        print(f"\n  balanced_training_subset CV2:")
        print(f"    train_fraction={train_fraction}, repetitions={repetitions}, "
              f"random_seed={random_seed}")
    elif scheme.get("leave_one_out"):
        # Auto-detect all plantings from phenotype file
        pheno_df = pd.read_csv(args.phenotypes)
        col = "Planting" if "Planting" in pheno_df.columns else "planting"
        all_plantings = sorted(pheno_df[col].unique())
        folds = {}
        for p in all_plantings:
            others = [x for x in all_plantings if x != p]
            folds[str(p)] = {"predict": [str(p)], "train": [str(x) for x in others]}
    else:
        folds = scheme.get("folds", {})
        if not folds:
            print(f"Error: CV scheme '{args.cv_scheme}' has no folds defined", file=sys.stderr)
            sys.exit(1)

    crop = args.crop
    label_base = args.run_label or f"{crop.lower()}-{args.cv_scheme}"

    # Create parent output directory for the CV run
    location = extract_location_from_weather(args.weather)
    cv_run_dir = create_run_dir(crop=crop, location=location,
                                label=f"{label_base}",
                                cv_scheme=args.cv_scheme)
    folds_dir = cv_run_dir / "folds"
    folds_dir.mkdir(exist_ok=True)
    combined_dir = cv_run_dir / "combined"
    combined_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CV Scheme: {args.cv_scheme} ({len(folds)} folds)")
    print(f"  Output: {cv_run_dir}")
    print(f"{'='*60}")

    # Build base command (everything except train/predict/run-dir/run-label)
    base_cmd = [
        args.python, "-m", "cgm_wgp.main_fit",
        "--phenotypes", args.phenotypes,
        "--weather", args.weather,
        "--Tb", str(args.Tb), "--Topt", str(args.Topt), "--Tc", str(args.Tc),
        "--maxiter", str(args.maxiter), "--seed", str(args.seed),
        "--crop", crop,
        "--max-posterior-samples", str(args.max_posterior_samples),
        "--workers", str(args.workers),
    ]
    if args.evd_path:
        base_cmd += ["--evd-path", args.evd_path]
    if args.no_gblup_dap:
        base_cmd += ["--no-gblup-dap"]
    if args.alpha_bounds != [0.5, 8.0]:
        base_cmd += ["--alpha-bounds", str(args.alpha_bounds[0]), str(args.alpha_bounds[1])]
    if args.beta_bounds != [0.5, 8.0]:
        base_cmd += ["--beta-bounds", str(args.beta_bounds[0]), str(args.beta_bounds[1])]
    if args.fit_topt:
        base_cmd += ["--fit-topt"]
    if args.topt_bounds:
        base_cmd += ["--topt-bounds", str(args.topt_bounds[0]), str(args.topt_bounds[1])]
    if args.exclude_genotypes:
        base_cmd += ["--exclude-genotypes", args.exclude_genotypes]
    if args.n_restarts > 1:
        base_cmd += ["--n-restarts", str(args.n_restarts)]
    if args.normalize_loss:
        base_cmd += ["--normalize-loss"]
    if args.photo_type != "short_day":
        base_cmd += ["--photo-type", args.photo_type]
    # Always pass --photo-model through (see note above).
    base_cmd += ["--photo-model", args.photo_model]
    # Linear-plateau photoperiod bounds (Grimm 1993)
    if getattr(args, 'photo_n_min_bounds', None):
        base_cmd += ["--photo-n-min-bounds", str(args.photo_n_min_bounds[0]), str(args.photo_n_min_bounds[1])]
    if getattr(args, 'photo_n_opt_bounds', None):
        base_cmd += ["--photo-n-opt-bounds", str(args.photo_n_opt_bounds[0]), str(args.photo_n_opt_bounds[1])]
    # Logistic3 photoperiod bounds — pass through whenever photo_model is logistic3
    # so the new bolted-on DAP-error fit can use configured bounds.
    if args.photo_model == "logistic3":
        if getattr(args, 'photo_a_bounds', None):
            base_cmd += ["--photo-a-bounds", str(args.photo_a_bounds[0]), str(args.photo_a_bounds[1])]
        if getattr(args, 'photo_b_bounds', None):
            base_cmd += ["--photo-b-bounds", str(args.photo_b_bounds[0]), str(args.photo_b_bounds[1])]
        if getattr(args, 'photo_d_bounds', None):
            base_cmd += ["--photo-d-bounds", str(args.photo_d_bounds[0]), str(args.photo_d_bounds[1])]
    # RN-GBLUP photoperiod kernel
    if getattr(args, 'rn_photoperiod', False):
        base_cmd += ["--rn-photoperiod"]
    # Mech+photo coupling
    if getattr(args, 'mech_no_photo', False):
        base_cmd += ["--mech-no-photo"]
    if getattr(args, 'mech_dual_threshold', False):
        base_cmd += ["--mech-dual-threshold"]
    if args.latitude_map is not None:
        base_cmd += ["--latitude-map", args.latitude_map]
    if args.top_k != 100:
        base_cmd += ["--top-k", str(args.top_k)]
    if args.obs_perturbation_sd > 0:
        base_cmd += ["--obs-perturbation-sd", str(args.obs_perturbation_sd)]
    if args.obs_perturbation_sd_auto:
        base_cmd += ["--obs-perturbation-sd-auto"]
    if getattr(args, 'run_joint', False):
        base_cmd += ["--run-joint"]
        base_cmd += _build_joint_subcmd_args(args)

    # For Monte Carlo schemes (balanced_training_subset, new_varieties,
    # cv00_double_novelty), load the master pheno once for partitioning.
    _needs_pheno_split = is_balanced_subset or is_new_varieties or is_double_novelty
    if _needs_pheno_split:
        master_pheno = pd.read_csv(args.phenotypes)
        master_pheno_cols = list(master_pheno.columns)
        _genotype_col = "id" if "id" in master_pheno_cols else "Genotype"
        _planting_col = "Planting" if "Planting" in master_pheno_cols else "planting"
        all_envs_balanced = sorted(master_pheno[_planting_col].astype(str).unique())
        import numpy as _np

    # Run each fold
    fold_dirs = []
    _partition_rows = []  # per-fold partition diagnostics (written to combined/partition_report.csv)
    for fold_name, fold_def in folds.items():
        fold_dir = folds_dir / fold_name
        fold_dir.mkdir(exist_ok=True)
        # Create structured subdirs for fold
        for sub in ("params", "predictions", "diagnostics", "plots"):
            (fold_dir / sub).mkdir(exist_ok=True)

        if is_new_varieties:
            # ── New Varieties: whole-genotype holdout ──────────────────
            # Shuffle ALL genotypes, hold out (1 - train_fraction) of them
            # from every environment. Zero genotype overlap.
            rep_idx = int(fold_name.replace("rep_", "")) - 1
            rep_seed = random_seed + rep_idx
            rng = _np.random.RandomState(rep_seed)

            all_gids = master_pheno[_genotype_col].astype(str).unique()
            all_gids = _np.array(sorted(all_gids))
            rng.shuffle(all_gids)
            n_train_g = int(round(len(all_gids) * train_fraction))
            train_gids = set(all_gids[:n_train_g])
            test_gids  = set(all_gids[n_train_g:])

            id_series = master_pheno[_genotype_col].astype(str)
            train_df = master_pheno.loc[id_series.isin(train_gids)].copy()
            test_df  = master_pheno.loc[id_series.isin(test_gids)].copy()

            train_path = fold_dir / "train_phenotypes.csv"
            test_path  = fold_dir / "test_phenotypes.csv"
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            _partition_rows.append(_partition_diagnostics(
                fold_name, args.cv_scheme, train_df, test_df,
                list(all_envs_balanced), list(all_envs_balanced),
                len(all_envs_balanced),
            ))

            predict_str = ",".join(
                str(p).replace("Planting ", "") for p in all_envs_balanced
            )
            train_str = predict_str

            print(
                f"\n  Rep '{fold_name}' (seed={rep_seed}): "
                f"train_genotypes={n_train_g}, test_genotypes={len(test_gids)}, "
                f"train_cells={len(train_df)}, test_cells={len(test_df)}"
            )

            cmd = base_cmd + [
                "--train-plantings", train_str,
                "--predict-plantings", predict_str,
                "--run-dir", str(fold_dir),
                "--run-label", f"{label_base}-{fold_name}",
            ]
            try:
                pi = cmd.index("--phenotypes")
                cmd[pi + 1] = str(train_path)
            except ValueError:
                cmd += ["--phenotypes", str(train_path)]
            cmd += ["--test-phenotypes", str(test_path)]

        elif is_double_novelty:
            # ── CV00 Double Novelty: unseen genotypes × unseen env ─────
            # For each fold: hold out one env AND (1 - train_fraction) of
            # genotypes. Train = remaining genotypes × remaining envs.
            # Test = held-out genotypes × held-out env.
            held_out_env = fold_def["predict"][0]
            train_envs = fold_def["train"]
            fold_idx = list(folds.keys()).index(fold_name)
            fold_seed = random_seed + fold_idx
            rng = _np.random.RandomState(fold_seed)

            all_gids = master_pheno[_genotype_col].astype(str).unique()
            all_gids = _np.array(sorted(all_gids))
            rng.shuffle(all_gids)
            n_train_g = int(round(len(all_gids) * train_fraction))
            train_gids = set(all_gids[:n_train_g])
            test_gids  = set(all_gids[n_train_g:])

            id_series = master_pheno[_genotype_col].astype(str)
            env_series = master_pheno[_planting_col].astype(str)
            # Train: train genotypes × train envs (no held-out env, no test genotypes)
            train_df = master_pheno.loc[
                id_series.isin(train_gids) & env_series.isin([str(e) for e in train_envs])
            ].copy()
            # Test: test genotypes × held-out env only
            test_df = master_pheno.loc[
                id_series.isin(test_gids) & (env_series == str(held_out_env))
            ].copy()

            train_path = fold_dir / "train_phenotypes.csv"
            test_path  = fold_dir / "test_phenotypes.csv"
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            _partition_rows.append(_partition_diagnostics(
                fold_name, args.cv_scheme, train_df, test_df,
                [str(e) for e in train_envs], [str(held_out_env)],
                len(all_envs_balanced),
            ))

            train_str = ",".join(str(e).replace("Planting ", "") for e in train_envs)
            predict_str = str(held_out_env).replace("Planting ", "")

            print(
                f"\n  Fold '{fold_name}': held_out_env={held_out_env}, "
                f"train_gids={n_train_g}, test_gids={len(test_gids)}, "
                f"train_cells={len(train_df)}, test_cells={len(test_df)}"
            )

            cmd = base_cmd + [
                "--train-plantings", train_str,
                "--predict-plantings", predict_str,
                "--run-dir", str(fold_dir),
                "--run-label", f"{label_base}-{fold_name}",
            ]
            try:
                pi = cmd.index("--phenotypes")
                cmd[pi + 1] = str(train_path)
            except ValueError:
                cmd += ["--phenotypes", str(train_path)]
            cmd += ["--test-phenotypes", str(test_path)]

        elif is_balanced_subset:
            # Per-rep deterministic partition: seed = base_seed + rep_index
            rep_idx = int(fold_name.replace("rep_", "")) - 1
            rep_seed = random_seed + rep_idx
            rng = _np.random.RandomState(rep_seed)

            train_mask = _np.zeros(len(master_pheno), dtype=bool)
            test_mask  = _np.zeros(len(master_pheno), dtype=bool)
            for _env_name in all_envs_balanced:
                env_rows = master_pheno[_planting_col].astype(str) == _env_name
                env_gids = master_pheno.loc[env_rows, _genotype_col].astype(str).unique()
                env_gids = _np.array(env_gids)
                rng.shuffle(env_gids)
                n_train = int(round(len(env_gids) * train_fraction))
                train_gids = set(env_gids[:n_train])
                test_gids  = set(env_gids[n_train:])
                # Mark rows by (id, env)
                id_series = master_pheno[_genotype_col].astype(str)
                train_mask |= env_rows & id_series.isin(train_gids)
                test_mask  |= env_rows & id_series.isin(test_gids)

            train_df = master_pheno.loc[train_mask].copy()
            test_df  = master_pheno.loc[test_mask].copy()

            train_path = fold_dir / "train_phenotypes.csv"
            test_path  = fold_dir / "test_phenotypes.csv"
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            _partition_rows.append(_partition_diagnostics(
                fold_name, args.cv_scheme, train_df, test_df,
                list(all_envs_balanced), list(all_envs_balanced),
                len(all_envs_balanced),
            ))

            # predict_plantings = all envs (every env is both train and test
            # in balanced mode; the test_phenotypes file tells main_fit what to
            # validate against).
            predict_str = ",".join(
                str(p).replace("Planting ", "") for p in all_envs_balanced
            )
            train_str = predict_str  # same set

            print(
                f"\n  Rep '{fold_name}' (seed={rep_seed}): "
                f"train={len(train_df)} cells, test={len(test_df)} cells, "
                f"envs={len(all_envs_balanced)}"
            )

            cmd = base_cmd + [
                "--train-plantings", train_str,
                "--predict-plantings", predict_str,
                "--run-dir", str(fold_dir),
                "--run-label", f"{label_base}-{fold_name}",
            ]
            # Override --phenotypes to point at the per-rep training file, and
            # attach --test-phenotypes for validation. Since base_cmd already
            # contains `--phenotypes <master>`, we replace those two entries.
            try:
                pi = cmd.index("--phenotypes")
                cmd[pi + 1] = str(train_path)
            except ValueError:
                cmd += ["--phenotypes", str(train_path)]
            cmd += ["--test-phenotypes", str(test_path)]
        else:
            predict = fold_def["predict"]
            train = fold_def["train"]
            predict_str = ",".join(str(p).replace("Planting ", "") for p in predict)
            train_str = ",".join(str(p).replace("Planting ", "") for p in train)

            print(f"\n  Fold '{fold_name}': predict=[{predict_str}], train=[{train_str}]")

            # Synthesize LOO partition diagnostics from master pheno
            try:
                _loo_pheno = pd.read_csv(args.phenotypes)
                _id_col = "id" if "id" in _loo_pheno.columns else "Genotype"
                _pl_col = "Planting" if "Planting" in _loo_pheno.columns else "planting"
                _loo_pheno = _loo_pheno.rename(columns={_id_col: "id", _pl_col: "Planting"})
                _loo_pheno["Planting"] = _loo_pheno["Planting"].astype(str)
                _train_envs = [str(p) for p in train]
                _test_envs  = [str(p) for p in predict]
                _train_df = _loo_pheno[_loo_pheno["Planting"].isin(_train_envs)].copy()
                _test_df  = _loo_pheno[_loo_pheno["Planting"].isin(_test_envs)].copy()
                _partition_rows.append(_partition_diagnostics(
                    fold_name, args.cv_scheme, _train_df, _test_df,
                    _train_envs, _test_envs,
                    int(_loo_pheno["Planting"].nunique()),
                ))
            except Exception as _exc:
                print(f"  [warn] LOO partition diagnostics failed: {_exc}",
                      file=sys.stderr)

            cmd = base_cmd + [
                "--train-plantings", train_str,
                "--predict-plantings", predict_str,
                "--run-dir", str(fold_dir),
                "--run-label", f"{label_base}-{fold_name}",
            ]
        rc = run_cmd(cmd, env=env)
        if rc != 0:
            print(f"  WARNING: Fold '{fold_name}' failed (exit {rc})")
            continue
        fold_dirs.append(fold_dir)

    if not fold_dirs:
        print("Error: All folds failed.", file=sys.stderr)
        sys.exit(1)

    # Write partition report and abort if any fold violated scheme invariants
    if _partition_rows:
        part_report_path = combined_dir / "partition_report.csv"
        combined_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(_partition_rows).to_csv(part_report_path, index=False)
        n_viol = sum(1 for r in _partition_rows if not r["invariant_passed"])
        if n_viol:
            print(f"\n  ❌ {n_viol}/{len(_partition_rows)} folds violated CV invariants. "
                  f"See {part_report_path}. Aborting to prevent publishing bad results.",
                  file=sys.stderr)
            sys.exit(2)
        else:
            print(f"\n  ✅ All {len(_partition_rows)} folds passed CV invariants. "
                  f"Report: {part_report_path}")

    # Combine held-out predictions across folds
    print(f"\n{'='*60}")
    print(f"  Combining {len(fold_dirs)} fold predictions")
    print(f"{'='*60}")

    for pred_type in ["mechanistic", "raw_mechanistic", "gblup", "jarquin", "joint"]:
        frames = []
        for fd in fold_dirs:
            f = fd / "predictions" / f"{pred_type}.csv"
            if f.exists():
                frames.append(pd.read_csv(f))
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined.to_csv(combined_dir / f"{pred_type}.csv", index=False)

    # Combine fitted params (deduplicate — keep first occurrence per genotype)
    param_frames = []
    for fd in fold_dirs:
        f = fd / "params" / "fitted.csv"
        if f.exists():
            param_frames.append(pd.read_csv(f))
    if param_frames:
        combined_params = pd.concat(param_frames, ignore_index=True).drop_duplicates(
            subset="id", keep="first")
        combined_params.to_csv(combined_dir / "fitted.csv", index=False)

    # Copy run_config from first fold for reference
    first_cfg = fold_dirs[0] / "diagnostics" / "run_config.json"
    if first_cfg.exists():
        import shutil
        (combined_dir / "diagnostics").mkdir(exist_ok=True)
        shutil.copy2(first_cfg, combined_dir / "diagnostics" / "run_config.json")

    # Per-run CV summary (RMSE + PA, overall + per-planting, for all cv_scheme types)
    _compute_cv_summary(combined_dir, fold_dirs, args.cv_scheme, crop)

    # For Monte Carlo schemes (balanced_training_subset, new_varieties),
    # aggregate per-rep validation comparisons into rep_summary.csv with
    # mean±std across reps for each method/metric.
    if is_balanced_subset or is_new_varieties:
        rep_frames = []
        for fd in fold_dirs:
            vc_path = fd / "diagnostics" / "validation_comparison.csv"
            if vc_path.exists():
                _df = pd.read_csv(vc_path)
                _df["rep"] = fd.name
                rep_frames.append(_df)
        if rep_frames:
            rep_all = pd.concat(rep_frames, ignore_index=True)
            # Compute per-method mean±std over numeric metric columns
            metric_cols = [c for c in ("n_evaluated", "mae", "rmse", "bias",
                                        "median_error", "within_3d", "within_5d",
                                        "within_7d", "within_10d")
                           if c in rep_all.columns]
            summary = (
                rep_all.groupby("method")[metric_cols]
                .agg(["mean", "std"])
                .round(3)
            )
            summary.columns = [f"{m}_{s}" for m, s in summary.columns]
            summary.reset_index(inplace=True)
            (combined_dir / "diagnostics").mkdir(exist_ok=True)
            rep_all.to_csv(combined_dir / "diagnostics" / "per_rep_validation.csv", index=False)
            summary.to_csv(combined_dir / "diagnostics" / "rep_summary.csv", index=False)
            print(f"\n  Cross-rep Monte Carlo summary ({len(fold_dirs)} reps):")
            for _, row in summary.iterrows():
                print(f"    {row['method']:20s}  "
                      f"MAE = {row.get('mae_mean', float('nan')):.2f} ± {row.get('mae_std', float('nan')):.2f}")

    # Generate combined plots
    print(f"\n  Generating combined plots...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from cgm_wgp.config import DATA_DIR

        plots_dir = cv_run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        pheno_path = args.phenotypes

        # Obs vs Pred for each method (include Joint / CGM-WGP).
        # Subtitle reflects the actual cv_scheme instead of a blanket "LOO-CV".
        _scheme_tag, _scheme_desc = _CV_SCHEME_LABELS.get(
            args.cv_scheme or "", ("CV", "Cross-Validation"))
        for method, fname in [("Mechanistic", "raw_mechanistic"),
                              ("Mech+GBLUP", "mechanistic"),
                              ("GBLUP", "gblup"),
                              ("RN-GBLUP", "jarquin"),
                              ("CGM-WGP", "joint")]:
            csv_f = str(combined_dir / f"{fname}.csv")
            if Path(csv_f).exists():
                _plot_obs_vs_pred(csv_f, plots_dir / f"obs_vs_pred_{fname}.png",
                                  f"Observed vs. Predicted Flowering Time: {crop}",
                                  plt, np,
                                  subtitle=f"{method} ({_scheme_tag}: {_scheme_desc})")

        # Posterior distributions
        fitted_csv = str(combined_dir / "fitted.csv")
        if Path(fitted_csv).exists():
            _plot_alpha_beta(fitted_csv, plots_dir / "alpha_beta_distribution.png",
                             crop, plt, np, pheno_path=pheno_path)

            # Thermal response curves
            if fold_dirs:
                _plot_thermal_response(fitted_csv, plots_dir / "thermal_response_curves.png",
                                       crop, fold_dirs[0], plt, np)

        # AMMI biplot
        if pheno_path and Path(pheno_path).exists():
            _plot_ammi_biplot(pheno_path, plots_dir / "ammi_biplot.png", crop, plt, np)

        # Per-scheme comparison (RMSE + Pearson r grouped bars, overall + per-env)
        generate_loo_cv_plots(fold_dirs, crop, str(plots_dir),
                              cv_scheme=args.cv_scheme)

    except ImportError:
        print("  Warning: matplotlib not available, skipping plots")

    # Update meta.json
    import json as _jj
    meta_path = cv_run_dir / "meta.json"
    if meta_path.exists():
        meta = _jj.loads(meta_path.read_text())
        meta["cv_scheme"] = args.cv_scheme
        meta["n_folds"] = len(fold_dirs)
        meta["fold_names"] = list(folds.keys())
        meta_path.write_text(_jj.dumps(meta, indent=2))

    print(f"\n{'='*60}")
    print(f"  CV complete: {len(fold_dirs)} folds")
    print(f"  Combined predictions: {combined_dir}")
    print(f"  Plots: {cv_run_dir / 'plots'}")
    print(f"  All outputs: {cv_run_dir}")
    print(f"{'='*60}")


def _finalize_run_metrics(run_dir):
    """Extract key metrics from diagnostics and write to meta.json + index."""
    metrics_file = run_dir / "diagnostics" / "metrics.csv"
    metrics = {}
    try:
        import pandas as pd
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            if "mae" in df.columns:
                metrics["mech_mae"] = round(float(df["mae"].mean()), 2)
            if "rmse" in df.columns:
                metrics["mech_rmse"] = round(float(df["rmse"].mean()), 2)
        # Count genotypes from params/fitted.csv
        fitted_file = run_dir / "params" / "fitted.csv"
        if fitted_file.exists():
            df = pd.read_csv(fitted_file)
            metrics["n_genotypes"] = len(df)
    except Exception:
        pass
    if metrics:
        finalize_run(run_dir, metrics)


def _generate_run_plots(run_dir):
    """Generate diagnostic plots for a completed run."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  Warning: matplotlib not available, skipping plots")
        return

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Determine crop name from meta.json for plot titles
    crop_name = "Unknown"
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        try:
            import json as _j
            crop_name = _j.loads(meta_path.read_text()).get("crop", "Unknown")
        except Exception:
            pass

    # Obs vs Pred scatter for each prediction method
    for method, fname in [("Mechanistic", "mechanistic"), ("G-BLUP", "gblup"),
                          ("RN-GBLUP", "jarquin")]:
        pred_csv = str(run_dir / "predictions" / f"{fname}.csv")
        if Path(pred_csv).exists():
            _plot_obs_vs_pred(pred_csv, plots_dir / f"obs_vs_pred_{fname}.png",
                              f"Observed vs. Predicted Flowering Time: {crop_name}",
                              plt, np, subtitle=method)

    # Resolve phenotype path (used by parameter distributions + AMMI biplot)
    run_cfg_path = run_dir / "diagnostics" / "run_config.json"
    pheno_path = None
    if run_cfg_path.exists():
        try:
            import json as _j2
            rc = _j2.loads(run_cfg_path.read_text())
            pp = rc.get("phenotypes")
            if pp and Path(pp).exists():
                pheno_path = pp
        except Exception:
            pass
    if pheno_path is None:
        from cgm_wgp.config import DATA_DIR
        candidate = DATA_DIR / crop_name / "phenotypes_dated.csv"
        if candidate.exists():
            pheno_path = str(candidate)

    # 3. Posterior parameter distributions (alpha, beta, flowering time)
    fitted = str(run_dir / "params" / "fitted.csv")
    _plot_alpha_beta(fitted, plots_dir / "alpha_beta_distribution.png",
                     crop_name, plt, np, pheno_path=pheno_path)

    # 4. Convergence history (if iterative)
    conv_file = str(run_dir / "diagnostics" / "convergence.csv")
    if Path(conv_file).exists():
        _plot_convergence(conv_file, plots_dir / "convergence_history.png", plt, np)

    # 5. AMMI biplot (if phenotypes available with multiple environments)
    if pheno_path:
        _plot_ammi_biplot(pheno_path, plots_dir / "ammi_biplot.png",
                          crop_name, plt, np)

    # 6. Cardinal temperature response curves (10 diverse genotypes)
    _plot_thermal_response(fitted, plots_dir / "thermal_response_curves.png",
                           crop_name, run_dir, plt, np)

    print(f"  Per-run plots saved to: {plots_dir}")


def _plot_obs_vs_pred(csv_path, out_path, title, plt, np, subtitle=None):
    """Minimalist scatter plot of observed vs predicted flowering time, colored by planting."""
    import csv as _csv
    from scipy.stats import pearsonr
    if not Path(csv_path).exists():
        return

    # Read data grouped by planting
    data = {}  # {planting: [(obs, pred), ...]}
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            try:
                o, p = float(row["observed_dap"]), float(row["predicted_dap"])
            except (ValueError, TypeError, KeyError):
                continue
            if not (np.isfinite(o) and np.isfinite(p)):
                continue
            planting = row.get("planting", "all")
            data.setdefault(planting, []).append((o, p))

    all_obs = [v for pts in data.values() for v, _ in pts]
    all_pred = [v for pts in data.values() for _, v in pts]
    if len(all_obs) < 2:
        return

    all_obs_arr, all_pred_arr = np.array(all_obs), np.array(all_pred)

    # Muted color palette
    palette = ["#5B8DB8", "#E8985E", "#6BAF6B", "#D47B7B", "#9B8DC4",
               "#C4956A", "#E48AC0", "#8DC4C4", "#B8B84E", "#C48A8A"]

    fig, ax = plt.subplots(figsize=(6.5, 6))

    from scipy.stats import spearmanr

    for i, (planting, pts) in enumerate(sorted(data.items())):
        obs_p = np.array([v[0] for v in pts])
        pred_p = np.array([v[1] for v in pts])
        if len(pts) >= 3:
            pr = np.corrcoef(obs_p, pred_p)[0, 1]
            sr, _ = spearmanr(obs_p, pred_p)
            lbl = f"{planting} (p={pr:.2f}, s={sr:.2f}, n={len(pts)})"
        else:
            lbl = f"{planting} (n={len(pts)})"
        ax.scatter(obs_p, pred_p, alpha=0.55, s=22,
                   color=palette[i % len(palette)], edgecolors="none", label=lbl)

    # Auto-clip axis if outliers compress the main data
    # Use 1.5*IQR on combined obs+pred to find a sensible range
    all_vals = np.concatenate([all_obs_arr, all_pred_arr])
    q1, q3 = np.percentile(all_vals, [2, 98])
    iqr = q3 - q1
    clip_lo = max(all_vals.min(), q1 - 2.0 * iqr) - 2
    clip_hi = min(all_vals.max(), q3 + 2.0 * iqr) + 2
    # Only clip if the range would shrink meaningfully (outlier present)
    raw_lo = all_vals.min() - 2
    raw_hi = all_vals.max() + 2
    n_outside = int(np.sum((all_obs_arr < clip_lo) | (all_obs_arr > clip_hi) |
                           (all_pred_arr < clip_lo) | (all_pred_arr > clip_hi)))
    if n_outside > 0 and (raw_hi - raw_lo) > 1.5 * (clip_hi - clip_lo):
        lo, hi = clip_lo, clip_hi
    else:
        lo, hi = raw_lo, raw_hi
        n_outside = 0

    # 1:1 reference line
    ax.plot([lo, hi], [lo, hi], color="#999999", ls="--", lw=0.8, zorder=0)

    # Overall stats: r (Pearson), s (Spearman), RMSE, n
    r_val, _ = pearsonr(all_obs_arr, all_pred_arr)
    sr_val, _ = spearmanr(all_obs_arr, all_pred_arr)
    rmse = np.sqrt(np.mean((all_pred_arr - all_obs_arr) ** 2))
    n = len(all_obs)
    stat_text = f"r={r_val:.3f}, s={sr_val:.3f}\nRMSE={rmse:.1f}d\nn={n}"
    if n_outside > 0:
        stat_text += f"\n({n_outside} outlier{'s' if n_outside > 1 else ''} clipped)"
    ax.text(0.04, 0.96, stat_text, transform=ax.transAxes, fontsize=8.5,
            va="top", ha="left", color="#333333",
            fontfamily="sans-serif")

    # Minimalist styling
    ax.set_xlabel("Observed DAP", fontsize=10, color="#333333")
    ax.set_ylabel("Predicted DAP", fontsize=10, color="#333333")
    full_title = title
    if subtitle:
        full_title = f"{title}\n{subtitle}"
    ax.set_title(full_title, fontsize=12, fontweight="bold", color="#222222", pad=12)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    # Clean up spines and ticks
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#CCCCCC")
    ax.tick_params(colors="#666666", labelsize=9)
    ax.legend(fontsize=7, loc="lower right", frameon=False, labelcolor="#555555")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_alpha_beta(csv_path, out_path, crop_name, plt, np, pheno_path=None):
    """Posterior distributions: histogram + KDE for alpha, beta, and observed flowering time."""
    import csv as _csv
    from scipy.stats import gaussian_kde
    if not Path(csv_path).exists():
        return
    alphas, betas = [], []
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            try:
                a, b = float(row["alpha_mean"]), float(row["beta_mean"])
            except (ValueError, TypeError, KeyError):
                continue
            if np.isfinite(a) and np.isfinite(b):
                alphas.append(a)
                betas.append(b)
    if len(alphas) < 5:
        return

    alphas, betas = np.array(alphas), np.array(betas)

    # Load observed flowering time from phenotypes
    ft_vals = None
    if pheno_path and Path(pheno_path).exists():
        try:
            import pandas as pd
            df = pd.read_csv(pheno_path)
            if "ft" in df.columns:
                ft = df["ft"].dropna()
                ft = ft[ft > 0].values
                if len(ft) >= 5:
                    ft_vals = ft
        except Exception:
            pass

    # Build panels list
    panels = [
        (alphas, "alpha", "#5B8DB8"),
        (betas, "beta", "#5B8DB8"),
    ]
    if ft_vals is not None:
        panels.append((ft_vals, "Flowering Time (DAP)", "#6BAF6B"))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    for ax, (vals, label, color) in zip(axes, panels):
        med = np.median(vals)
        n_bins = min(30, max(10, len(vals) // 6))

        ax.hist(vals, bins=n_bins, density=True, color=color, alpha=0.45,
                edgecolor="white", linewidth=0.4, zorder=2)

        try:
            kde = gaussian_kde(vals, bw_method=0.3)
            x_pad = (vals.max() - vals.min()) * 0.15 + 0.5
            x_range = np.linspace(vals.min() - x_pad, vals.max() + x_pad, 200)
            ax.plot(x_range, kde(x_range), color=color, lw=1.8, zorder=3)
        except Exception:
            pass

        ax.axvline(med, color="#222222", ls="--", lw=1.2, zorder=4)
        ax.text(0.97, 0.95, f"median={med:.1f}" if label.startswith("F") else f"median={med:.2f}",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                color="#444444", style="italic")

        ax.set_xlabel(label, fontsize=10, color="#333333")
        ax.set_ylabel("Density", fontsize=10, color="#333333")
        ax.set_title(label, fontsize=11, fontweight="bold", color="#333333")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#CCCCCC")
        ax.tick_params(colors="#666666", labelsize=8)

    fig.suptitle(f"Posterior Distributions: {crop_name}",
                 fontsize=13, fontweight="bold", color="#222222", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_convergence(csv_path, out_path, plt, np):
    """Plot convergence history (max param shift per round)."""
    import csv as _csv
    rounds, alpha_shifts, beta_shifts = [], [], []
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            rounds.append(int(row["round"]))
            a_s = row.get("max_alpha_shift", "")
            b_s = row.get("max_beta_shift", "")
            alpha_shifts.append(float(a_s) if a_s else None)
            beta_shifts.append(float(b_s) if b_s else None)
    if not rounds:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    valid_rounds = [r for r, a in zip(rounds, alpha_shifts) if a is not None]
    valid_alpha = [a for a in alpha_shifts if a is not None]
    valid_beta = [b for b in beta_shifts if b is not None]
    if valid_rounds:
        ax.plot(valid_rounds, valid_alpha, "o-", label="Max α shift", markersize=4)
        ax.plot(valid_rounds, valid_beta, "s-", label="Max β shift", markersize=4)
        ax.axhline(y=0.001, color="red", linestyle="--", alpha=0.5, label="Convergence tol")
        ax.set_yscale("log")
    ax.set_xlabel("Round")
    ax.set_ylabel("Max Parameter Shift")
    ax.set_title("Convergence History", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ammi_biplot(pheno_path, out_path, crop_name, plt, np):
    """AMMI biplot (IPCA1 vs IPCA2) from observed flowering time data."""
    import pandas as pd
    try:
        df = pd.read_csv(pheno_path)
    except Exception:
        return
    if "ft" not in df.columns or "id" not in df.columns:
        return

    # Build genotype x environment matrix
    col = "Planting" if "Planting" in df.columns else "planting"
    if col not in df.columns:
        return
    df = df.dropna(subset=["ft"])
    df = df[df["ft"] > 0]
    pivot = df.pivot_table(index="id", columns=col, values="ft", aggfunc="mean")
    pivot = pivot.dropna()  # complete cases only
    if pivot.shape[0] < 3 or pivot.shape[1] < 2:
        return

    Y = pivot.values
    geno_names = list(pivot.index)
    env_names = list(pivot.columns)
    n_g, n_e = Y.shape

    # Double-center: subtract grand mean, row means, col means
    grand_mean = Y.mean()
    row_means = Y.mean(axis=1, keepdims=True)
    col_means = Y.mean(axis=0, keepdims=True)
    R = Y - row_means - col_means + grand_mean

    # SVD on residuals
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    k = min(2, len(S))
    if k < 2 or S[0] == 0:
        return

    total_gxe_ss = np.sum(S**2)
    pct = 100 * S[:2]**2 / total_gxe_ss

    # Biplot scaling: genotypes = U * sqrt(S), environments = V * sqrt(S)
    g_scores = U[:, :2] * np.sqrt(S[:2])
    e_scores = Vt[:2, :].T * np.sqrt(S[:2])

    # Muted palette for genotypes (color by row mean FT)
    ft_means = row_means.ravel()
    ft_norm = (ft_means - ft_means.min()) / (ft_means.max() - ft_means.min() + 1e-9)

    # Environment colors — distinct per environment
    env_palette = ["#2E86AB", "#E8985E", "#4CAF50", "#D47B7B", "#9B8DC4",
                   "#C4956A", "#E48AC0", "#5B8DB8", "#B8B84E", "#C48A8A"]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Genotype points — colored by mean FT
    cmap = plt.cm.coolwarm
    sc = ax.scatter(g_scores[:, 0], g_scores[:, 1],
                    c=ft_norm, cmap=cmap, s=28, alpha=0.6,
                    edgecolors="none", zorder=2)

    # Label only outlier genotypes (beyond 1.5 IQR on either axis)
    for axis_idx in range(2):
        vals = g_scores[:, axis_idx]
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lo_thresh, hi_thresh = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        for i, name in enumerate(geno_names):
            if vals[i] < lo_thresh or vals[i] > hi_thresh:
                ax.annotate(name, (g_scores[i, 0], g_scores[i, 1]),
                            fontsize=5.5, color="#777777", ha="center", va="bottom",
                            xytext=(0, 3), textcoords="offset points")

    # Environment triangles — no text labels, just color-coded with legend
    from matplotlib.lines import Line2D
    env_legend = []
    for j, env in enumerate(env_names):
        color = env_palette[j % len(env_palette)]
        ax.scatter(e_scores[j, 0], e_scores[j, 1],
                   marker="^", s=120, color=color, edgecolors="white",
                   linewidths=0.8, zorder=4)
        env_legend.append(
            Line2D([0], [0], marker="^", color="w", markerfacecolor=color,
                   markersize=8, markeredgecolor="white", markeredgewidth=0.5,
                   label=env))

    # Reference lines
    ax.axhline(0, color="#DDDDDD", lw=0.6, zorder=0)
    ax.axvline(0, color="#DDDDDD", lw=0.6, zorder=0)

    # Minimalist styling
    ax.set_xlabel(f"IPCA1 ({pct[0]:.1f}% of GxE)", fontsize=10, color="#333333")
    ax.set_ylabel(f"IPCA2 ({pct[1]:.1f}% of GxE)", fontsize=10, color="#333333")
    ax.set_title(f"AMMI Biplot: {crop_name}", fontsize=13, fontweight="bold",
                 color="#222222", pad=12)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#CCCCCC")
    ax.tick_params(colors="#666666", labelsize=9)

    # Legend: genotypes + each environment by color
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markersize=6, label="Genotypes"),
    ] + env_legend
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7.5,
              frameon=True, facecolor="white", edgecolor="#DDDDDD",
              labelcolor="#444444")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_thermal_response(fitted_path, out_path, crop_name, run_dir, plt, np):
    """Cardinal temperature response curves for 10 diverse genotypes."""
    import csv as _csv
    import json as _json
    from cgm_wgp.mech_fit import cardinal_beta

    if not Path(fitted_path).exists():
        return

    # Read fitted params
    genotypes = []
    with open(fitted_path) as f:
        for row in _csv.DictReader(f):
            try:
                a = float(row["alpha_mean"])
                b = float(row["beta_mean"])
            except (ValueError, TypeError, KeyError):
                continue
            if np.isfinite(a) and np.isfinite(b):
                genotypes.append({"id": row["id"], "alpha": a, "beta": b})

    if len(genotypes) < 3:
        return

    # Get Tb, Topt, Tc from run_config
    Tb, Topt, Tc = 5.0, 25.0, 40.0
    cfg_path = run_dir / "diagnostics" / "run_config.json"
    if cfg_path.exists():
        try:
            rc = _json.loads(cfg_path.read_text())
            Tb = float(rc.get("Tb", 5.0))
            Topt = float(rc.get("Topt", 25.0))
            Tc = float(rc.get("Tc", 40.0))
        except Exception:
            pass

    # Select 10 diverse genotypes: spread across alpha range
    alphas = np.array([g["alpha"] for g in genotypes])
    n_pick = min(10, len(genotypes))
    # Pick evenly spaced percentiles
    percentiles = np.linspace(0, 100, n_pick + 2)[1:-1]
    targets = np.percentile(alphas, percentiles)
    selected = []
    used = set()
    for t in targets:
        dists = [(abs(g["alpha"] - t), i, g) for i, g in enumerate(genotypes) if i not in used]
        dists.sort()
        if dists:
            _, idx, g = dists[0]
            selected.append(g)
            used.add(idx)

    # Temperature range
    temps = np.linspace(Tb - 1, Tc + 1, 500)

    # Color palette
    palette = ["#D45E5E", "#5B8DB8", "#4CAF50", "#9B8DC4", "#E8985E",
               "#C4956A", "#E48AC0", "#2E86AB", "#B8B84E", "#7B68AE"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, g in enumerate(selected):
        a, b = g["alpha"], g["beta"]
        rates = cardinal_beta(temps, Tb, Topt, Tc, a, b)
        # Normalize to peak = 1
        peak = rates.max()
        if peak > 0:
            rates = rates / peak
        label = f"{g['id']}  (\u03b1={a:.2f}, \u03b2={b:.2f})"
        ax.plot(temps, rates, color=palette[i % len(palette)], lw=1.8,
                alpha=0.85, label=label)

    # Minimalist styling
    ax.set_xlabel("Temperature (\u00b0C)", fontsize=10, color="#333333")
    ax.set_ylabel("Relative Development Rate", fontsize=10, color="#333333")
    ax.set_title(f"Cardinal Temperature Response Curves \u2014 {crop_name}",
                 fontsize=12, fontweight="bold", color="#222222", pad=12)
    ax.set_xlim(Tb - 1, Tc + 1)
    ax.set_ylim(-0.02, 1.08)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#CCCCCC")
    ax.tick_params(colors="#666666", labelsize=9)

    # Legend outside plot area to avoid covering curves
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, labelcolor="#444444", borderaxespad=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _place_env_labels(ax, env_texts, fontsize=9):
    """Place environment labels with leader lines, repelled from each other and the marker."""
    import numpy as np
    if not env_texts:
        return

    coords = np.array([(x, y) for x, y, _, _ in env_texts])
    n = len(coords)
    base_offset = 36  # points — far enough to clear the triangle and neighboring labels

    # Normalize coords to display space for distance calc
    # Use data-to-display transform
    display_coords = np.array([ax.transData.transform((x, y)) for x, y, _, _ in env_texts])

    # Compute repulsion-based direction (in display space)
    offsets = np.zeros((n, 2))
    for i in range(n):
        dx, dy = 0.0, 0.0
        for j in range(n):
            if i == j:
                continue
            diff = display_coords[i] - display_coords[j]
            dist = np.linalg.norm(diff) + 1e-6
            dx += diff[0] / (dist ** 1.5)
            dy += diff[1] / (dist ** 1.5)
        # Also repel from plot center
        center = display_coords.mean(axis=0)
        diff_c = display_coords[i] - center
        dist_c = np.linalg.norm(diff_c) + 1e-6
        dx += diff_c[0] / (dist_c * 0.5)
        dy += diff_c[1] / (dist_c * 0.5)

        norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-9
        offsets[i] = [dx / norm * base_offset, dy / norm * base_offset]

    # Second pass: check for collisions in offset space and add perpendicular kick
    for i in range(n):
        for j in range(i + 1, n):
            label_dist = np.linalg.norm(offsets[i] - offsets[j])
            marker_dist = np.linalg.norm(display_coords[i] - display_coords[j])
            if label_dist < base_offset * 1.2 or marker_dist < base_offset * 2:
                # Kick apart perpendicular to the line between them
                perp = np.array([-offsets[i][1], offsets[i][0]])
                perp = perp / (np.linalg.norm(perp) + 1e-9) * base_offset * 0.6
                offsets[i] += perp
                offsets[j] -= perp

    for i, (x, y, label, color) in enumerate(env_texts):
        ax.annotate(label, (x, y),
                    fontsize=fontsize, fontweight="bold", color=color,
                    ha="center", va="center",
                    xytext=(offsets[i, 0], offsets[i, 1]),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color=color, alpha=0.4, lw=0.8))


def _partition_diagnostics(fold_name, cv_scheme, train_df, test_df,
                            train_envs_list, test_envs_list, total_expected_envs):
    """Compute partition diagnostics for one fold and return a dict row.

    Asserts per-scheme invariants and prints a red-flag message on violation.
    Caller is responsible for aggregating rows and deciding whether to exit.

    Parameters
    ----------
    fold_name : str
    cv_scheme : str
    train_df, test_df : DataFrame (must have 'id' and 'Planting' cols)
    train_envs_list, test_envs_list : list[str]
        The env names that are in train / test for this fold (authoritative,
        not derived from the dfs — covers LOO where test_df may be the full
        pheno filtered only by env).
    total_expected_envs : int
        Number of distinct envs in the dataset (for sanity-checks).
    """
    train_gids = set(train_df["id"].astype(str).unique()) if train_df is not None else set()
    test_gids  = set(test_df["id"].astype(str).unique())  if test_df  is not None else set()
    n_train_cells = len(train_df) if train_df is not None else 0
    n_test_cells  = len(test_df)  if test_df  is not None else 0
    overlap = test_gids & train_gids
    overlap_pct = 100.0 * len(overlap) / len(test_gids) if test_gids else float("nan")

    # avg envs per test gid within training (key CV2 vs CV1 differentiator)
    avg_train_envs_per_test_gid = float("nan")
    if train_df is not None and test_gids:
        sub = train_df[train_df["id"].astype(str).isin(test_gids)]
        if len(sub):
            avg_train_envs_per_test_gid = float(
                sub.groupby("id")["Planting"].nunique().mean()
            )
        else:
            avg_train_envs_per_test_gid = 0.0

    # Invariants per scheme
    violations = []
    if cv_scheme == "balanced_training_subset":
        if overlap_pct < 95.0:
            violations.append(
                f"CV2 expects test gids ≥95% present in training (gap-filling); got {overlap_pct:.1f}%")
    elif cv_scheme == "new_varieties":
        if overlap_pct != 0.0 and not (overlap_pct != overlap_pct):  # not NaN
            violations.append(
                f"CV1 expects test gid overlap = 0% (whole-gid holdout); got {overlap_pct:.1f}%")
    elif cv_scheme == "individual":
        test_env_set = set(test_envs_list)
        train_env_set = set(train_envs_list)
        if test_env_set & train_env_set:
            violations.append(
                f"CV0 expects held-out env absent from train; got overlap {test_env_set & train_env_set}")
    elif cv_scheme == "cv00_double_novelty":
        test_env_set = set(test_envs_list)
        train_env_set = set(train_envs_list)
        if test_env_set & train_env_set:
            violations.append(
                f"CV00 expects held-out env absent from train; got overlap {test_env_set & train_env_set}")
        if overlap_pct != 0.0 and not (overlap_pct != overlap_pct):
            violations.append(
                f"CV00 expects test gid overlap with train = 0%; got {overlap_pct:.1f}%")

    passed = not violations
    if not passed:
        print(f"\n  🚩 PARTITION VIOLATION in fold '{fold_name}' ({cv_scheme}):")
        for v in violations:
            print(f"     - {v}")

    return {
        "fold_name": fold_name,
        "cv_scheme": cv_scheme,
        "n_train_cells": n_train_cells,
        "n_test_cells": n_test_cells,
        "n_train_gids": len(train_gids),
        "n_test_gids": len(test_gids),
        "test_gid_overlap_pct": round(overlap_pct, 1) if overlap_pct == overlap_pct else "NA",
        "avg_train_envs_per_test_gid": (
            round(avg_train_envs_per_test_gid, 2)
            if avg_train_envs_per_test_gid == avg_train_envs_per_test_gid else "NA"
        ),
        "train_envs": "|".join(sorted(train_envs_list)),
        "test_envs": "|".join(sorted(test_envs_list)),
        "invariant_passed": passed,
        "violations": "; ".join(violations) if violations else "",
    }


def _compute_cv_summary(combined_dir, fold_dirs, cv_scheme, crop):
    """Compute per-method RMSE and PA (overall + per-planting) from combined predictions.

    Writes combined_dir / "cv_summary.csv" with columns:
        crop, cv_scheme, method, planting, n, rmse, pa
    """
    import pandas as pd
    import numpy as np
    from scipy.stats import pearsonr
    from pathlib import Path

    # Method list comes from the central registry — adding a new method is
    # one entry in src/cgm_wgp/methods.py, no edits needed here.
    from cgm_wgp.methods import in_summary as _registry_in_summary
    METHODS = [(s.pred_file, s.display) for s in _registry_in_summary()]

    # The raw mechanistic baseline uses a single population-mean (α̅, β̅)
    # applied uniformly to all genotypes and cannot differentiate them. In
    # CV schemes whose evaluation axis is genotype novelty (CV1, CV00) this
    # baseline is degenerate — every held-out genotype in a given environment
    # receives the same prediction — so we omit it from the summary there.
    if cv_scheme in ("new_varieties", "cv00_double_novelty"):
        METHODS = [(f, d) for (f, d) in METHODS if d != "Mechanistic"]

    rows = []
    for pred_file, display_name in METHODS:
        csv_path = Path(combined_dir) / f"{pred_file}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df["observed_dap"] = pd.to_numeric(df.get("observed_dap"), errors="coerce")
        df["predicted_dap"] = pd.to_numeric(df.get("predicted_dap"), errors="coerce")
        df = df.dropna(subset=["observed_dap", "predicted_dap"])
        if len(df) < 3:
            continue

        obs = df["observed_dap"].values
        pred = df["predicted_dap"].values
        r_all, _ = pearsonr(obs, pred)
        rmse_all = float(np.sqrt(np.mean((pred - obs) ** 2)))
        rows.append({
            "crop": crop, "cv_scheme": cv_scheme, "method": display_name,
            "planting": "Overall", "n": len(df),
            "rmse": round(rmse_all, 2), "pa": round(r_all, 4),
        })

        for planting, grp in df.groupby("planting"):
            o = grp["observed_dap"].values
            p = grp["predicted_dap"].values
            if len(o) < 3:
                continue
            r_val, _ = pearsonr(o, p)
            rmse_val = float(np.sqrt(np.mean((p - o) ** 2)))
            rows.append({
                "crop": crop, "cv_scheme": cv_scheme, "method": display_name,
                "planting": str(planting), "n": len(o),
                "rmse": round(rmse_val, 2), "pa": round(r_val, 4),
            })

    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(Path(combined_dir) / "cv_summary.csv", index=False)
        print(f"\n  CV summary ({len(rows)} rows) → {combined_dir}/cv_summary.csv")
        overall = out[out["planting"] == "Overall"]
        for _, r in overall.iterrows():
            print(f"    {r['method']:14s}  RMSE={r['rmse']:6.2f}  PA={r['pa']:.3f}  n={r['n']}")


_CV_SCHEME_LABELS = {
    "balanced_training_subset": ("CV2", "Gap-Filling"),
    "new_varieties":            ("CV1", "New Varieties"),
    "individual":               ("CV0", "New Environment (LOO)"),
    "cv00_double_novelty":      ("CV00", "Double Novelty"),
}


def generate_loo_cv_plots(run_dirs, crop_name, out_dir, cv_scheme=None):
    """Generate per-scheme comparison plots (Pearson r and RMSE by model, colored by planting).

    Args:
        run_dirs: list of Path objects, one per fold (rep or LOO env)
        crop_name: e.g. "Broccoli", "Beans"
        out_dir: directory to write plots into
        cv_scheme: CV scheme key; drives title + filename prefix so outputs from
            different schemes are distinguishable when browsing the plots dir.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # MODELS list comes from the central registry. Methods can be added/
    # removed by editing src/cgm_wgp/methods.py only.
    from cgm_wgp.methods import in_paper as _registry_in_paper
    MODELS = [(s.display, s.pred_file) for s in _registry_in_paper()]

    # Collect all predictions across folds
    all_preds = {}  # {model_name: DataFrame with id, planting, observed_dap, predicted_dap}
    for model_name, pred_file in MODELS:
        frames = []
        for rd in run_dirs:
            rd = Path(rd)
            # Structured path
            f = rd / "predictions" / f"{pred_file}.csv"
            if not f.exists():
                # Legacy path
                f = rd / f"alpha_beta_results_{pred_file}_predictions.csv"
            if not f.exists():
                continue
            try:
                df = pd.read_csv(f)
                if "observed_dap" in df.columns and "predicted_dap" in df.columns:
                    frames.append(df[["id", "planting", "observed_dap", "predicted_dap"]].dropna())
            except Exception:
                continue
        if frames:
            all_preds[model_name] = pd.concat(frames, ignore_index=True)

    if not all_preds:
        print("  No prediction data found for LOO-CV plots")
        return

    # Compute per-planting metrics for each model
    # {model: {planting: {r, rmse, n}}}
    model_metrics = {}
    model_overall = {}
    all_plantings = set()

    for model_name, df in all_preds.items():
        per_plant = {}
        for planting, grp in df.groupby("planting"):
            obs = grp["observed_dap"].values
            pred = grp["predicted_dap"].values
            if len(obs) < 3:
                continue
            r_val, _ = pearsonr(obs, pred)
            rmse = np.sqrt(np.mean((pred - obs) ** 2))
            per_plant[planting] = {"r": r_val, "rmse": rmse, "n": len(obs)}
            all_plantings.add(planting)
        model_metrics[model_name] = per_plant

        # Overall
        obs_all = df["observed_dap"].values
        pred_all = df["predicted_dap"].values
        if len(obs_all) >= 3:
            r_all, _ = pearsonr(obs_all, pred_all)
            rmse_all = np.sqrt(np.mean((pred_all - obs_all) ** 2))
            model_overall[model_name] = {"r": r_all, "rmse": rmse_all, "n": len(obs_all)}

    # Order models as defined, but only those with data
    model_order = [m for m, _ in MODELS if m in model_metrics]
    plantings_sorted = sorted(all_plantings)

    # Planting color palette
    palette = ["#5B8DB8", "#E8985E", "#4CAF50", "#D47B7B", "#9B8DC4",
               "#C4956A", "#E48AC0", "#8DC4C4", "#B8B84E", "#C48A8A"]
    planting_colors = {p: palette[i % len(palette)] for i, p in enumerate(plantings_sorted)}

    tag, description = _CV_SCHEME_LABELS.get(cv_scheme or "", ("CV", "Cross-Validation"))
    file_prefix = tag.lower()

    _plot_loo_metric(model_order, model_metrics, model_overall, plantings_sorted,
                     planting_colors, "r",
                     f"{crop_name} Flowering Time: {tag} ({description}) Prediction Accuracy (Pearson r)",
                     "Pearson r", out_dir / f"{file_prefix}_pearson_r.png", plt, np)

    _plot_loo_metric(model_order, model_metrics, model_overall, plantings_sorted,
                     planting_colors, "rmse",
                     f"{crop_name} Flowering Time: {tag} ({description}) RMSE",
                     "RMSE (days)", out_dir / f"{file_prefix}_rmse.png", plt, np)

    print(f"  {tag} comparison plots saved to: {out_dir}")


def _plot_loo_metric(model_order, model_metrics, model_overall, plantings_sorted,
                     planting_colors, metric_key, title, ylabel, out_path, plt, np):
    """Grouped bar chart of LOO-CV metric by model, colored by planting, with value labels."""

    n_models = len(model_order)
    n_plantings = len(plantings_sorted)

    fig, ax = plt.subplots(figsize=(max(7, n_models * 2.2), 5.5))

    x_positions = np.arange(n_models)
    bar_width = 0.72 / max(n_plantings, 1)
    group_width = bar_width * n_plantings
    offset_start = -group_width / 2 + bar_width / 2

    # Grouped bars per planting
    for j, planting in enumerate(plantings_sorted):
        x_bars = []
        y_bars = []
        for i, model in enumerate(model_order):
            m = model_metrics.get(model, {}).get(planting)
            if m:
                x_bars.append(x_positions[i] + offset_start + j * bar_width)
                y_bars.append(m[metric_key])
            else:
                x_bars.append(x_positions[i] + offset_start + j * bar_width)
                y_bars.append(0)

        # Clamp negative values to a small sliver for display, but keep real value for label
        y_display = [max(0.02 * max(abs(v) for v in y_bars if v != 0) if any(v != 0 for v in y_bars) else 1, v)
                     for v in y_bars]
        bars = ax.bar(x_bars, y_display, width=bar_width * 0.9,
                       color=planting_colors[planting], alpha=0.85,
                       label=planting, edgecolor="white", linewidth=0.3,
                       zorder=2)

        # Value labels — show real value even if bar is clamped to 0
        for bar, real_val, disp_val in zip(bars, y_bars, y_display):
            if real_val != 0:
                fmt = f"{real_val:.2f}" if metric_key == "r" else f"{real_val:.1f}"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        max(disp_val, 0.01),  # slight offset so label isn't on axis
                        fmt, ha="center", va="bottom", fontsize=6.5,
                        color="#444444", fontweight="medium")

    # Overall dashed line per model
    for i, model in enumerate(model_order):
        ov = model_overall.get(model)
        if ov:
            val = ov[metric_key]
            half = group_width / 2 + bar_width * 0.3
            ax.plot([x_positions[i] - half, x_positions[i] + half], [val, val],
                    color="#333333", ls="--", lw=1.3, zorder=4, alpha=0.65)

    ax.set_ylim(bottom=0)

    # Minimalist styling
    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_order, fontsize=10, color="#333333")
    ax.set_xlabel("Model", fontsize=10, color="#333333")
    ax.set_ylabel(ylabel, fontsize=10, color="#333333")
    ax.set_title(title, fontsize=12, fontweight="bold", color="#222222", pad=14)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#CCCCCC")
    ax.tick_params(colors="#666666", labelsize=9)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="best",
              frameon=True, facecolor="white", edgecolor="#DDDDDD",
              labelcolor="#444444", ncol=min(3, n_plantings))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
