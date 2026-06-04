"""
Path configuration for CGM-WGP project.
Centralized path management for all scripts.
"""

from pathlib import Path
from datetime import datetime
import json as _json
import subprocess
import sys

# Project root: for editable installs, __file__ is src/cgm_wgp/config.py → 3 levels up.
# CGM_WGP_ROOT env var overrides for non-editable or relocated deployments.
import os as _os
PROJECT_ROOT = Path(
    _os.environ.get("CGM_WGP_ROOT", str(Path(__file__).resolve().parents[2]))
)

# Pipeline directories (Phase 2 structure)
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
DATA_DIR = PIPELINE_DIR / "1_data"
GMATRIX_DIR = PIPELINE_DIR / "2_gmatrix"
CV_PREP_DIR = PIPELINE_DIR / "3_cv_prep"
ALPHA_MODELS_DIR = PIPELINE_DIR / "4_alpha_models"
BETA_MODELS_DIR = PIPELINE_DIR / "4_beta_models"
RESULTS_DIR = PIPELINE_DIR / "5_results"

# Backward compatibility (old numbered folder names)
# Use these constants in code, they'll map to new locations
ROOT_DATA_DIR = DATA_DIR
GMATRIX_OLD = GMATRIX_DIR
PHENOS_CV_DIR = CV_PREP_DIR
RUN_MODELS_DIR = ALPHA_MODELS_DIR
RUN_BETA_MODELS_DIR = BETA_MODELS_DIR
RESULTS_OLD = RESULTS_DIR

# Output subdirectories
RESULTS_OUTPUT_DIR = RESULTS_DIR / "output"
RESULTS_BETA_OUTPUT_DIR = RESULTS_DIR / "beta_output"

# Shared weather directory (used by all crops)
WEATHER_DIR = DATA_DIR / "weather"

# Timestamped run output directory
OUTPUT_DIR = PIPELINE_DIR / "output"

# Crop config directory
CONFIGS_DIR = PROJECT_ROOT / "configs"


def extract_location_from_weather(weather_path: str) -> str:
    """Extract location name from weather file or directory path.

    Single-file mode:
      'pipeline/1_data/weather/FL_Hasting/weather.csv' -> 'FL_Hasting'
    Multi-location directory mode:
      'pipeline/1_data/weather/' -> 'multi'
    """
    p = Path(weather_path)
    if p.is_dir():
        return "multi"
    return p.parent.name


def _get_git_sha() -> str:
    """Return short git SHA of HEAD, or '' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=PROJECT_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


# Structured output subdirectory names
RUN_SUBDIRS = ("params", "predictions", "diagnostics", "plots")


def create_run_dir(
    crop: str = "unknown",
    location: str = "unknown",
    label: str | None = None,
    cv_scheme: str | None = None,
) -> Path:
    """Create and return a timestamped run output directory with structured subdirs.

    Format: pipeline/output/runs/{crop}/{location}/{YYYY-MM-DD_HH-MM-SS}/
        params/          — fitted parameters (alpha, beta, photoperiod)
        predictions/     — model predictions (mechanistic, ensemble, gblup, etc.)
        diagnostics/     — convergence, metrics, iteration detail
        plots/           — per-run diagnostic plots

    Also writes initial meta.json, updates latest symlink, and appends to index.json.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = OUTPUT_DIR / "runs" / crop / location / ts

    # Create structured subdirectories
    for sub in RUN_SUBDIRS:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    # Write initial meta.json
    meta = {
        "crop": crop,
        "location": location,
        "created": ts,
        "label": label,
        "tags": [],
        "git_sha": _get_git_sha(),
        "metrics": {},
        "cv_scheme": cv_scheme,
    }
    (run_dir / "meta.json").write_text(_json.dumps(meta, indent=2))

    # Update latest symlink
    latest = run_dir.parent / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.name)
    except OSError:
        pass  # symlink creation may fail on some filesystems

    # Append to index.json
    _append_to_index(meta, run_dir)

    return run_dir


def get_output_path(run_dir: Path, category: str, filename: str) -> str:
    """Return the full path for an output file in a structured run directory.

    Args:
        run_dir: The run's root directory.
        category: One of 'params', 'predictions', 'diagnostics', 'plots',
                  or 'root' for files at the run directory level.
        filename: The output filename (e.g. 'fitted.csv').
    """
    if category == "root":
        return str(run_dir / filename)
    return str(run_dir / category / filename)


def finalize_run(run_dir: Path, metrics: dict):
    """Update meta.json with post-run metrics and refresh the index entry."""
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return
    meta = _json.loads(meta_path.read_text())
    meta["metrics"] = metrics
    meta_path.write_text(_json.dumps(meta, indent=2))
    _update_index_entry(run_dir, metrics)


def _append_to_index(meta: dict, run_dir: Path):
    """Append a run entry to the global index.json."""
    index_path = OUTPUT_DIR / "index.json"
    entries = []
    if index_path.exists():
        try:
            entries = _json.loads(index_path.read_text())
        except (_json.JSONDecodeError, ValueError):
            entries = []
    entry = dict(meta)
    entry["path"] = str(run_dir.relative_to(OUTPUT_DIR))
    entries.append(entry)
    index_path.write_text(_json.dumps(entries, indent=2))


def _update_index_entry(run_dir: Path, metrics: dict):
    """Update the metrics for an existing entry in index.json."""
    index_path = OUTPUT_DIR / "index.json"
    if not index_path.exists():
        return
    try:
        entries = _json.loads(index_path.read_text())
    except (_json.JSONDecodeError, ValueError):
        return
    rel = str(run_dir.relative_to(OUTPUT_DIR))
    for entry in entries:
        if entry.get("path") == rel:
            entry["metrics"] = metrics
            break
    index_path.write_text(_json.dumps(entries, indent=2))


def resolve_run(crop: str, location: str = None, tag: str = None,
                label: str = None) -> Path | None:
    """Resolve a run directory from the index by tag or label.

    Returns the Path to the run directory, or None if not found.
    """
    index_path = OUTPUT_DIR / "index.json"
    if not index_path.exists():
        return None
    try:
        entries = _json.loads(index_path.read_text())
    except (_json.JSONDecodeError, ValueError):
        return None

    candidates = [e for e in entries if e.get("crop") == crop]
    if location:
        candidates = [e for e in candidates if e.get("location") == location]
    if tag:
        candidates = [e for e in candidates if tag in e.get("tags", [])]
    if label:
        candidates = [e for e in candidates if e.get("label") == label]
    if not candidates:
        return None
    # Return the most recent match
    best = sorted(candidates, key=lambda e: e.get("created", ""))[-1]
    return OUTPUT_DIR / best["path"]

# Crop-specific directory helpers
def get_crop_data_dir(crop: str = "Broccoli") -> Path:
    """Get path to a crop's data directory under 1_data/."""
    return DATA_DIR / crop

def get_crop_gmatrix_dir(crop: str = "Broccoli") -> Path:
    """Get path to a crop's G-matrix directory under 2_gmatrix/."""
    return GMATRIX_DIR / crop

# Common file paths
def get_phenotypes_path(crop: str = "Broccoli", filename: str = "phenotypes_dated.csv") -> Path:
    """Get path to phenotypes file for a crop."""
    return DATA_DIR / crop / filename

def get_weather_path(crop: str = "Broccoli", filename: str = "weather.csv") -> Path:
    """Get path to weather file for a crop."""
    return DATA_DIR / crop / filename

def get_weather_dir() -> Path:
    """Get path to the shared weather directory: pipeline/1_data/weather/"""
    return WEATHER_DIR

def get_evd_path(crop: str = "Broccoli") -> Path:
    """Get path to EVD.rda for a crop."""
    return GMATRIX_DIR / crop / "EVD.rda"

def get_gmatrix_path(crop: str = "Broccoli") -> Path:
    """Get path to Gmatrix.rda for a crop."""
    return GMATRIX_DIR / crop / "Gmatrix.rda"

def load_crop_config(config_path: str | None = None, crop: str | None = None) -> dict:
    """Load a crop YAML config.

    If config_path provided, load that file.
    If only crop provided, look for configs/{crop.lower()}.yaml
    Returns the parsed config dict.
    """
    import yaml

    if config_path:
        p = Path(config_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
    elif crop:
        p = CONFIGS_DIR / f"{crop.lower()}.yaml"
    else:
        raise ValueError("Provide config_path or crop name")

    if not p.exists():
        print(f"Error: Config not found: {p}", file=sys.stderr)
        raise FileNotFoundError(f"Config not found: {p}")

    with open(p) as f:
        config = yaml.safe_load(f)

    return config


def get_gblup_global_rmse():
    """Get path to global RMSE file from GBLUP."""
    return RESULTS_OUTPUT_DIR / "global_rmse.csv"

def get_per_genotype_targets():
    """Get path to per-genotype GBLUP targets."""
    return RESULTS_OUTPUT_DIR / "per_genotype_targets.csv"

def get_rmse_history():
    """Get path to RMSE iteration history."""
    return RESULTS_OUTPUT_DIR / "rmse_history.csv"

# Print paths for debugging
if __name__ == "__main__":
    print("CGM-WGP Path Configuration")
    print("=" * 60)
    print(f"Project Root:        {PROJECT_ROOT}")
    print(f"Pipeline Dir:        {PIPELINE_DIR}")
    print(f"Data Dir:            {DATA_DIR}")
    print(f"G-matrix Dir:        {GMATRIX_DIR}")
    print(f"CV Prep Dir:         {CV_PREP_DIR}")
    print(f"Alpha Models Dir:    {ALPHA_MODELS_DIR}")
    print(f"Beta Models Dir:     {BETA_MODELS_DIR}")
    print(f"Results Dir:         {RESULTS_DIR}")
    print("=" * 60)
    print(f"Weather Dir (shared):{WEATHER_DIR}")
    print(f"Broccoli Data Dir:   {get_crop_data_dir('Broccoli')}")
    print(f"Broccoli GMatrix:    {get_crop_gmatrix_dir('Broccoli')}")
    print(f"Phenotypes:          {get_phenotypes_path()}")
    print(f"EVD:                 {get_evd_path()}")
    print(f"Global RMSE:         {get_gblup_global_rmse()}")
    print(f"Per-gen Targets:     {get_per_genotype_targets()}")
    print(f"RMSE History:        {get_rmse_history()}")
