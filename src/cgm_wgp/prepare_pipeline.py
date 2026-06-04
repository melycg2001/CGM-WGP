"""
prepare_pipeline.py

Config-driven data preparation for the CGM-WGP pipeline.
Reads a YAML config and prepares phenotypes, weather, and genomic data.

Usage:
  python scripts/prepare_pipeline.py --config configs/beans.yaml
  python scripts/prepare_pipeline.py --config configs/beans.yaml --validate-only
  python scripts/prepare_pipeline.py --config configs/beans.yaml --force
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from cgm_wgp.config import (
    PROJECT_ROOT, DATA_DIR, GMATRIX_DIR, WEATHER_DIR,
    get_crop_data_dir, get_crop_gmatrix_dir, get_evd_path, get_gmatrix_path,
    load_crop_config,
)
from cgm_wgp.config_schema import validate_config
from cgm_wgp.prepare_crop_data import prepare_phenotypes, prepare_phenotypes_wide
from cgm_wgp.generate_evd import generate_evd_from_gmatrix, generate_evd_from_csv


# ── Helpers ──────────────────────────────────────────────────────────────────

def year_doy_to_date(year: int, doy: int) -> str:
    """Convert year + day-of-year to YYYY-MM-DD string."""
    dt = datetime(int(year), 1, 1) + timedelta(days=int(doy) - 1)
    return dt.strftime("%Y-%m-%d")


def resolve_path(p: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT if not absolute."""
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


# ── Directory setup ──────────────────────────────────────────────────────────

def prepare_directories(crop: str):
    """Create pipeline directories for a crop if they don't exist."""
    dirs = [
        get_crop_data_dir(crop),
        get_crop_gmatrix_dir(crop),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  Directory: {d}")


# ── Phenotype preparation ───────────────────────────────────────────────────

def prepare_phenotype_data(config: dict, force: bool = False) -> Path | None:
    """Prepare phenotype data based on config. Returns output path or None."""
    crop = config["crop"]
    pheno = config["phenotypes"]
    fmt = pheno["format"]
    source = resolve_path(pheno["source"])

    out_path = get_crop_data_dir(crop) / "phenotypes_dated.csv"

    if fmt == "pipeline_ready":
        # Already in expected format — validate it exists
        if not source.exists():
            print(f"  ERROR: pipeline_ready source not found: {source}", file=sys.stderr)
            return None
        # If source is not the expected location, copy it
        if source.resolve() != out_path.resolve():
            if out_path.exists() and not force:
                print(f"  phenotypes_dated.csv already exists (use --force to overwrite)")
                return out_path
            shutil.copy(str(source), str(out_path))
            print(f"  Copied {source} -> {out_path}")
        else:
            print(f"  phenotypes_dated.csv already in place: {out_path}")
        return out_path

    if out_path.exists() and not force:
        print(f"  phenotypes_dated.csv already exists (use --force to overwrite)")
        return out_path

    layout = pheno.get("layout", "long")

    if layout == "wide":
        return prepare_phenotypes_wide(
            input_path=str(source),
            crop=crop,
            output_path=str(out_path),
            genotype_col=pheno.get("genotype_col", "RIL"),
            ft_columns=pheno.get("ft_columns"),
            planting_dates=pheno.get("planting_dates"),
            missing_value=pheno.get("missing_value", "-"),
            sheet=pheno.get("sheet"),
        )
    else:
        return prepare_phenotypes(
            input_path=str(source),
            crop=crop,
            output_path=str(out_path),
            genotype_col=pheno.get("genotype_col", "Genotype"),
            location_col=pheno.get("location_col", "Location"),
            ft_col=pheno.get("ft_col", "R1"),
            rep_col=pheno.get("rep_col"),
            planting_dates=pheno.get("planting_dates"),
            season=pheno.get("season"),
            normalize_locations=pheno.get("normalize_locations", False),
            location_map=pheno.get("location_map"),
        )


# ── Weather preparation ─────────────────────────────────────────────────────

def prepare_weather_data(config: dict, force: bool = False) -> bool:
    """Prepare weather data based on config. Returns True on success."""
    weather = config["weather"]
    fmt = weather["format"]
    source = resolve_path(weather["source"])

    if fmt == "single_location":
        # Expect source to be a directory with weather.csv inside
        wx_file = source / "weather.csv" if source.is_dir() else source
        if not wx_file.exists():
            print(f"  ERROR: weather file not found: {wx_file}", file=sys.stderr)
            return False
        print(f"  Single-location weather OK: {wx_file}")
        return True

    if fmt == "multi_location_dir":
        if not source.is_dir():
            print(f"  ERROR: weather directory not found: {source}", file=sys.stderr)
            return False
        # Check that subdirectories have weather.csv
        subdirs = [d for d in source.iterdir() if d.is_dir()]
        ok = True
        for d in sorted(subdirs):
            wx = d / "weather.csv"
            if wx.exists():
                print(f"  Weather OK: {d.name}/weather.csv")
            else:
                print(f"  WARNING: missing {d.name}/weather.csv")
        print(f"  Multi-location weather directory: {len(subdirs)} locations")
        return True

    if fmt == "combined_csv":
        return _split_combined_weather(config, force)

    print(f"  ERROR: Unknown weather format '{fmt}'", file=sys.stderr)
    return False


def _split_combined_weather(config: dict, force: bool = False) -> bool:
    """Split a combined weather CSV into per-location weather.csv files."""
    weather = config["weather"]
    source = resolve_path(weather["source"])

    if not source.exists():
        print(f"  ERROR: combined weather CSV not found: {source}", file=sys.stderr)
        return False

    site_col = weather["site_column"]
    tmax_col = weather.get("tmax_col", "Tmax")
    tmin_col = weather.get("tmin_col", "Tmin")
    date_fmt = weather.get("date_format", "calendar")

    wx = pd.read_csv(source)
    print(f"  Loaded {len(wx)} weather rows from {source}")

    if site_col not in wx.columns:
        print(f"  ERROR: site_column '{site_col}' not found. Available: {list(wx.columns)}", file=sys.stderr)
        return False

    sites = wx[site_col].unique()
    print(f"  Sites found: {list(sites)}")

    for site in sorted(sites):
        site_dir = WEATHER_DIR / site
        site_file = site_dir / "weather.csv"

        if site_file.exists() and not force:
            print(f"  {site}/weather.csv already exists (skip)")
            continue

        site_dir.mkdir(parents=True, exist_ok=True)
        sub = wx[wx[site_col] == site].copy()

        # Convert dates
        if date_fmt == "year_doy":
            year_col = weather.get("year_col", "YEAR")
            doy_col = weather.get("doy_col", "DOY")
            sub["Period"] = sub.apply(
                lambda r: year_doy_to_date(r[year_col], r[doy_col]), axis=1
            )
        elif "Period" not in sub.columns:
            # Try to find a date-like column
            sub["Period"] = pd.NaT

        # Rename temp columns to pipeline standard
        col_renames = {}
        if tmax_col != "tmx" and tmax_col in sub.columns:
            col_renames[tmax_col] = "tmx"
        if tmin_col != "tmn" and tmin_col in sub.columns:
            col_renames[tmin_col] = "tmn"
        if col_renames:
            sub = sub.rename(columns=col_renames)

        # Build output with required columns
        out_cols = ["Period"]
        for c in ["relHum", "srad", "Srad"]:
            if c in sub.columns:
                if c == "Srad":
                    sub = sub.rename(columns={"Srad": "srad"})
                out_cols.append("srad")
                break

        # Add year/doy if available
        for c in ["year", "YEAR"]:
            if c in sub.columns:
                if c != "year":
                    sub = sub.rename(columns={c: "year"})
                out_cols.append("year")
                break
        for c in ["doy", "DOY"]:
            if c in sub.columns:
                if c != "doy":
                    sub = sub.rename(columns={c: "doy"})
                out_cols.append("doy")
                break

        out_cols.extend(["tmx", "tmn"])

        # Only keep columns that exist
        out_cols = [c for c in out_cols if c in sub.columns]
        sub[out_cols].to_csv(site_file, index=False)
        print(f"  Wrote {len(sub)} rows to {site}/weather.csv")

    return True


# ── Genomic preparation ─────────────────────────────────────────────────────

def prepare_genomic_data(config: dict, force: bool = False) -> bool:
    """Prepare genomic data based on config tier. Returns True on success."""
    crop = config["crop"]
    genomics = config["genomics"]
    tier = genomics["tier"]

    evd_out = get_evd_path(crop)
    gmat_out = get_gmatrix_path(crop)

    if tier == "evd":
        return _prepare_evd_tier(genomics, crop, evd_out, gmat_out, force)
    elif tier == "gmatrix":
        return _prepare_gmatrix_tier(genomics, crop, evd_out, force)
    elif tier == "markers":
        return _prepare_markers_tier(genomics, crop, evd_out, force)
    else:
        print(f"  ERROR: Unknown genomic tier '{tier}'", file=sys.stderr)
        return False


def _prepare_evd_tier(genomics: dict, crop: str, evd_out: Path, gmat_out: Path, force: bool) -> bool:
    """Tier 3: EVD already exists — validate/copy."""
    evd_src = resolve_path(genomics["evd_path"])
    if not evd_src.exists():
        print(f"  ERROR: EVD not found: {evd_src}", file=sys.stderr)
        return False

    # Copy to crop gmatrix dir if not already there
    if evd_src.resolve() != evd_out.resolve():
        if evd_out.exists() and not force:
            print(f"  EVD.rda already in place (use --force to overwrite)")
        else:
            evd_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(evd_src), str(evd_out))
            print(f"  Copied EVD: {evd_src} -> {evd_out}")
    else:
        print(f"  EVD.rda already in place: {evd_out}")

    # Copy Gmatrix.rda if specified
    gmat_src_path = genomics.get("gmatrix_path")
    if gmat_src_path:
        gmat_src = resolve_path(gmat_src_path)
        if gmat_src.exists() and gmat_src.resolve() != gmat_out.resolve():
            if not gmat_out.exists() or force:
                shutil.copy(str(gmat_src), str(gmat_out))
                print(f"  Copied Gmatrix: {gmat_src} -> {gmat_out}")

    return True


def _prepare_gmatrix_tier(genomics: dict, crop: str, evd_out: Path, force: bool) -> bool:
    """Tier 2: G-matrix → EVD."""
    source = resolve_path(genomics["source"])
    if not source.exists():
        print(f"  ERROR: G-matrix source not found: {source}", file=sys.stderr)
        return False

    if evd_out.exists() and not force:
        print(f"  EVD.rda already exists (use --force to overwrite)")
        return True

    print(f"  Computing EVD from G-matrix: {source}")
    return generate_evd_from_gmatrix(
        gmatrix_path=str(source),
        output_path=str(evd_out),
        save_gmatrix=True,
    )


def _prepare_markers_tier(genomics: dict, crop: str, evd_out: Path, force: bool) -> bool:
    """Tier 1: Markers → G-matrix → EVD."""
    source = resolve_path(genomics["source"])
    if not source.exists():
        print(f"  ERROR: Marker source not found: {source}", file=sys.stderr)
        return False

    if evd_out.exists() and not force:
        print(f"  EVD.rda already exists (use --force to overwrite)")
        return True

    # Load marker data
    genotype_ids, marker_matrix = _load_markers(genomics, source)
    if marker_matrix is None:
        return False

    print(f"  Marker matrix: {marker_matrix.shape[0]} genotypes x {marker_matrix.shape[1]} markers")

    # Compute G-matrix
    coding = genomics.get("marker_coding", "biallelic_12")
    G = compute_vanraden_gmatrix(marker_matrix, coding=coding)
    print(f"  G-matrix computed: {G.shape[0]} x {G.shape[1]}")

    # Save G-matrix as CSV for R to read
    gmatrix_dir = get_crop_gmatrix_dir(crop)
    gmatrix_dir.mkdir(parents=True, exist_ok=True)
    csv_path = gmatrix_dir / "_Gmatrix_temp.csv"
    np.savetxt(str(csv_path), G, delimiter=",")

    # Generate EVD via R
    success = generate_evd_from_csv(
        csv_path=str(csv_path),
        genotype_ids=genotype_ids,
        output_path=str(evd_out),
        save_gmatrix=True,
    )

    # Clean up temp CSV
    csv_path.unlink(missing_ok=True)

    return success


def _load_markers(genomics: dict, source: Path) -> tuple[list[str], np.ndarray | None]:
    """Load marker matrix from source file. Returns (genotype_ids, matrix) or ([], None)."""
    fmt = genomics.get("format", "csv")
    missing_val = genomics.get("missing_value", "-")
    metadata_rows = genomics.get("metadata_rows", 0)
    genotype_col = genomics.get("genotype_col", "MkID")

    if fmt == "excel":
        sheet = genomics.get("sheet", "Genotypes")
        df = pd.read_excel(str(source), sheet_name=sheet)
    else:
        # Auto-detect delimiter: check first non-metadata line for tabs
        sep = genomics.get("delimiter", None)
        if sep is None:
            with open(str(source)) as f:
                for _ in range(metadata_rows + 1):
                    line = f.readline()
                sep = "\t" if "\t" in line else ","
        df = pd.read_csv(str(source), sep=sep, skiprows=metadata_rows)
        metadata_rows = 0  # already handled by skiprows

    print(f"  Loaded marker data: {df.shape[0]} rows x {df.shape[1]} cols")

    # Skip metadata rows (e.g., Position, Chromosome, Chr Label — for Excel format)
    if metadata_rows > 0:
        df = df.iloc[metadata_rows:].reset_index(drop=True)
        print(f"  Skipped {metadata_rows} metadata rows -> {len(df)} genotypes")

    # Extract genotype IDs
    if genotype_col in df.columns:
        genotype_ids = df[genotype_col].astype(str).tolist()
        marker_df = df.drop(columns=[genotype_col])
    else:
        genotype_ids = df.iloc[:, 0].astype(str).tolist()
        marker_df = df.iloc[:, 1:]

    # Convert to numeric, handling missing values
    marker_df = marker_df.replace(missing_val, np.nan)
    marker_matrix = marker_df.apply(pd.to_numeric, errors="coerce").values.astype(float)

    n_missing = np.isnan(marker_matrix).sum()
    total = marker_matrix.size
    if n_missing > 0:
        print(f"  Missing values: {n_missing}/{total} ({n_missing/total*100:.1f}%)")

    return genotype_ids, marker_matrix


def compute_vanraden_gmatrix(M: np.ndarray, coding: str = "biallelic_12") -> np.ndarray:
    """Compute VanRaden Method 1 G-matrix from a marker matrix.

    M: (n_genotypes x n_markers) array.
    coding: 'biallelic_12' (values 1/2), 'biallelic_01' (values 0/1), 'dosage' (values 0/1/2).
    """
    M = M.copy().astype(float)

    if coding == "biallelic_12":
        M = M - 1  # convert {1,2} to {0,1} dosage-like

    # Impute missing with column mean
    col_means = np.nanmean(M, axis=0)
    for j in range(M.shape[1]):
        mask = np.isnan(M[:, j])
        if mask.any():
            M[mask, j] = col_means[j]

    # Allele frequencies
    p = np.mean(M, axis=0) / 2.0

    # Remove monomorphic markers (p=0 or p=1)
    poly = (p > 0.001) & (p < 0.999)
    if poly.sum() < M.shape[1]:
        n_removed = M.shape[1] - poly.sum()
        print(f"  Removed {n_removed} monomorphic markers")
        M = M[:, poly]
        p = p[poly]

    # Center
    Z = M - 2 * p

    # Scale
    denom = 2.0 * np.sum(p * (1 - p))
    G = Z @ Z.T / denom

    return G


# ── Validation ───────────────────────────────────────────────────────────────

def validate_pipeline(config: dict) -> list[str]:
    """Check all expected output files are in place. Returns list of issues."""
    crop = config["crop"]
    issues = []

    pheno = get_crop_data_dir(crop) / "phenotypes_dated.csv"
    if not pheno.exists():
        issues.append(f"Missing: {pheno}")

    evd = get_evd_path(crop)
    if not evd.exists():
        issues.append(f"Missing: {evd}")

    # Check weather
    weather = config["weather"]
    fmt = weather["format"]
    if fmt == "single_location":
        source = resolve_path(weather["source"])
        wx = source / "weather.csv" if source.is_dir() else source
        if not wx.exists():
            issues.append(f"Missing weather: {wx}")
    elif fmt == "multi_location_dir":
        source = resolve_path(weather["source"])
        if not source.is_dir():
            issues.append(f"Missing weather directory: {source}")
    elif fmt == "combined_csv":
        # Check that per-location files were created
        wx_csv = resolve_path(weather["source"])
        if wx_csv.exists():
            df = pd.read_csv(wx_csv)
            sites = df[weather["site_column"]].unique()
            for site in sites:
                wx_file = WEATHER_DIR / site / "weather.csv"
                if not wx_file.exists():
                    issues.append(f"Missing: {wx_file}")

    return issues


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Config-driven data preparation for the CGM-WGP pipeline."
    )
    p.add_argument("--config", required=False, help="Path to crop YAML config file")
    p.add_argument("--validate-only", action="store_true",
                   help="Only check if pipeline data is in place, don't prepare anything")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing files")
    p.add_argument("--inspect", action="store_true",
                   help="Inspect raw data files before preparing (or standalone with --crop)")
    p.add_argument("--crop", help="Crop name for --inspect without --config")
    p.add_argument("--generate-config", action="store_true",
                   help="Generate a draft YAML config (used with --inspect)")
    args = p.parse_args()

    # Inspect mode
    if args.inspect:
        from cgm_wgp.inspect_data import inspect_and_report
        crop = args.crop
        if not crop and args.config:
            cfg = load_crop_config(config_path=args.config)
            crop = cfg.get("crop")
        if not crop:
            print("ERROR: --inspect requires --crop or --config", file=sys.stderr)
            sys.exit(1)
        inspect_and_report(crop=crop, generate_config=args.generate_config)
        if not args.config:
            return  # inspect-only mode, no preparation
        print()  # blank line before proceeding to preparation

    if not args.config:
        p.error("--config is required (unless using --inspect --crop)")
        return

    # Load config
    config = load_crop_config(config_path=args.config)
    crop = config.get("crop", "unknown")
    print(f"\n{'='*60}")
    print(f"CGM-WGP Pipeline Preparation: {crop}")
    print(f"{'='*60}\n")

    # Validate config schema
    errors = validate_config(config, project_root=PROJECT_ROOT)
    if errors:
        print("Config validation errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("Config validation: OK\n")

    # Validate-only mode
    if args.validate_only:
        print("Validation mode: checking pipeline data...\n")
        issues = validate_pipeline(config)
        if issues:
            print("Issues found:")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)
        else:
            print("All pipeline data in place!")
        return

    # Full preparation
    print("Step 1: Creating directories")
    prepare_directories(crop)
    print()

    print("Step 2: Preparing phenotype data")
    pheno_path = prepare_phenotype_data(config, force=args.force)
    if pheno_path is None:
        print("  FAILED: phenotype preparation", file=sys.stderr)
        sys.exit(1)
    print()

    print("Step 3: Preparing weather data")
    if not prepare_weather_data(config, force=args.force):
        print("  FAILED: weather preparation", file=sys.stderr)
        sys.exit(1)
    print()

    print("Step 4: Preparing genomic data")
    if not prepare_genomic_data(config, force=args.force):
        print("  FAILED: genomic preparation", file=sys.stderr)
        sys.exit(1)
    print()

    # Final validation
    print("Step 5: Validating pipeline data")
    issues = validate_pipeline(config)
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("  All pipeline data in place!")

    print(f"\n{'='*60}")
    print(f"Pipeline preparation complete for {crop}!")
    print(f"{'='*60}")

    # Print run command hint
    model = config.get("model", {})
    weather = config["weather"]
    weather_arg = str(resolve_path(weather["source"]))
    if weather["format"] == "combined_csv":
        weather_arg = str(WEATHER_DIR)

    print(f"\nTo run the pipeline:")
    cmd_parts = [
        f"python scripts/app.py",
        f"--config {args.config}",
        f"--weather {weather_arg}",
    ]
    print(f"  {' '.join(cmd_parts)}")


if __name__ == "__main__":
    main()
