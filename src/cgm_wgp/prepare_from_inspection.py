"""
prepare_from_inspection.py

Auto-extract and prepare raw data files into pipeline-ready formats.
Called by inspect_data.py when --prepare is passed.

Only generates files that don't already exist — never overwrites raw sources.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cgm_wgp.config import DATA_DIR, WEATHER_DIR, CONFIGS_DIR, PROJECT_ROOT

# ── Constants ────────────────────────────────────────────────────────────────

CROP_CARDINAL_TEMPS = {
    "beans":    {"Tb": 10.0, "Topt": 25.0, "Tc": 38.0},
    "broccoli": {"Tb": 5.0, "Topt": 20.0, "Tc": 35.0},
}

# Crops known to be short-day sensitive
SHORT_DAY_CROPS = {"beans"}
LONG_DAY_CROPS = {"broccoli"}

DAYLENGTH_KEYWORDS = {"dl", "daylength", "daylhr", "daylight_hours", "photoperiod",
                       "daylight", "day_length"}

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_POWER_PARAMS = "T2M_MAX,T2M_MIN,ALLSKY_SFC_SW_DWN"


# ── Result Dataclass ─────────────────────────────────────────────────────────

@dataclass
class PreparationResult:
    crop: str
    phenotype_path: Path | None = None
    phenotype_stats: dict = field(default_factory=dict)
    weather_created: list[str] = field(default_factory=list)
    weather_skipped: list[str] = field(default_factory=list)
    weather_fetched: list[str] = field(default_factory=list)
    weather_failed: list[str] = field(default_factory=list)
    temp_unit: str = "celsius"
    config_path: Path | None = None
    warnings: list[str] = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _normalize_col(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_").replace(".", "_").strip("_")


def _detect_temp_unit(tmax_values: pd.Series) -> str:
    """Fahrenheit if median max temp > 50."""
    median = tmax_values.dropna().median()
    return "fahrenheit" if median > 50 else "celsius"


def _f_to_c(f: float) -> float:
    return round((f - 32.0) * 5.0 / 9.0, 2)


def _find_daylength_column(headers: list[str]) -> str | None:
    for h in headers:
        if _normalize_col(h) in DAYLENGTH_KEYWORDS:
            return h
    return None


def _pick_ft_column(column_hints: list) -> str | None:
    """Pick best FT column — prefer DAP over GDD."""
    ft_hints = [h for h in column_hints if h.role == "ft"]
    if not ft_hints:
        return None
    # Prefer column with "dap" or "days" in name
    for h in ft_hints:
        norm = _normalize_col(h.name)
        if "dap" in norm or "days" in norm:
            return h.name
    return ft_hints[0].name


def _col_by_role(column_hints: list, role: str) -> str | None:
    for h in column_hints:
        if h.role == role:
            return h.name
    return None


def _compute_daylength(doy: int, latitude: float) -> float:
    """Spencer (1971) daylength computation."""
    lat_r = latitude * math.pi / 180.0
    theta = 0.2163108 + 2.0 * math.atan(0.9671396 * math.tan(0.00860 * (doy - 186)))
    phi = math.asin(0.39795 * math.cos(theta))
    arg = (math.sin(0.8333 * math.pi / 180.0) + math.sin(lat_r) * math.sin(phi)) / (
        math.cos(lat_r) * math.cos(phi)
    )
    arg = max(-1.0, min(1.0, arg))
    ha = math.pi - math.acos(arg)
    return (2.0 * ha / math.pi) * 12.0


# ── Phenotype Preparation ───────────────────────────────────────────────────

def _prepare_phenotypes(report, crop: str, metadata: dict | None,
                        result: PreparationResult) -> pd.DataFrame | None:
    """Extract phenotype data into phenotypes_dated.csv. Returns DataFrame of prepared data."""
    out_path = DATA_DIR / crop / "phenotypes_dated.csv"
    if out_path.exists():
        print(f"\n  [SKIP] {out_path.relative_to(PROJECT_ROOT)} already exists")
        # Still load and return it for downstream use
        return pd.read_csv(out_path)

    # Find phenotype source from any file or sheet
    pheno_source = None
    for fi in report.files:
        # Check sheets first (Excel)
        for sheet in fi.sheets:
            if sheet.role == "phenotype" and sheet.role_confidence in ("high", "medium"):
                ft_col = _pick_ft_column(sheet.column_hints)
                geno_col = _col_by_role(sheet.column_hints, "genotype")
                env_col = _col_by_role(sheet.column_hints, "location")
                if ft_col and geno_col and env_col:
                    pheno_source = {
                        "type": "excel_sheet", "file": fi.path,
                        "sheet": sheet.name, "header_idx": sheet.header_idx,
                        "geno_col": geno_col, "env_col": env_col,
                        "ft_col": ft_col,
                        "rep_col": _col_by_role(sheet.column_hints, "rep"),
                    }
                    break
        if pheno_source:
            break

        # Check text files
        if fi.role == "phenotype" and fi.role_confidence in ("high", "medium"):
            ft_col = _pick_ft_column(fi.column_hints)
            geno_col = _col_by_role(fi.column_hints, "genotype")
            env_col = _col_by_role(fi.column_hints, "location")
            if ft_col and geno_col and env_col:
                pheno_source = {
                    "type": "text_file", "file": fi.path,
                    "delimiter": fi.delimiter,
                    "geno_col": geno_col, "env_col": env_col,
                    "ft_col": ft_col,
                    "rep_col": _col_by_role(fi.column_hints, "rep"),
                }
                break

    if not pheno_source:
        result.warnings.append("No phenotype source found with genotype + env + FT columns")
        print("\n  [WARN] No suitable phenotype source found")
        return None

    print(f"\n  Phenotype source: {Path(pheno_source['file']).name}", end="")
    if pheno_source["type"] == "excel_sheet":
        print(f" (sheet: {pheno_source['sheet']})")
    else:
        print()

    # Load data
    if pheno_source["type"] == "excel_sheet":
        df = pd.read_excel(
            str(pheno_source["file"]),
            sheet_name=pheno_source["sheet"],
            header=pheno_source["header_idx"],
        )
    else:
        sep = pheno_source["delimiter"] or r"\s+"
        df = pd.read_csv(str(pheno_source["file"]), sep=sep, engine="python")

    geno_col = pheno_source["geno_col"]
    env_col = pheno_source["env_col"]
    ft_col = pheno_source["ft_col"]
    rep_col = pheno_source["rep_col"]

    # Keep only needed columns
    keep = [geno_col, env_col, ft_col]
    if rep_col and rep_col in df.columns:
        keep.append(rep_col)
    df = df[[c for c in keep if c in df.columns]].copy()

    # Convert FT to numeric, drop missing
    df[ft_col] = pd.to_numeric(df[ft_col], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=[ft_col])
    if n_before - len(df) > 0:
        print(f"  Dropped {n_before - len(df)} rows with missing FT")

    # Average reps
    if rep_col and rep_col in df.columns:
        n_before = len(df)
        df = df.groupby([geno_col, env_col], as_index=False).agg({ft_col: "mean"})
        df[ft_col] = df[ft_col].round().astype(int)
        print(f"  Averaged reps: {n_before} -> {len(df)} rows")
    else:
        df[ft_col] = df[ft_col].round().astype(int)

    # Rename to pipeline columns
    df = df.rename(columns={geno_col: "id", env_col: "Planting", ft_col: "ft"})

    # Map planting dates from metadata
    if metadata and metadata.get("dates"):
        df["Start_Date"] = df["Planting"].map(metadata["dates"])
        missing = df[df["Start_Date"].isna()]["Planting"].unique()
        if len(missing) > 0:
            result.warnings.append(
                f"No planting dates for {len(missing)} envs: {list(missing)[:5]}")
            print(f"  [WARN] No planting dates for {len(missing)} envs: {list(missing)[:5]}")
            df = df.dropna(subset=["Start_Date"])
        # Calculate End_Date
        df["Start_Date"] = pd.to_datetime(df["Start_Date"])
        df["End_Date"] = df["Start_Date"] + pd.to_timedelta(df["ft"], unit="D")
        df["Start_Date"] = df["Start_Date"].dt.strftime("%Y-%m-%d")
        df["End_Date"] = df["End_Date"].dt.strftime("%Y-%m-%d")
    else:
        result.warnings.append("No metadata sheet with planting dates found")
        print("  [WARN] No planting dates available — Start_Date/End_Date will be empty")
        df["Start_Date"] = ""
        df["End_Date"] = ""

    df["censored"] = 0

    # Sort for reproducibility
    df = df.sort_values(["Planting", "id"]).reset_index(drop=True)

    # Write
    out_cols = ["id", "Planting", "ft", "Start_Date", "End_Date", "censored"]
    df[out_cols].to_csv(out_path, index=False)
    print(f"  Written: {out_path.relative_to(PROJECT_ROOT)}")
    print(f"  {df['id'].nunique()} genotypes, {df['Planting'].nunique()} environments, "
          f"FT range {df['ft'].min()}-{df['ft'].max()} days")

    result.phenotype_path = out_path
    result.phenotype_stats = {
        "n_genotypes": df["id"].nunique(),
        "n_envs": df["Planting"].nunique(),
        "ft_min": int(df["ft"].min()),
        "ft_max": int(df["ft"].max()),
        "n_obs": len(df),
        "envs": sorted(df["Planting"].unique().tolist()),
    }
    return df


# ── Weather Preparation ─────────────────────────────────────────────────────

def _load_weather_from_source(report) -> tuple[pd.DataFrame | None, dict]:
    """Find and load embedded weather data from inspection results.
    Returns (DataFrame, info_dict) or (None, {})."""
    for fi in report.files:
        for sheet in fi.sheets:
            if sheet.role == "weather":
                date_col = _col_by_role(sheet.column_hints, "date")
                tmax_col = _col_by_role(sheet.column_hints, "tmax")
                tmin_col = _col_by_role(sheet.column_hints, "tmin")
                env_col = _col_by_role(sheet.column_hints, "location")
                if date_col and tmax_col and tmin_col and env_col:
                    df = pd.read_excel(
                        str(fi.path),
                        sheet_name=sheet.name,
                        header=sheet.header_idx,
                    )
                    dl_col = _find_daylength_column(sheet.headers)
                    return df, {
                        "date_col": date_col, "tmax_col": tmax_col,
                        "tmin_col": tmin_col, "env_col": env_col,
                        "dl_col": dl_col, "source": f"{fi.path.name} ({sheet.name})",
                    }

        # Check text files with weather role
        if fi.role == "weather" and fi.role_confidence in ("high", "medium"):
            date_col = _col_by_role(fi.column_hints, "date")
            tmax_col = _col_by_role(fi.column_hints, "tmax")
            tmin_col = _col_by_role(fi.column_hints, "tmin")
            env_col = _col_by_role(fi.column_hints, "location")
            if date_col and tmax_col and tmin_col:
                sep = fi.delimiter or ","
                df = pd.read_csv(str(fi.path), sep=sep)
                dl_col = _find_daylength_column(fi.headers)
                return df, {
                    "date_col": date_col, "tmax_col": tmax_col,
                    "tmin_col": tmin_col, "env_col": env_col,
                    "dl_col": dl_col, "source": fi.path.name,
                }

    return None, {}


def _write_weather_csv(env_df: pd.DataFrame, env_code: str,
                       date_col: str, tmax_col: str, tmin_col: str,
                       dl_col: str | None, is_fahrenheit: bool) -> Path:
    """Write a single environment's weather CSV."""
    out_dir = WEATHER_DIR / env_code
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "weather.csv"

    out = pd.DataFrame()
    out["Period"] = pd.to_datetime(env_df[date_col]).dt.strftime("%Y-%m-%d")

    if is_fahrenheit:
        out["tmx"] = env_df[tmax_col].apply(_f_to_c)
        out["tmn"] = env_df[tmin_col].apply(_f_to_c)
    else:
        out["tmx"] = env_df[tmax_col].round(2)
        out["tmn"] = env_df[tmin_col].round(2)

    if dl_col and dl_col in env_df.columns:
        out["DAYLhr"] = env_df[dl_col].round(3)

    out["year"] = pd.to_datetime(env_df[date_col]).dt.year
    out["doy"] = pd.to_datetime(env_df[date_col]).dt.day_of_year

    out.to_csv(out_path, index=False)
    return out_path


def _fetch_nasa_power(lat: float, lon: float,
                      start_date: str, end_date: str) -> pd.DataFrame | None:
    """Fetch daily weather from NASA POWER API. Returns DataFrame or None on failure."""
    import urllib.request
    import json as _json

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    url = (
        f"{NASA_POWER_URL}"
        f"?parameters={NASA_POWER_PARAMS}"
        f"&community=ag"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={start_dt.strftime('%Y%m%d')}"
        f"&end={end_dt.strftime('%Y%m%d')}"
        f"&format=JSON"
    )

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read().decode())
    except Exception as e:
        print(f"    NASA POWER API error: {e}")
        return None

    params = data.get("properties", {}).get("parameter", {})
    tmax_data = params.get("T2M_MAX", {})
    tmin_data = params.get("T2M_MIN", {})
    srad_data = params.get("ALLSKY_SFC_SW_DWN", {})

    if not tmax_data:
        print("    NASA POWER returned no data")
        return None

    rows = []
    for date_str in sorted(tmax_data.keys()):
        dt = datetime.strptime(date_str, "%Y%m%d")
        tmax = tmax_data.get(date_str)
        tmin = tmin_data.get(date_str)
        srad = srad_data.get(date_str)

        # NASA POWER uses -999 for missing
        if tmax is not None and tmax < -900:
            tmax = None
        if tmin is not None and tmin < -900:
            tmin = None
        if srad is not None and srad < -900:
            srad = None

        doy = dt.timetuple().tm_yday
        dl = _compute_daylength(doy, lat)

        rows.append({
            "Period": dt.strftime("%Y-%m-%d"),
            "tmx": round(tmax, 2) if tmax is not None else None,
            "tmn": round(tmin, 2) if tmin is not None else None,
            "srad": round(srad, 2) if srad is not None else None,
            "DAYLhr": round(dl, 3),
            "year": dt.year,
            "doy": doy,
        })

    return pd.DataFrame(rows)


def _prepare_weather(report, crop: str, pheno_envs: list[str],
                     metadata: dict | None, result: PreparationResult):
    """Prepare weather data for all phenotype environments."""
    # Determine which envs need weather
    envs_needed = []
    envs_existing = []
    for env in pheno_envs:
        wpath = WEATHER_DIR / env / "weather.csv"
        if wpath.exists():
            envs_existing.append(env)
        else:
            envs_needed.append(env)

    if envs_existing:
        print(f"\n  [SKIP] Weather already exists for {len(envs_existing)} envs: "
              f"{envs_existing[:5]}{'...' if len(envs_existing) > 5 else ''}")
        result.weather_skipped = envs_existing

    if not envs_needed:
        print("  All weather files present")
        return

    # Try embedded weather data first
    wx_df, wx_info = _load_weather_from_source(report)

    if wx_df is not None:
        env_col = wx_info["env_col"]
        date_col = wx_info["date_col"]
        tmax_col = wx_info["tmax_col"]
        tmin_col = wx_info["tmin_col"]
        dl_col = wx_info.get("dl_col")

        # Detect temperature units
        temp_unit = _detect_temp_unit(wx_df[tmax_col])
        is_fahrenheit = temp_unit == "fahrenheit"
        result.temp_unit = temp_unit
        if is_fahrenheit:
            print(f"\n  Temperature unit: Fahrenheit (auto-detected) -> converting to Celsius")
        else:
            print(f"\n  Temperature unit: Celsius")
        print(f"  Weather source: {wx_info['source']}")

        # Extract per-environment
        available_envs = set(wx_df[env_col].dropna().unique())
        for env in envs_needed:
            if env in available_envs:
                env_data = wx_df[wx_df[env_col] == env].copy()
                _write_weather_csv(env_data, env, date_col, tmax_col, tmin_col,
                                   dl_col, is_fahrenheit)
                result.weather_created.append(env)
                print(f"    Created: weather/{env}/weather.csv ({len(env_data)} days)")
            else:
                # Env not in embedded weather — try NASA POWER
                _fetch_env_weather_nasa(env, metadata, result)
    else:
        # No embedded weather — try NASA POWER for all
        print(f"\n  No embedded weather data found — trying NASA POWER API")
        for env in envs_needed:
            _fetch_env_weather_nasa(env, metadata, result)


def _fetch_env_weather_nasa(env: str, metadata: dict | None,
                            result: PreparationResult):
    """Fetch weather for a single environment from NASA POWER."""
    if not metadata or not metadata.get("coordinates") or env not in metadata["coordinates"]:
        result.weather_failed.append(env)
        result.warnings.append(f"No coordinates for {env} — cannot fetch weather")
        print(f"    [FAIL] {env}: no coordinates available")
        return

    if not metadata.get("dates") or env not in metadata["dates"]:
        result.weather_failed.append(env)
        result.warnings.append(f"No planting date for {env} — cannot determine date range")
        print(f"    [FAIL] {env}: no planting date for date range")
        return

    lat, lon = metadata["coordinates"][env]
    start_date = metadata["dates"][env]
    # Buffer: planting date to +200 days
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = start_dt + timedelta(days=200)
    end_date = end_dt.strftime("%Y-%m-%d")

    print(f"    Fetching {env} ({lat:.2f}, {lon:.2f}) {start_date} to {end_date}...", end="")
    sys.stdout.flush()

    wx = _fetch_nasa_power(lat, lon, start_date, end_date)
    if wx is not None and not wx.empty:
        out_dir = WEATHER_DIR / env
        out_dir.mkdir(parents=True, exist_ok=True)
        wx.to_csv(out_dir / "weather.csv", index=False)
        result.weather_fetched.append(env)
        print(f" OK ({len(wx)} days)")
    else:
        result.weather_failed.append(env)
        print(f" FAILED")

    # Rate limit
    time.sleep(1)


# ── Config Generation ────────────────────────────────────────────────────────

def _generate_complete_config(report, crop: str, metadata: dict | None,
                              pheno_stats: dict, cardinal_temps: dict | None,
                              config_output: str | None,
                              result: PreparationResult) -> Path | None:
    """Generate a complete config YAML with no TODOs."""
    if config_output:
        out_path = Path(config_output)
    else:
        out_path = CONFIGS_DIR / f"{crop.lower()}.yaml"

    if out_path.exists():
        print(f"\n  [SKIP] {out_path.relative_to(PROJECT_ROOT)} already exists")
        return out_path

    # Determine cardinal temps
    crop_lower = crop.lower()
    ct = cardinal_temps or CROP_CARDINAL_TEMPS.get(crop_lower, {"Tb": 10.0, "Topt": 30.0, "Tc": 40.0})

    # Find marker file
    marker_file = None
    for fi in report.files:
        if fi.role == "markers":
            marker_file = fi
            break

    # Determine photoperiod
    photo_type = "short_day" if crop_lower in SHORT_DAY_CROPS else (
        "long_day" if crop_lower in LONG_DAY_CROPS else None)

    # Check latitude range for multi-latitude photoperiod fitting
    lat_range = 0.0
    if metadata and metadata.get("coordinates"):
        lats = [c[0] for c in metadata["coordinates"].values()]
        if lats:
            lat_range = max(lats) - min(lats)

    lines = []
    def add(text: str = ""):
        lines.append(text)

    add(f"crop: {crop}")
    add()

    # Phenotypes
    add("phenotypes:")
    pheno_path = DATA_DIR / crop / "phenotypes_dated.csv"
    rel_pheno = pheno_path.relative_to(PROJECT_ROOT)
    add(f"  source: {rel_pheno}")
    add("  format: pipeline_ready")
    add()

    # Weather
    add("weather:")
    add(f"  source: {WEATHER_DIR.relative_to(PROJECT_ROOT)}")
    add("  format: multi_location_dir")
    add()

    # Genomics
    add("genomics:")
    if marker_file:
        rel_marker = marker_file.path.relative_to(PROJECT_ROOT)
        add("  tier: markers")
        add(f'  source: "{rel_marker}"')
        add("  format: csv")
        if marker_file.metadata_rows_detected:
            add(f"  metadata_rows: {marker_file.metadata_rows_detected}")
        if marker_file.marker_coding:
            add(f"  marker_coding: {marker_file.marker_coding}")
        else:
            add("  marker_coding: dosage")
    else:
        add("  tier: markers")
        add('  source: "TODO"')
    add()

    # Locations
    if metadata and metadata.get("coordinates"):
        add("locations:")
        for env in sorted(metadata["coordinates"].keys()):
            lat, lon = metadata["coordinates"][env]
            add(f"  {env}:")
            add(f"    latitude: {lat}")
            add(f"    longitude: {lon}")
        add()

    # Planting dates (if not already in phenotypes_dated.csv via Start_Date)
    # Include for reference/override
    if metadata and metadata.get("dates"):
        add("# Planting dates (also stored in phenotypes_dated.csv Start_Date column)")
        add("# planting_dates:")
        for env in sorted(metadata["dates"].keys()):
            add(f'#   "{env}": "{metadata["dates"][env]}"')
        add()

    # Model
    add("model:")
    add(f"  Tb: {ct['Tb']}")
    add(f"  Topt: {ct['Topt']}")
    add(f"  Tc: {ct['Tc']}")
    add("  abc:")
    add("    n_proposals: 10000")
    add("    accept_quantile: 0.01")
    add("    max_posterior_samples: 200")

    if photo_type and lat_range > 3.0:
        add("  photoperiod:")
        add(f"    type: {photo_type}")
        add("    fit: true")
        add("    B: 3.0")
        add("    Pc: 13.0")
        add("    B_bounds: [1.0, 20.0]")
        add("    outer_maxiter: 30")
        add("    inner_maxiter: 500")
    elif photo_type:
        add(f"  # photoperiod: disabled (latitude range {lat_range:.1f}° < 3° — single location confounding)")
        add(f"  # type: {photo_type}")
    add()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"\n  Config written: {out_path.relative_to(PROJECT_ROOT)}")
    result.config_path = out_path
    return out_path


# ── Summary ──────────────────────────────────────────────────────────────────

def _print_summary(result: PreparationResult):
    print()
    print("=" * 60)
    print(f"PREPARATION SUMMARY: {result.crop}")
    print("=" * 60)

    if result.phenotype_path:
        ps = result.phenotype_stats
        print(f"\n  PHENOTYPES")
        print(f"    {result.phenotype_path.relative_to(PROJECT_ROOT)}")
        print(f"    {ps.get('n_genotypes', '?')} genotypes, "
              f"{ps.get('n_envs', '?')} environments, "
              f"FT {ps.get('ft_min', '?')}-{ps.get('ft_max', '?')} days "
              f"({ps.get('n_obs', '?')} observations)")

    total_wx = len(result.weather_created) + len(result.weather_fetched)
    if total_wx > 0 or result.weather_skipped:
        print(f"\n  WEATHER")
        if result.weather_created:
            print(f"    Extracted from data: {len(result.weather_created)} environments")
            if result.temp_unit == "fahrenheit":
                print(f"    Temperature: Fahrenheit -> Celsius (auto-converted)")
        if result.weather_fetched:
            print(f"    Fetched from NASA POWER: {len(result.weather_fetched)} environments")
        if result.weather_skipped:
            print(f"    Already existed: {len(result.weather_skipped)} environments")

    if result.config_path:
        print(f"\n  CONFIG")
        print(f"    {result.config_path.relative_to(PROJECT_ROOT)}")

    if result.warnings:
        print(f"\n  WARNINGS")
        for w in result.warnings:
            print(f"    [!] {w}")

    if result.weather_failed:
        print(f"\n  MISSING WEATHER ({len(result.weather_failed)} envs)")
        for env in result.weather_failed:
            print(f"    - {env}")

    print(f"\n  NEXT STEPS")
    if result.config_path:
        print(f"    1. Review config:  cat {result.config_path.relative_to(PROJECT_ROOT)}")
        print(f"    2. Run pipeline:   venv/bin/python3 scripts/app.py --config {result.config_path.relative_to(PROJECT_ROOT)}")
    print("=" * 60)


# ── Main Entry Point ─────────────────────────────────────────────────────────

def prepare_from_report(report, crop: str, config_output: str | None = None,
                        cardinal_temps: dict | None = None) -> PreparationResult:
    """Main preparation entry point. Called by inspect_data.py --prepare."""
    from cgm_wgp.inspect_data import _extract_metadata_from_sheets

    print("\n" + "=" * 60)
    print(f"PREPARING: {crop}")
    print("=" * 60)

    result = PreparationResult(crop=crop)

    # Extract metadata (dates, coordinates)
    metadata = _extract_metadata_from_sheets(report.files)
    if metadata:
        n_dates = len(metadata.get("dates", {}))
        n_coords = len(metadata.get("coordinates", {}))
        print(f"\n  Metadata: {n_dates} planting dates, {n_coords} coordinates")
    else:
        print("\n  [WARN] No metadata sheet found (planting dates/coordinates)")

    # Step 1: Phenotypes
    pheno_df = _prepare_phenotypes(report, crop, metadata, result)

    # Get list of environments for weather
    if pheno_df is not None:
        pheno_envs = sorted(pheno_df["Planting"].unique().tolist())
    elif result.phenotype_stats.get("envs"):
        pheno_envs = result.phenotype_stats["envs"]
    else:
        pheno_envs = []
        if metadata and metadata.get("dates"):
            pheno_envs = sorted(metadata["dates"].keys())

    # Step 2: Weather
    if pheno_envs:
        _prepare_weather(report, crop, pheno_envs, metadata, result)
    else:
        result.warnings.append("No environments identified — skipping weather preparation")

    # Step 3: Config
    _generate_complete_config(report, crop, metadata,
                              result.phenotype_stats, cardinal_temps,
                              config_output, result)

    # Step 4: Summary
    _print_summary(result)

    return result
