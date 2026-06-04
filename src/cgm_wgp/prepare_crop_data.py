"""
prepare_crop_data.py

Convert raw crop phenotype data into the pipeline's expected phenotypes_dated.csv format.

Example (Broccoli):
  venv/bin/python3 scripts/prepare_crop_data.py \
    --input pipeline/1_data/Broccoli/phenotypes.csv \
    --crop Broccoli \
    --genotype-col Genotype \
    --location-col Planting \
    --ft-col DAP
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def normalize_name(name: str) -> str:
    """Convert location name to a filesystem-safe folder name.

    'ND Loc 1'  -> 'ND_Loc_1'
    'NE RM Set 2 Loc 1' -> 'NE_RM_Set_2_Loc_1'
    """
    return re.sub(r"[^\w]+", "_", name.strip()).strip("_")


def parse_key_value_pairs(spec: str) -> dict[str, str]:
    """Parse 'key1=val1,key2=val2' into a dict.

    Handles keys with spaces by splitting on '=' only at boundaries.
    """
    mapping = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        key, val = pair.split("=", 1)
        mapping[key.strip()] = val.strip()
    return mapping


def prepare_phenotypes(
    input_path: str,
    crop: str,
    output_path: str | None = None,
    genotype_col: str = "Genotype",
    location_col: str = "Location",
    ft_col: str = "R1",
    rep_col: str | None = None,
    planting_dates: dict[str, str] | None = None,
    season: str | None = None,
    normalize_locations: bool = False,
    location_map: dict[str, str] | None = None,
) -> Path:
    """Convert raw phenotype CSV to pipeline phenotypes_dated.csv format.

    Returns the output path.
    """
    raw = pd.read_csv(input_path)
    print(f"Loaded {len(raw)} rows from {input_path}")

    for col_name, col_val in [("genotype", genotype_col), ("location", location_col), ("ft", ft_col)]:
        if col_val not in raw.columns:
            raise ValueError(f"{col_name} column '{col_val}' not found in input. Available: {list(raw.columns)}")

    planting_date_map: dict[str, str] = dict(planting_dates) if planting_dates else {}

    location_name_map: dict[str, str] = dict(location_map) if location_map else {}

    # Build output DataFrame
    out = pd.DataFrame()
    out["id"] = raw[genotype_col].astype(str).str.strip()
    out["Planting"] = raw[location_col].astype(str).str.strip()
    out["ft"] = pd.to_numeric(raw[ft_col], errors="coerce")
    out["censored"] = 0

    # Drop rows with missing ft
    n_before = len(out)
    out = out.dropna(subset=["ft"])
    n_dropped = n_before - len(out)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows with missing ft values")

    # De-duplicate: average ft within (genotype, location, rep) groups
    if rep_col and rep_col in raw.columns:
        out["_rep"] = raw.loc[out.index, rep_col]
        n_before_dedup = len(out)
        out = out.groupby(["id", "Planting", "_rep"], as_index=False).agg(
            ft=("ft", "mean"), censored=("censored", "first")
        )
        out["ft"] = out["ft"].round().astype(int)
        out = out.drop(columns=["_rep"])
        print(f"  De-duplicated: {n_before_dedup} -> {len(out)} rows (averaged within genotype x location x rep)")
    else:
        out["ft"] = out["ft"].astype(int)

    # Assign Start_Date
    unique_locations = out["Planting"].unique()
    print(f"\n  Unique locations ({len(unique_locations)}):")

    missing_dates = []
    for loc in sorted(unique_locations):
        if loc in planting_date_map:
            print(f"    {loc}: {planting_date_map[loc]}")
        elif season:
            planting_date_map[loc] = season
            print(f"    {loc}: {season} (from season fallback)")
        else:
            missing_dates.append(loc)
            print(f"    {loc}: ** NO DATE **")

    if missing_dates:
        raise ValueError(f"No planting dates for: {missing_dates}. Provide planting_dates or season.")

    out["Start_Date"] = out["Planting"].map(planting_date_map)
    out["Start_Date"] = pd.to_datetime(out["Start_Date"])
    out["End_Date"] = out["Start_Date"] + pd.to_timedelta(out["ft"], unit="D")

    out["Start_Date"] = out["Start_Date"].dt.strftime("%Y-%m-%d")
    out["End_Date"] = out["End_Date"].dt.strftime("%Y-%m-%d")

    # Rename locations to match weather folder names
    if location_name_map:
        out["Planting"] = out["Planting"].map(
            lambda x: location_name_map.get(x, normalize_name(x) if normalize_locations else x)
        )
        print(f"\n  Location mapping applied:")
        for raw_name, folder_name in sorted(location_name_map.items()):
            print(f"    {raw_name} -> {folder_name}")
    elif normalize_locations:
        out["Planting"] = out["Planting"].apply(normalize_name)

    # Output
    if output_path:
        out_path = Path(output_path)
    else:
        from cgm_wgp.config import DATA_DIR
        out_path = DATA_DIR / crop / "phenotypes_dated.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_cols = ["id", "Planting", "ft", "Start_Date", "End_Date", "censored"]
    out[out_cols].to_csv(out_path, index=False)

    print(f"\nWrote {len(out)} rows to {out_path}")
    print(f"  Genotypes: {out['id'].nunique()}")
    print(f"  Locations:  {out['Planting'].nunique()}")
    print(f"  ft range:   {out['ft'].min()} - {out['ft'].max()} days")

    final_locations = sorted(out["Planting"].unique())
    print(f"\n  Weather folders expected in pipeline/1_data/weather/:")
    for loc in final_locations:
        print(f"    pipeline/1_data/weather/{loc}/weather.csv")

    return out_path


def prepare_phenotypes_wide(
    input_path: str,
    crop: str,
    output_path: str | None = None,
    genotype_col: str = "RIL",
    ft_columns: dict[str, str] | None = None,
    planting_dates: dict[str, str] | None = None,
    missing_value: str = "-",
    sheet: str | None = None,
) -> Path:
    """Convert wide-format phenotypes (one ft column per location) to pipeline format.

    Wide format has columns like R1-CIT, R1-ND, etc. where each is a separate
    location's flowering time. This melts to long format and writes phenotypes_dated.csv.

    Returns the output path.
    """
    if ft_columns is None:
        raise ValueError("ft_columns mapping {location: column_name} is required for wide format")
    if planting_dates is None:
        raise ValueError("planting_dates mapping {location: date} is required")

    # Load data
    if sheet:
        raw = pd.read_excel(input_path, sheet_name=sheet)
    else:
        raw = pd.read_csv(input_path)
    print(f"Loaded {len(raw)} rows from {input_path}" + (f" (sheet: {sheet})" if sheet else ""))

    if genotype_col not in raw.columns:
        raise ValueError(f"Genotype column '{genotype_col}' not found. Available: {list(raw.columns)}")

    # Melt wide to long
    rows = []
    for location, col_name in ft_columns.items():
        if col_name not in raw.columns:
            print(f"  Warning: column '{col_name}' for location '{location}' not found, skipping")
            continue
        sub = raw[[genotype_col, col_name]].copy()
        sub.columns = ["id", "ft"]
        sub["id"] = sub["id"].astype(str).str.strip()
        # Handle missing values
        sub["ft"] = sub["ft"].replace(missing_value, pd.NA)
        sub["ft"] = pd.to_numeric(sub["ft"], errors="coerce")
        sub = sub.dropna(subset=["ft"])
        sub["ft"] = sub["ft"].astype(int)
        sub["Planting"] = location
        rows.append(sub)

    if not rows:
        raise ValueError("No valid data found after melting wide columns")

    out = pd.concat(rows, ignore_index=True)
    out["censored"] = 0

    print(f"  Melted to {len(out)} rows across {len(rows)} locations")

    # Assign dates
    missing_dates = [loc for loc in ft_columns if loc not in planting_dates]
    if missing_dates:
        raise ValueError(f"No planting dates for locations: {missing_dates}")

    out["Start_Date"] = out["Planting"].map(planting_dates)
    out["Start_Date"] = pd.to_datetime(out["Start_Date"])
    out["End_Date"] = out["Start_Date"] + pd.to_timedelta(out["ft"], unit="D")

    out["Start_Date"] = out["Start_Date"].dt.strftime("%Y-%m-%d")
    out["End_Date"] = out["End_Date"].dt.strftime("%Y-%m-%d")

    # Output
    if output_path:
        out_path = Path(output_path)
    else:
        from cgm_wgp.config import DATA_DIR
        out_path = DATA_DIR / crop / "phenotypes_dated.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_cols = ["id", "Planting", "ft", "Start_Date", "End_Date", "censored"]
    out[out_cols].to_csv(out_path, index=False)

    print(f"\nWrote {len(out)} rows to {out_path}")
    print(f"  Genotypes: {out['id'].nunique()}")
    print(f"  Locations:  {out['Planting'].nunique()}")
    print(f"  ft range:   {out['ft'].min()} - {out['ft'].max()} days")

    return out_path


def main():
    p = argparse.ArgumentParser(
        description="Convert raw crop phenotype data to pipeline phenotypes_dated.csv format."
    )
    p.add_argument("--input", required=True, help="Path to raw phenotype CSV")
    p.add_argument("--crop", required=True, help="Crop name (e.g. Beans)")
    p.add_argument("--output", default=None, help="Output path. Default: pipeline/1_data/{crop}/phenotypes_dated.csv")

    # Column mapping
    p.add_argument("--genotype-col", default="Genotype", help="Column name for genotype ID (default: Genotype)")
    p.add_argument("--location-col", default="Location", help="Column name for location/environment (default: Location)")
    p.add_argument("--ft-col", default="R1", help="Column name for flowering time in days (default: R1)")
    p.add_argument("--rep-col", default=None, help="Column name for replication (optional, kept if present)")

    # Date handling
    p.add_argument(
        "--planting-dates",
        default=None,
        help="Inline mapping of location=date pairs (e.g. 'ND Loc 1=2024-05-20,KS Loc 1=2024-05-10'). "
             "Or path to a CSV with columns: location,planting_date",
    )
    p.add_argument(
        "--season",
        default=None,
        help="Default planting date for all locations (e.g. '2024-05-15'). "
             "Used as fallback when --planting-dates doesn't cover a location.",
    )

    # Location name handling
    p.add_argument(
        "--normalize-locations",
        action="store_true",
        help="Normalize location names to filesystem-safe folder names (spaces -> underscores)",
    )
    p.add_argument(
        "--location-map",
        default=None,
        help="Explicit mapping of raw location names to weather folder names. "
             "Format: 'Raw Name 1=folder1,Raw Name 2=folder2'. "
             "Overrides --normalize-locations for mapped locations.",
    )

    args = p.parse_args()

    # Parse CLI string args into dicts for the function
    planting_dates = None
    if args.planting_dates:
        path_check = Path(args.planting_dates)
        if path_check.is_file():
            date_df = pd.read_csv(path_check)
            planting_dates = {}
            for _, row in date_df.iterrows():
                planting_dates[str(row.iloc[0]).strip()] = str(row.iloc[1]).strip()
            print(f"  Loaded {len(planting_dates)} planting dates from {path_check}")
        else:
            planting_dates = parse_key_value_pairs(args.planting_dates)
            print(f"  Parsed {len(planting_dates)} planting dates from inline spec")

    loc_map = None
    if args.location_map:
        loc_map = parse_key_value_pairs(args.location_map)
        print(f"  Loaded {len(loc_map)} location name mappings")

    prepare_phenotypes(
        input_path=args.input,
        crop=args.crop,
        output_path=args.output,
        genotype_col=args.genotype_col,
        location_col=args.location_col,
        ft_col=args.ft_col,
        rep_col=args.rep_col,
        planting_dates=planting_dates,
        season=args.season,
        normalize_locations=args.normalize_locations,
        location_map=loc_map,
    )


if __name__ == "__main__":
    main()
