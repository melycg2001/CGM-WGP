"""
config_schema.py

Validation for crop YAML config files.
"""

from __future__ import annotations

from pathlib import Path

VALID_PHENO_FORMATS = {"csv", "excel", "pipeline_ready"}
VALID_PHENO_LAYOUTS = {"long", "wide"}
VALID_WEATHER_FORMATS = {"single_location", "multi_location_dir", "combined_csv"}
VALID_GENOMIC_TIERS = {"markers", "gmatrix", "evd"}
VALID_DATE_FORMATS = {"calendar", "year_doy"}
VALID_MARKER_CODINGS = {"biallelic_12", "biallelic_01", "dosage"}
VALID_GENOMIC_FORMATS = {"csv", "excel", "rds", "rda"}


def validate_config(config: dict, project_root: Path | None = None) -> list[str]:
    """Validate a crop config dict. Returns list of error messages (empty = valid)."""
    errors = []

    # Top-level
    if "crop" not in config:
        errors.append("Missing required field: 'crop'")

    # Phenotypes
    pheno = config.get("phenotypes", {})
    if not pheno:
        errors.append("Missing required section: 'phenotypes'")
    else:
        if "source" not in pheno:
            errors.append("phenotypes: missing 'source'")
        elif project_root:
            src = project_root / pheno["source"]
            if not src.exists():
                errors.append(f"phenotypes: source not found: {src}")

        fmt = pheno.get("format", "")
        if fmt not in VALID_PHENO_FORMATS:
            errors.append(f"phenotypes: invalid format '{fmt}'. Use: {VALID_PHENO_FORMATS}")

        if fmt in ("csv", "excel"):
            layout = pheno.get("layout", "long")
            if layout not in VALID_PHENO_LAYOUTS:
                errors.append(f"phenotypes: invalid layout '{layout}'. Use: {VALID_PHENO_LAYOUTS}")

            if layout == "wide":
                if not pheno.get("ft_columns"):
                    errors.append("phenotypes: 'ft_columns' mapping required for wide layout")
                if not pheno.get("planting_dates"):
                    errors.append("phenotypes: 'planting_dates' mapping required for wide layout")
            else:
                if not pheno.get("genotype_col"):
                    errors.append("phenotypes: 'genotype_col' required for long layout")
                if not pheno.get("location_col"):
                    errors.append("phenotypes: 'location_col' required for long layout")
                if not pheno.get("ft_col"):
                    errors.append("phenotypes: 'ft_col' required for long layout")

    # Weather
    weather = config.get("weather", {})
    if not weather:
        errors.append("Missing required section: 'weather'")
    else:
        if "source" not in weather:
            errors.append("weather: missing 'source'")

        fmt = weather.get("format", "")
        if fmt not in VALID_WEATHER_FORMATS:
            errors.append(f"weather: invalid format '{fmt}'. Use: {VALID_WEATHER_FORMATS}")

        if fmt == "combined_csv":
            if not weather.get("site_column"):
                errors.append("weather: 'site_column' required for combined_csv format")
            if not weather.get("tmax_col"):
                errors.append("weather: 'tmax_col' required for combined_csv format")
            if not weather.get("tmin_col"):
                errors.append("weather: 'tmin_col' required for combined_csv format")
            date_fmt = weather.get("date_format", "calendar")
            if date_fmt not in VALID_DATE_FORMATS:
                errors.append(f"weather: invalid date_format '{date_fmt}'. Use: {VALID_DATE_FORMATS}")
            if date_fmt == "year_doy":
                if not weather.get("year_col"):
                    errors.append("weather: 'year_col' required for year_doy date format")
                if not weather.get("doy_col"):
                    errors.append("weather: 'doy_col' required for year_doy date format")

    # Genomics
    genomics = config.get("genomics", {})
    if not genomics:
        errors.append("Missing required section: 'genomics'")
    else:
        tier = genomics.get("tier", "")
        if tier not in VALID_GENOMIC_TIERS:
            errors.append(f"genomics: invalid tier '{tier}'. Use: {VALID_GENOMIC_TIERS}")

        if tier == "evd":
            if not genomics.get("evd_path"):
                errors.append("genomics: 'evd_path' required for evd tier")
        elif tier == "gmatrix":
            if not genomics.get("source"):
                errors.append("genomics: 'source' required for gmatrix tier")
        elif tier == "markers":
            if not genomics.get("source"):
                errors.append("genomics: 'source' required for markers tier")
            coding = genomics.get("marker_coding", "")
            if coding and coding not in VALID_MARKER_CODINGS:
                errors.append(f"genomics: invalid marker_coding '{coding}'. Use: {VALID_MARKER_CODINGS}")

    return errors
