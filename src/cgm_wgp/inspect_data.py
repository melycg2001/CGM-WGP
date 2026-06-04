"""
inspect_data.py

Pre-preparation data inspector for the CGM-WGP pipeline.
Scans raw data files in a crop's input folder, detects formats,
classifies column roles, checks for missing inputs, and reports findings.

Usage:
  python scripts/inspect_data.py --crop Broccoli
  python scripts/inspect_data.py --crop Broccoli --generate-config
  python scripts/inspect_data.py --crop Broccoli --generate-config --config-output configs/broccoli_draft.yaml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from cgm_wgp.config import DATA_DIR, WEATHER_DIR, CONFIGS_DIR, get_crop_data_dir, PROJECT_ROOT

# ── Constants ────────────────────────────────────────────────────────────────

SCANNABLE_EXTENSIONS = {".csv", ".txt", ".tsv", ".xlsx", ".xls", ".rds", ".rda"}

GENOTYPE_KW = {
    "genotype", "line", "line_code", "linecode", "id", "accession",
    "entry", "cultivar", "variety", "mkid", "ril", "name", "line_name",
    "linename", "inbred", "hybrid", "germplasm",
}
LOCATION_KW = {
    "location", "loc", "env", "environment", "env_code", "envcode",
    "site", "planting", "environment_code",
}
FT_KW = {
    "ft", "flowering", "dap", "days", "r1", "heading", "maturity",
    "anthesis", "silking", "flowering_time",
}
REP_KW = {"rep", "replication", "replicate", "block", "reps"}
TMAX_KW = {"tmx", "tmax", "temp_max", "maxtemp", "temperature_max"}
TMIN_KW = {"tmn", "tmin", "temp_min", "mintemp", "temperature_min"}
DATE_KW = {"period", "date", "datetime", "day", "doy", "year"}
LAT_KW = {"latitude", "lat"}
LON_KW = {"longitude", "lon", "long"}
PLANTING_KW = {
    "planting_date", "plantingdate", "plantdate", "sowing_date",
    "sowdate", "sowing",
}

MARKER_CODING_SETS = {
    "dosage": frozenset({"0", "0.5", "1", "2"}),
    "biallelic_01": frozenset({"0", "1"}),
    "biallelic_12": frozenset({"1", "2"}),
}

METADATA_ROW_MARKERS = {
    "<numeric>", "<marker>", "position", "chromosome", "chr label",
    "chr_label", "chrlabel",
}


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class ColumnHint:
    name: str
    role: str           # genotype, location, ft, rep, date, tmax, tmin,
                        # lat, lon, planting_date, marker, unknown
    confidence: str     # high, medium, low
    reason: str


@dataclass
class SheetInfo:
    name: str
    n_rows: int
    n_cols: int
    headers: list[str]
    column_hints: list[ColumnHint]
    role: str
    role_confidence: str
    role_reason: str
    sample_data: list[dict] = field(default_factory=list)
    header_idx: int = 0


@dataclass
class FileInspection:
    path: Path
    extension: str
    role: str
    role_confidence: str
    role_reason: str
    delimiter: str | None = None
    n_rows: int | None = None
    n_cols: int | None = None
    headers: list[str] = field(default_factory=list)
    sample_rows: list[list] = field(default_factory=list)
    column_hints: list[ColumnHint] = field(default_factory=list)
    sheets: list[SheetInfo] = field(default_factory=list)
    metadata_rows_detected: int = 0
    marker_coding: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class InspectionReport:
    crop: str
    scan_path: Path
    files: list[FileInspection]
    weather_locations_found: list[str]
    genotype_overlap: dict | None = None
    missing_inputs: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _normalize_col(name: str) -> str:
    """Normalize a column name for keyword matching."""
    return name.lower().replace("-", "_").replace(" ", "_").replace(".", "_").strip("_")


def _count_lines(path: Path) -> int:
    """Fast line count using raw byte read."""
    count = 0
    with open(path, "rb") as f:
        while True:
            buf = f.read(1 << 20)  # 1MB chunks
            if not buf:
                break
            count += buf.count(b"\n")
    return count


# ── Format Detection ─────────────────────────────────────────────────────────

def detect_delimiter(path: Path) -> tuple[str, bool, bool]:
    """
    Detect delimiter, BOM, and Windows line endings.
    Returns (delimiter_char, has_bom, has_crlf).
    delimiter_char: "," or "\t" or " " or None
    """
    with open(path, "rb") as f:
        raw = f.read(8192)

    has_bom = raw[:3] == b"\xef\xbb\xbf"
    has_crlf = b"\r\n" in raw

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    lines = text.split("\n")
    # Take up to 10 non-empty lines
    lines = [l.rstrip("\r") for l in lines if l.strip()][:10]
    if not lines:
        return None, has_bom, has_crlf

    # Score each delimiter by consistency of column count across lines
    best_delim = None
    best_score = 0

    for delim in [",", "\t", " "]:
        counts = []
        for line in lines:
            if delim == " ":
                parts = line.split()
            else:
                parts = line.split(delim)
            counts.append(len(parts))

        if not counts or max(counts) <= 1:
            continue

        # Score = median column count * consistency
        counts_set = set(counts)
        # Consistency: fraction of lines matching the most common count
        from collections import Counter
        most_common_count = Counter(counts).most_common(1)[0][1]
        consistency = most_common_count / len(counts)
        median_cols = sorted(counts)[len(counts) // 2]
        score = median_cols * consistency

        if score > best_score:
            best_score = score
            best_delim = delim

    return best_delim, has_bom, has_crlf


def read_text_sample(path: Path, delimiter: str | None, max_data_rows: int = 5
                     ) -> tuple[list[str], list[list[str]], int, int]:
    """
    Read headers and sample rows from a delimited text file.
    Returns (headers, sample_rows, total_line_count, leading_rows_skipped).
    Uses readline() to avoid loading huge files into memory.

    Handles files where the first line(s) are metadata headers (e.g. <Numeric>)
    and the real header is on a later line. Detects this when the first line
    has far fewer columns than subsequent lines.
    """
    total_lines = _count_lines(path)

    def _split(line: str) -> list[str]:
        if delimiter and delimiter == " ":
            return line.split()
        elif delimiter:
            return line.split(delimiter)
        return [line]

    # Read first several lines to detect metadata prefix
    raw_lines = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for _ in range(max_data_rows + 5):
            line = f.readline()
            if not line:
                break
            raw_lines.append(line.rstrip("\r\n"))

    if not raw_lines:
        return [], [], total_lines, 0

    # Split all lines
    split_lines = [_split(l) for l in raw_lines if l.strip()]

    if not split_lines:
        return [], [], total_lines, 0

    # Detect leading metadata rows: if first line has far fewer columns
    # than a later line, treat leading narrow lines as metadata to skip.
    skip = 0
    if len(split_lines) >= 2:
        max_cols = max(len(s) for s in split_lines[1:])
        while skip < len(split_lines) - 1:
            if len(split_lines[skip]) < max_cols * 0.5:
                skip += 1
            else:
                break

    headers = split_lines[skip] if skip < len(split_lines) else split_lines[0]
    sample_rows = split_lines[skip + 1:][:max_data_rows]

    return headers, sample_rows, total_lines, skip


# ── Excel Reading ────────────────────────────────────────────────────────────

def read_excel_sheets(path: Path) -> list[SheetInfo]:
    """Read all sheets from an Excel file using openpyxl read-only mode."""
    try:
        import openpyxl
    except ImportError:
        return []

    sheets = []
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)

    for ws_name in wb.sheetnames:
        ws = wb[ws_name]
        # Read first 10 rows to find header
        rows_raw = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i >= 10:
                break
            rows_raw.append(list(row))

        if not rows_raw:
            sheets.append(SheetInfo(
                name=ws_name, n_rows=0, n_cols=0, headers=[],
                column_hints=[], role="unknown", role_confidence="low",
                role_reason="empty sheet",
            ))
            continue

        # Find header row: first row where >= half cells are non-None strings
        header_idx = 0
        for idx, row in enumerate(rows_raw):
            non_none = sum(1 for c in row if c is not None and isinstance(c, str) and c.strip())
            if len(row) > 0 and non_none / len(row) >= 0.4:
                header_idx = idx
                break

        headers = [str(c).strip() if c is not None else "" for c in rows_raw[header_idx]]
        # Remove trailing empty headers
        while headers and not headers[-1]:
            headers.pop()

        # Sample data rows after header
        data_rows = rows_raw[header_idx + 1:]
        sample_data = []
        for row in data_rows[:5]:
            d = {}
            for j, h in enumerate(headers):
                if h and j < len(row):
                    d[h] = row[j]
            sample_data.append(d)

        n_rows = ws.max_row or 0
        n_cols = len(headers)

        # Classify columns
        column_hints = classify_columns(headers, data_rows)

        # Classify sheet role
        role, conf, reason = _classify_sheet_role(headers, data_rows, n_cols, n_rows, column_hints)

        sheets.append(SheetInfo(
            name=ws_name, n_rows=n_rows, n_cols=n_cols, headers=headers,
            column_hints=column_hints, role=role, role_confidence=conf,
            role_reason=reason, sample_data=sample_data,
            header_idx=header_idx,
        ))

    wb.close()
    return sheets


# ── Column Classification ────────────────────────────────────────────────────

def classify_columns(headers: list[str], sample_rows: list[list]) -> list[ColumnHint]:
    """Classify each column header into a pipeline role."""
    hints = []
    for i, h in enumerate(headers):
        norm = _normalize_col(h)
        role = "unknown"
        confidence = "low"
        reason = ""

        # Check each keyword set (order matters: more specific before generic)
        # Check planting_date BEFORE location since "planting" is in both
        if norm in PLANTING_KW or ("planting" in norm and "date" in norm):
            role, confidence, reason = "planting_date", "high", f"matches keyword '{norm}'"
        elif norm in LAT_KW:
            role, confidence, reason = "lat", "high", f"matches keyword '{norm}'"
        elif norm in LON_KW:
            role, confidence, reason = "lon", "high", f"matches keyword '{norm}'"
        elif norm in TMAX_KW:
            role, confidence, reason = "tmax", "high", f"matches keyword '{norm}'"
        elif norm in TMIN_KW:
            role, confidence, reason = "tmin", "high", f"matches keyword '{norm}'"
        elif norm in GENOTYPE_KW or any(kw in norm for kw in GENOTYPE_KW if len(kw) > 3):
            role, confidence, reason = "genotype", "high", f"matches keyword '{norm}'"
        elif norm in LOCATION_KW or any(kw in norm for kw in LOCATION_KW if len(kw) > 3):
            role, confidence, reason = "location", "high", f"matches keyword '{norm}'"
        elif norm in FT_KW or any(kw in norm for kw in FT_KW if len(kw) > 3):
            role, confidence, reason = "ft", "high", f"matches keyword '{norm}'"
        elif norm in REP_KW or any(kw in norm for kw in REP_KW if len(kw) > 3):
            role, confidence, reason = "rep", "high", f"matches keyword '{norm}'"
        elif norm in DATE_KW or any(kw in norm for kw in DATE_KW if len(kw) > 3):
            role, confidence, reason = "date", "high", f"matches keyword '{norm}'"
        else:
            # Check for FT-like numeric values
            if sample_rows and i < len(sample_rows[0]):
                vals = []
                for row in sample_rows:
                    if i < len(row) and row[i] is not None:
                        try:
                            vals.append(float(str(row[i])))
                        except (ValueError, TypeError):
                            pass
                if vals and 20 <= sum(vals) / len(vals) <= 250:
                    # Could be FT but low confidence without keyword match
                    pass

        hints.append(ColumnHint(name=h, role=role, confidence=confidence, reason=reason))
    return hints


# ── Role Classification ──────────────────────────────────────────────────────

def _classify_sheet_role(headers: list[str], sample_rows: list, n_cols: int, n_rows: int,
                         column_hints: list[ColumnHint]) -> tuple[str, str, str]:
    """Classify a sheet/file role from its column hints and shape."""
    roles = {h.role for h in column_hints if h.role != "unknown"}

    # Metadata: has lat/lon or planting date columns
    if {"lat", "lon"}.issubset(roles) or "planting_date" in roles:
        return "metadata", "high", "has coordinate or planting date columns"

    # Weather: has date + temperature columns, narrow shape
    if ("date" in roles or "planting_date" in roles) and ("tmax" in roles or "tmin" in roles):
        return "weather", "high", "has date + temperature columns"

    # Markers: very wide
    if n_cols > 100:
        return "markers", "high", f"{n_cols} columns suggests marker matrix"

    # Phenotype: has genotype + location + ft
    if "genotype" in roles and "location" in roles and "ft" in roles:
        return "phenotype", "high", "has genotype, location, and flowering time columns"

    # Phenotype with just genotype + ft (wide format maybe)
    if "genotype" in roles and "ft" in roles:
        return "phenotype", "medium", "has genotype and flowering time columns"

    # Partial matches
    if "genotype" in roles and "location" in roles:
        return "phenotype", "low", "has genotype and location but no clear flowering time column"

    return "unknown", "low", "no clear role pattern detected"


def _detect_markers(headers: list[str], sample_rows: list[list[str]], n_cols: int
                    ) -> tuple[str | None, int]:
    """Detect marker coding and metadata rows to skip.
    Returns (marker_coding, metadata_rows_to_skip)."""
    # Detect metadata rows at the start
    skip = 0
    for row in sample_rows:
        if row and row[0].lower().strip() in METADATA_ROW_MARKERS:
            skip += 1
        else:
            break

    # Sample values from data rows (after metadata)
    data_rows = sample_rows[skip:]
    if not data_rows:
        return None, skip

    values = set()
    for row in data_rows[:3]:
        # Skip first column (genotype ID), sample up to 200 values
        for val in row[1:201]:
            val_s = str(val).strip()
            if val_s:
                values.add(val_s)

    if not values:
        return None, skip

    # Check against known coding sets
    for coding_name, valid_set in MARKER_CODING_SETS.items():
        if values.issubset(valid_set):
            return coding_name, skip

    # Check if values look numeric and restricted
    try:
        numeric_vals = {float(v) for v in values}
        unique_ints = {int(v) if v == int(v) else v for v in numeric_vals}
        if unique_ints.issubset({0, 1, 2}):
            if 2 in unique_ints:
                return "dosage", skip
            else:
                return "biallelic_01", skip
        if unique_ints.issubset({1, 2}):
            return "biallelic_12", skip
    except (ValueError, TypeError):
        pass

    return None, skip


# ── File Inspection ──────────────────────────────────────────────────────────

def inspect_text_file(path: Path) -> FileInspection:
    """Inspect a delimited text file."""
    delimiter, has_bom, has_crlf = detect_delimiter(path)
    headers, sample_rows, total_lines, leading_skip = read_text_sample(path, delimiter)

    n_rows = total_lines - 1 - leading_skip  # subtract header + metadata
    n_cols = len(headers)

    notes = []
    if has_bom:
        notes.append("File has BOM (byte order mark)")
    if has_crlf:
        notes.append("Windows line endings (CRLF)")

    column_hints = classify_columns(headers, sample_rows)

    # Check if it's a marker file
    marker_coding = None
    metadata_rows = leading_skip  # rows already skipped by reader
    if n_cols > 100:
        # Also check for metadata patterns in sample rows
        extra_coding, extra_skip = _detect_markers(headers, sample_rows, n_cols)
        if extra_coding:
            marker_coding = extra_coding
        metadata_rows += extra_skip

        if marker_coding or n_cols > 1000:
            role = "markers"
            conf = "high"
            coding_str = marker_coding or "unknown"
            reason = f"{n_cols:,} cols, coding={coding_str}, skip {metadata_rows} metadata rows"
            return FileInspection(
                path=path, extension=path.suffix, role=role,
                role_confidence=conf, role_reason=reason,
                delimiter=delimiter, n_rows=n_rows - extra_skip, n_cols=n_cols,
                headers=headers, sample_rows=sample_rows,
                column_hints=column_hints, metadata_rows_detected=metadata_rows,
                marker_coding=marker_coding, notes=notes,
            )

    # Check column-based role
    role, conf, reason = _classify_sheet_role(headers, sample_rows, n_cols, n_rows, column_hints)

    return FileInspection(
        path=path, extension=path.suffix, role=role,
        role_confidence=conf, role_reason=reason,
        delimiter=delimiter, n_rows=n_rows, n_cols=n_cols,
        headers=headers, sample_rows=sample_rows,
        column_hints=column_hints, metadata_rows_detected=leading_skip,
        notes=notes,
    )


def inspect_excel_file(path: Path) -> FileInspection:
    """Inspect an Excel file with all its sheets."""
    sheets = read_excel_sheets(path)

    # Overall file role is determined by the most important sheet
    file_role = "unknown"
    file_conf = "low"
    file_reason = "Excel file with multiple sheets"

    for s in sheets:
        if s.role == "metadata" and s.role_confidence == "high":
            file_role = "metadata"
            file_conf = "high"
            file_reason = f"sheet '{s.name}' has coordinate/planting date columns"
            break

    if file_role == "unknown":
        for s in sheets:
            if s.role == "phenotype" and s.role_confidence in ("high", "medium"):
                file_role = "supplementary"
                file_conf = "medium"
                file_reason = f"contains phenotype data in sheet '{s.name}'"
                break

    return FileInspection(
        path=path, extension=path.suffix, role=file_role,
        role_confidence=file_conf, role_reason=file_reason,
        sheets=sheets,
    )


def inspect_r_binary(path: Path) -> FileInspection:
    """Inspect an .rds or .rda file (just note type and size)."""
    size = path.stat().st_size
    ext = path.suffix.lower()
    if ext == ".rds":
        reason = f"R binary ({size / 1024:.0f} KB) — single object, likely G-matrix"
    else:
        reason = f"R workspace ({size / 1024:.0f} KB) — may contain EVD or G-matrix"

    return FileInspection(
        path=path, extension=ext, role="genomic_binary",
        role_confidence="high", role_reason=reason,
        notes=["Requires R to read; cannot inspect contents directly"],
    )


def inspect_file(path: Path) -> FileInspection:
    """Top-level dispatcher for file inspection."""
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return inspect_excel_file(path)
    elif ext in (".rds", ".rda"):
        return inspect_r_binary(path)
    else:
        return inspect_text_file(path)


# ── Cross-File Validation ────────────────────────────────────────────────────

def _read_genotype_ids_text(path: Path, delimiter: str | None, col_idx: int,
                            skip_rows: int = 0) -> set[str]:
    """Read genotype IDs from a specific column of a text file."""
    ids = set()
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        f.readline()  # skip header
        for _ in range(skip_rows):
            f.readline()  # skip metadata rows
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            if delimiter == " ":
                parts = line.split()
            elif delimiter:
                parts = line.split(delimiter)
            else:
                parts = [line]
            if col_idx < len(parts):
                ids.add(parts[col_idx].strip())
    return ids


def check_genotype_overlap(pheno: FileInspection, markers: FileInspection) -> dict:
    """Check genotype ID overlap between phenotype and marker files."""
    # Find genotype column index in phenotype
    pheno_geno_idx = 0
    for i, h in enumerate(pheno.column_hints):
        if h.role == "genotype":
            pheno_geno_idx = i
            break

    # Marker genotype is always first column (after metadata rows)
    pheno_ids = _read_genotype_ids_text(
        pheno.path, pheno.delimiter, pheno_geno_idx)
    marker_ids = _read_genotype_ids_text(
        markers.path, markers.delimiter, 0, skip_rows=markers.metadata_rows_detected)

    overlap = pheno_ids & marker_ids
    only_pheno = pheno_ids - marker_ids
    only_marker = marker_ids - pheno_ids

    return {
        "pheno_n": len(pheno_ids),
        "marker_n": len(marker_ids),
        "overlap_n": len(overlap),
        "only_in_pheno_n": len(only_pheno),
        "only_in_marker_n": len(only_marker),
        "only_in_pheno_examples": sorted(only_pheno)[:5],
        "only_in_marker_examples": sorted(only_marker)[:5],
    }


def check_weather_coverage(locations: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Check which locations have weather data.
    Returns (found_dirs, covered, missing)."""
    found_dirs = []
    if WEATHER_DIR.is_dir():
        found_dirs = sorted(d.name for d in WEATHER_DIR.iterdir()
                            if d.is_dir() and (d / "weather.csv").exists())

    covered = [loc for loc in locations if loc in found_dirs]
    missing = [loc for loc in locations if loc not in found_dirs]
    return found_dirs, covered, missing


def _extract_metadata_from_sheets(file_inspections: list[FileInspection]
                                  ) -> dict | None:
    """Extract planting dates and coordinates from detected metadata sheets.
    Returns dict with 'dates', 'coordinates', 'env_col', 'sheet_name', 'file_path'
    or None if not found."""
    for fi in file_inspections:
        for sheet in fi.sheets:
            if sheet.role != "metadata":
                continue

            # Find column indices by role
            env_col = None
            date_col = None
            lat_col = None
            lon_col = None

            # Prefer columns with "code" or "env" in the name for environment ID
            env_candidates = []
            for hint in sheet.column_hints:
                if hint.role == "location":
                    env_candidates.append(hint.name)
                elif hint.role == "planting_date":
                    date_col = hint.name
                elif hint.role == "lat":
                    lat_col = hint.name
                elif hint.role == "lon":
                    lon_col = hint.name

            # Rank env candidates: prefer "code"/"env" in name, then first match
            for cand in env_candidates:
                norm = _normalize_col(cand)
                if "code" in norm or "env" in norm:
                    env_col = cand
                    break
            if not env_col and env_candidates:
                env_col = env_candidates[0]

            # Fallback: direct header matching for environment code
            if not env_col:
                for h in sheet.headers:
                    norm = _normalize_col(h)
                    if "environment" in norm and "code" in norm:
                        env_col = h
                        break

            if not env_col or not sheet.sample_data:
                continue

            dates = {}
            coords = {}
            locations = {}

            for row in sheet.sample_data:
                env = row.get(env_col)
                if env is None or str(env).strip() == "":
                    # Try inheriting env from the set context (merged cells)
                    continue
                env = str(env).strip()

                if date_col and date_col in row and row[date_col] is not None:
                    val = row[date_col]
                    if isinstance(val, datetime):
                        dates[env] = val.strftime("%Y-%m-%d")
                    else:
                        dates[env] = str(val).split(" ")[0]  # "2011-03-30 00:00:00" -> date part

                if lat_col and lon_col:
                    lat = row.get(lat_col)
                    lon = row.get(lon_col)
                    if lat is not None and lon is not None:
                        try:
                            coords[env] = (float(lat), float(lon))
                        except (ValueError, TypeError):
                            pass

                # Location name (city)
                for h in sheet.headers:
                    norm = _normalize_col(h)
                    if norm == "location" and h in row and row[h]:
                        locations[env] = str(row[h])

            if dates or coords:
                # Read ALL data rows, not just the sample
                all_data = _read_full_metadata_sheet(fi.path, sheet.name,
                                                    env_col, date_col, lat_col, lon_col)
                if all_data:
                    dates, coords, locations = all_data

                return {
                    "dates": dates,
                    "coordinates": coords,
                    "locations": locations,
                    "env_col": env_col,
                    "sheet_name": sheet.name,
                    "file_path": fi.path,
                }
    return None


def _read_full_metadata_sheet(path: Path, sheet_name: str,
                              env_col: str, date_col: str | None,
                              lat_col: str | None, lon_col: str | None
                              ) -> tuple[dict, dict, dict] | None:
    """Read all rows from a metadata sheet to get complete date/coord data."""
    try:
        import openpyxl
    except ImportError:
        return None

    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    ws = wb[sheet_name]

    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if len(rows) < 2:
        return None

    # Find header row
    header_idx = 0
    for idx, row in enumerate(rows):
        non_none = sum(1 for c in row if c is not None and isinstance(c, str) and c.strip())
        if len(row) > 0 and non_none / len(row) >= 0.4:
            header_idx = idx
            break

    headers = [str(c).strip() if c is not None else "" for c in rows[header_idx]]

    # Find column indices
    col_map = {}
    for i, h in enumerate(headers):
        if h == env_col:
            col_map["env"] = i
        if date_col and h == date_col:
            col_map["date"] = i
        if lat_col and h == lat_col:
            col_map["lat"] = i
        if lon_col and h == lon_col:
            col_map["lon"] = i
        norm = _normalize_col(h)
        if norm == "location":
            col_map["location"] = i

    if "env" not in col_map:
        return None

    dates = {}
    coords = {}
    locations = {}

    for row in rows[header_idx + 1:]:
        env_val = row[col_map["env"]] if col_map["env"] < len(row) else None
        if env_val is None or str(env_val).strip() == "":
            continue
        env = str(env_val).strip()

        if "date" in col_map and col_map["date"] < len(row):
            val = row[col_map["date"]]
            if val is not None:
                if isinstance(val, datetime):
                    dates[env] = val.strftime("%Y-%m-%d")
                else:
                    s = str(val).strip()
                    if s:
                        dates[env] = s.split(" ")[0]

        if "lat" in col_map and "lon" in col_map:
            lat = row[col_map["lat"]] if col_map["lat"] < len(row) else None
            lon = row[col_map["lon"]] if col_map["lon"] < len(row) else None
            if lat is not None and lon is not None:
                try:
                    coords[env] = (float(lat), float(lon))
                except (ValueError, TypeError):
                    pass

        if "location" in col_map and col_map["location"] < len(row):
            loc_val = row[col_map["location"]]
            if loc_val is not None:
                locations[env] = str(loc_val).strip()

    return dates, coords, locations


# ── Phenotype Stats ──────────────────────────────────────────────────────────

def _get_pheno_stats(fi: FileInspection) -> dict:
    """Extract phenotype statistics: unique genotypes, locations, ft range."""
    geno_idx = 0
    loc_idx = 1
    ft_idx = -1

    for i, h in enumerate(fi.column_hints):
        if h.role == "genotype":
            geno_idx = i
        elif h.role == "location":
            loc_idx = i
        elif h.role == "ft":
            ft_idx = i

    if ft_idx < 0:
        return {}

    genotypes = set()
    locations = set()
    ft_vals = []

    with open(fi.path, "r", encoding="utf-8-sig", errors="replace") as f:
        f.readline()  # header
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            if fi.delimiter == " ":
                parts = line.split()
            elif fi.delimiter:
                parts = line.split(fi.delimiter)
            else:
                continue

            if geno_idx < len(parts):
                genotypes.add(parts[geno_idx].strip())
            if loc_idx < len(parts):
                locations.add(parts[loc_idx].strip())
            if ft_idx < len(parts):
                try:
                    ft_vals.append(float(parts[ft_idx].strip()))
                except ValueError:
                    pass

    result = {
        "n_genotypes": len(genotypes),
        "n_locations": len(locations),
        "locations": sorted(locations),
    }
    if ft_vals:
        result["ft_min"] = int(min(ft_vals))
        result["ft_max"] = int(max(ft_vals))
    return result


# ── Report Building ──────────────────────────────────────────────────────────

def build_report(crop: str, scan_path: Path,
                 file_inspections: list[FileInspection]) -> InspectionReport:
    """Assemble all findings into a structured report."""
    report = InspectionReport(
        crop=crop, scan_path=scan_path, files=file_inspections,
        weather_locations_found=[],
    )

    # Find phenotype and marker files
    pheno_file = None
    marker_file = None
    metadata_info = None

    for fi in file_inspections:
        if fi.role == "phenotype" and fi.role_confidence in ("high", "medium"):
            pheno_file = fi
        elif fi.role == "markers" and fi.role_confidence in ("high", "medium"):
            marker_file = fi

    # Extract metadata from Excel sheets
    metadata_info = _extract_metadata_from_sheets(file_inspections)

    # Get phenotype locations
    pheno_locations = []
    if pheno_file:
        stats = _get_pheno_stats(pheno_file)
        pheno_locations = stats.get("locations", [])

    # Weather check
    found_dirs, covered, missing = check_weather_coverage(pheno_locations)
    report.weather_locations_found = found_dirs

    # Genotype overlap
    if pheno_file and marker_file:
        report.genotype_overlap = check_genotype_overlap(pheno_file, marker_file)

    # Missing inputs
    if not pheno_file:
        report.missing_inputs.append("Phenotype data: no file detected with genotype + location + flowering time columns")
    # Check for genomic data including inside Excel sheets
    has_genomic = (
        marker_file is not None
        or any(fi.role == "genomic_binary" for fi in file_inspections)
        or any(s.role == "markers" for fi in file_inspections for s in fi.sheets)
    )
    if not has_genomic:
        report.missing_inputs.append("Genomic data: no marker matrix or R binary file detected")
    if missing:
        n = len(missing)
        if n == len(pheno_locations):
            report.missing_inputs.append(f"Weather data: missing for ALL {n} environments")
        else:
            report.missing_inputs.append(f"Weather data: missing for {n}/{len(pheno_locations)} environments: {', '.join(missing[:5])}")
        if metadata_info and metadata_info["coordinates"]:
            report.suggestions.append(
                f"Coordinates available in {metadata_info['file_path'].name} ({metadata_info['sheet_name']}) "
                "-- can fetch weather from NASA POWER API"
            )

    # Planting date coverage
    if pheno_locations and metadata_info and metadata_info["dates"]:
        covered_dates = [loc for loc in pheno_locations if loc in metadata_info["dates"]]
        missing_dates = [loc for loc in pheno_locations if loc not in metadata_info["dates"]]
        if missing_dates:
            report.alerts.append(
                f"Planting dates missing for: {', '.join(missing_dates)} "
                f"(available for {len(covered_dates)}/{len(pheno_locations)})"
            )
    elif pheno_locations and not metadata_info:
        report.alerts.append("No planting dates found in any data file -- must be provided manually in YAML config")

    # Genotype overlap alerts
    if report.genotype_overlap:
        ov = report.genotype_overlap
        if ov["only_in_pheno_n"] > 0:
            report.alerts.append(
                f"{ov['only_in_pheno_n']} phenotype genotypes have no markers "
                f"(will be excluded from genomic prediction)"
            )

    return report


# ── Report Printing ──────────────────────────────────────────────────────────

def _delim_name(d: str | None) -> str:
    if d == ",": return "comma"
    if d == "\t": return "tab"
    if d == " ": return "space"
    return "unknown"


def print_report(report: InspectionReport, quiet: bool = False):
    """Print the inspection report to stdout."""
    w = 60
    print(f"\n{'=' * w}")
    print(f"CGM-WGP Data Inspection: {report.crop}")
    print(f"Scanned: {report.scan_path}")
    print(f"{'=' * w}")

    # Files
    print(f"\nFILES FOUND ({len(report.files)})")
    print("-" * w)

    for i, fi in enumerate(report.files, 1):
        role_str = fi.role.upper()
        conf_str = f"[{fi.role_confidence.upper()}]"

        if fi.extension in (".xlsx", ".xls"):
            print(f"\n  [{i}] {fi.path.name} -- {role_str} {conf_str}")
            print(f"      Format: Excel | Sheets: {len(fi.sheets)}")
            print(f"      {fi.role_reason}")
            if fi.sheets:
                print()
                for s in fi.sheets:
                    r = s.role.upper()
                    c = s.role_confidence.upper()
                    print(f"      Sheet '{s.name}' ({s.n_rows} rows x {s.n_cols} cols) "
                          f"-- {r} [{c}]")
                    if s.role_reason:
                        print(f"        {s.role_reason}")
        elif fi.extension in (".rds", ".rda"):
            print(f"\n  [{i}] {fi.path.name} -- {role_str} {conf_str}")
            print(f"      {fi.role_reason}")
        else:
            print(f"\n  [{i}] {fi.path.name} -- {role_str} {conf_str}")
            print(f"      Delimiter: {_delim_name(fi.delimiter)} | "
                  f"{fi.n_rows:,} rows x {fi.n_cols:,} cols")
            print(f"      {fi.role_reason}")

        # Column mappings for phenotype/weather
        if fi.role in ("phenotype", "weather") and fi.column_hints:
            mapped = [h for h in fi.column_hints if h.role != "unknown"]
            if mapped:
                print()
                for h in mapped:
                    print(f"      {h.name:20s} -> {h.role:15s} [{h.confidence.upper()}]")

        # Phenotype stats
        if fi.role == "phenotype" and fi.role_confidence in ("high", "medium"):
            stats = _get_pheno_stats(fi)
            if stats:
                parts = [f"{stats['n_genotypes']} genotypes", f"{stats['n_locations']} environments"]
                if "ft_min" in stats:
                    parts.append(f"ft range {stats['ft_min']}-{stats['ft_max']} days")
                print(f"      Stats: {', '.join(parts)}")

        # Marker details
        if fi.role == "markers":
            if fi.metadata_rows_detected:
                print(f"      Metadata rows to skip: {fi.metadata_rows_detected}")
            if fi.marker_coding:
                print(f"      Marker coding: {fi.marker_coding}")

        # Notes
        for note in fi.notes:
            print(f"      NOTE: {note}")

    # Weather
    print(f"\n{'- ' * 30}")
    print("WEATHER CHECK")
    print("-" * w)

    if report.weather_locations_found:
        print(f"  Existing weather dirs: {len(report.weather_locations_found)}")
    else:
        print("  No weather directories found")

    # Cross-file validation
    if report.genotype_overlap:
        print(f"\n{'- ' * 30}")
        print("CROSS-FILE VALIDATION")
        print("-" * w)

        ov = report.genotype_overlap
        print(f"  Phenotype genotypes:  {ov['pheno_n']}")
        print(f"  Marker genotypes:     {ov['marker_n']}")
        print(f"  Overlap:              {ov['overlap_n']} "
              f"({ov['overlap_n']/max(ov['pheno_n'],1)*100:.1f}%)")
        if ov["only_in_pheno_n"]:
            print(f"  Only in phenotype:    {ov['only_in_pheno_n']}  "
                  f"e.g. {', '.join(ov['only_in_pheno_examples'][:3])}")
        if ov["only_in_marker_n"]:
            print(f"  Only in markers:      {ov['only_in_marker_n']}  "
                  f"e.g. {', '.join(ov['only_in_marker_examples'][:3])}")

    # Summary
    print(f"\n{'- ' * 30}")
    print("SUMMARY")
    print("-" * w)

    if report.missing_inputs:
        for m in report.missing_inputs:
            print(f"  [MISSING] {m}")
    if report.alerts:
        for a in report.alerts:
            print(f"  [!] {a}")
    if report.suggestions:
        for s in report.suggestions:
            print(f"  [->] {s}")

    # What's OK
    has_pheno = any(fi.role == "phenotype" and fi.role_confidence in ("high", "medium")
                    for fi in report.files)
    has_genomic = any(
        fi.role in ("markers", "genomic_binary")
        or any(s.role == "markers" for s in fi.sheets)
        for fi in report.files
    )
    if has_pheno:
        print("  [OK] Phenotype data detected")
    if has_genomic:
        print("  [OK] Genomic data detected")
    if not report.missing_inputs and not report.alerts:
        print("  All inputs appear ready!")

    print(f"\n{'=' * w}")


# ── Config Generation ────────────────────────────────────────────────────────

def generate_draft_config(report: InspectionReport) -> str:
    """Generate a draft YAML config string with TODO comments."""
    lines = []

    def add(text: str):
        lines.append(text)

    crop = report.crop

    # Find key files
    pheno_file = None
    marker_file = None
    genomic_binary = None
    metadata_info = _extract_metadata_from_sheets(report.files)
    pheno_stats = {}

    for fi in report.files:
        if fi.role == "phenotype" and fi.role_confidence in ("high", "medium"):
            pheno_file = fi
            pheno_stats = _get_pheno_stats(fi)
        elif fi.role == "markers":
            marker_file = fi
        elif fi.role == "genomic_binary":
            genomic_binary = fi

    add(f"crop: {crop}")
    add("")

    # Phenotypes
    add("phenotypes:")
    if pheno_file:
        rel_path = pheno_file.path.relative_to(PROJECT_ROOT)
        add(f'  source: "{rel_path}"')
        add("  format: csv")
        add("  layout: long")

        geno_col = next((h.name for h in pheno_file.column_hints if h.role == "genotype"), None)
        loc_col = next((h.name for h in pheno_file.column_hints if h.role == "location"), None)
        ft_col = next((h.name for h in pheno_file.column_hints if h.role == "ft"), None)
        rep_col = next((h.name for h in pheno_file.column_hints if h.role == "rep"), None)

        if geno_col:
            add(f"  genotype_col: {geno_col}")
        else:
            add("  genotype_col: TODO_GENOTYPE_COLUMN  # TODO: set genotype column name")
        if loc_col:
            add(f"  location_col: {loc_col}")
        else:
            add("  location_col: TODO_LOCATION_COLUMN  # TODO: set location column name")
        if ft_col:
            add(f"  ft_col: {ft_col}")
        else:
            add("  ft_col: TODO_FT_COLUMN  # TODO: set flowering time column name")
        if rep_col:
            add(f"  rep_col: {rep_col}")

        # Planting dates
        if metadata_info and metadata_info["dates"]:
            add("  planting_dates:")
            for env in sorted(metadata_info["dates"].keys()):
                date_str = metadata_info["dates"][env]
                add(f'    "{env}": "{date_str}"')
        elif pheno_stats.get("locations"):
            add("  planting_dates:  # TODO: add planting dates for each environment")
            for loc in pheno_stats["locations"]:
                add(f'    "{loc}": "YYYY-MM-DD"  # TODO')
    else:
        add('  source: "TODO"  # TODO: path to phenotype file')
        add("  format: csv")
        add("  layout: long")
        add("  genotype_col: TODO")
        add("  location_col: TODO")
        add("  ft_col: TODO")

    add("")

    # Weather
    add("weather:")
    add(f"  source: pipeline/1_data/weather")
    add("  format: multi_location_dir")
    weather_missing = any("Weather" in m for m in report.missing_inputs)
    if weather_missing:
        add("  # TODO: weather data not found for this crop's environments")
        add("  # Options:")
        add("  #   1. Place per-location weather.csv files in pipeline/1_data/weather/{env_code}/")
        add("  #   2. Provide a combined CSV and change format to combined_csv")
        if metadata_info and metadata_info["coordinates"]:
            add("  #   3. Fetch from NASA POWER using coordinates from metadata")

    add("")

    # Genomics
    add("genomics:")
    if marker_file:
        rel_path = marker_file.path.relative_to(PROJECT_ROOT)
        add(f"  tier: markers")
        add(f'  source: "{rel_path}"')
        add("  format: csv")
        if marker_file.metadata_rows_detected:
            add(f"  metadata_rows: {marker_file.metadata_rows_detected}")
        if marker_file.marker_coding:
            add(f"  marker_coding: {marker_file.marker_coding}")
        else:
            add('  marker_coding: dosage  # TODO: verify coding (0/1/2 or 0/0.5/1)')
    elif genomic_binary:
        ext = genomic_binary.extension
        rel_path = genomic_binary.path.relative_to(PROJECT_ROOT)
        if ext == ".rda":
            add("  tier: evd  # TODO: verify -- could be G-matrix (.rda)")
            add(f'  evd_path: "{rel_path}"')
        else:
            add("  tier: gmatrix")
            add(f'  source: "{rel_path}"')
            add(f"  format: {ext.lstrip('.')}")
    else:
        add('  tier: markers  # TODO: set tier (markers, gmatrix, or evd)')
        add('  source: "TODO"  # TODO: path to genomic data')

    add("")

    # Locations (coordinates)
    if metadata_info and metadata_info["coordinates"]:
        add("locations:")
        for env in sorted(metadata_info["coordinates"].keys()):
            lat, lon = metadata_info["coordinates"][env]
            add(f"  {env}:")
            add(f"    latitude: {lat}")
        add("")

    # Model
    add("model:")
    add(f"  Tb: 10.0    # TODO: set base temperature for {crop}")
    add(f"  Topt: 30.0  # TODO: set optimal temperature for {crop}")
    add(f"  Tc: 40.0    # TODO: set ceiling temperature for {crop}")
    add("  # photoperiod:")
    add("  #   type: short_day  # TODO: set if photoperiod-sensitive")
    add("  #   fit: true")
    add("  #   B: 3.0")
    add("  #   Pc: 13.0")

    return "\n".join(lines) + "\n"


# ── Main Entry Points ────────────────────────────────────────────────────────

def inspect_and_report(crop: str, generate_config: bool = False,
                       config_output: str | None = None,
                       quiet: bool = False,
                       prepare: bool = False,
                       cardinal_temps: dict | None = None,
                       ) -> InspectionReport:
    """Main inspection entry point."""
    scan_path = get_crop_data_dir(crop)

    if not scan_path.exists():
        print(f"ERROR: Data directory not found: {scan_path}", file=sys.stderr)
        print(f"Create it and place your raw data files there.", file=sys.stderr)
        sys.exit(1)

    # Discover and inspect files
    files = sorted(
        (p for p in scan_path.iterdir()
         if p.is_file() and p.suffix.lower() in SCANNABLE_EXTENSIONS),
        key=lambda p: p.name,
    )

    if not files:
        print(f"No data files found in {scan_path}", file=sys.stderr)
        print(f"Place .csv, .txt, .xlsx, .rds, or .rda files there.", file=sys.stderr)
        sys.exit(1)

    inspections = [inspect_file(f) for f in files]

    # Build and print report
    report = build_report(crop, scan_path, inspections)
    print_report(report, quiet=quiet)

    # Prepare data (implies config generation)
    if prepare:
        from cgm_wgp.prepare_from_inspection import prepare_from_report
        prepare_from_report(
            report=report,
            crop=crop,
            config_output=config_output,
            cardinal_temps=cardinal_temps,
        )
    elif generate_config:
        config_str = generate_draft_config(report)
        if config_output:
            out_path = Path(config_output)
        else:
            out_path = CONFIGS_DIR / f"{crop.lower()}.yaml"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(config_str)
        print(f"\nDraft config written to: {out_path}")
        print("Review TODO comments and fill in missing values before running prepare_pipeline.py")

    return report


def main():
    p = argparse.ArgumentParser(
        description="Inspect raw data files before writing a YAML config or running prepare_pipeline.py.",
    )
    p.add_argument("--crop", required=True,
                   help="Crop name (scans pipeline/1_data/{crop}/)")
    p.add_argument("--generate-config", action="store_true",
                   help="Write a draft YAML config to configs/{crop}.yaml")
    p.add_argument("--config-output", default=None,
                   help="Override output path for generated config")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress sample data rows in report")
    p.add_argument("--prepare", action="store_true",
                   help="Extract and prepare data into pipeline formats "
                        "(phenotypes, weather, config). Only generates missing files.")
    p.add_argument("--cardinal-temps", default=None,
                   help='Override cardinal temps as JSON: '
                        '\'{"Tb": 8, "Topt": 34, "Tc": 44}\'')
    args = p.parse_args()

    cardinal = None
    if args.cardinal_temps:
        import json
        cardinal = json.loads(args.cardinal_temps)

    inspect_and_report(
        crop=args.crop,
        generate_config=args.generate_config,
        config_output=args.config_output,
        quiet=args.quiet,
        prepare=args.prepare,
        cardinal_temps=cardinal,
    )


if __name__ == "__main__":
    main()
