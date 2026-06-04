# Data Format Specification - CGM-WGP

Required format for input data files and the output file formats.

## Table of Contents

1. [Phenotype Data](#phenotype-data)
2. [Weather Data](#weather-data)
3. [Genomic Data](#genomic-data)
4. [Output Formats](#output-formats)

## Phenotype Data

### File: `phenotypes_dated.csv`

**Location:** `pipeline/1_data/{Crop}/phenotypes_dated.csv` (crop-scoped, e.g. `pipeline/1_data/Beans/phenotypes_dated.csv`, `pipeline/1_data/Broccoli/phenotypes_dated.csv`)

**Required columns:**

| Column | Type | Description | Example | Constraints |
|--------|------|-------------|---------|-------------|
| `id` | string | Genotype identifier | FA002, G123 | Unique per genotype |
| `Planting/Environment` | string | Planting date or environment identifier | Planting 1, PR | Groups observations |
| `ft` | integer | Flowering time (days after planting) | 45, 52 | Positive integer |
| `Start_Date` | date | Planting date | 2020-05-01 | YYYY-MM-DD format |
| `End_Date` | date | End of observation window | 2020-07-15 | YYYY-MM-DD format |

`censored` is an additional internal column used by the loader to filter
out unobserved rows. The prep helpers (`inspect_data --prepare`,
`prepare_pipeline.py`) write it automatically with value `0`. When
authoring a phenotype CSV by hand, you can either omit the column (the
prep scripts will add it) or include `censored=0` for observed rows and
`censored=1` for excluded ones.

**Example:**

```csv
id,Planting,ft,Start_Date,End_Date
FA002,Planting 1,45,2020-05-01,2020-07-15
FA002,Planting 2,52,2020-06-15,2020-09-01
FA002,Planting 3,38,2020-04-01,2020-06-15
FA003,Planting 1,48,2020-05-01,2020-07-15
```

### Data Requirements

**Genotype IDs:**
- Can be alphanumeric (FA002, GEN_123, etc.)
- Must be consistent across phenotype and genomic data
- Case-sensitive

**Planting / Environment IDs:**
- Format: "Planting N" where N is a number (e.g., "Planting 1", "Planting 2") or environment name (e.g., "PR", "ND")
- When using `--train-plantings` or `--predict-plantings`, numeric IDs can be referenced as "1,2" and the "Planting " prefix is added automatically
- Can represent different planting dates, locations, or environments

**Flowering Time (ft):**
- Integer days after planting
- Must be positive
- Should be within observation window (End_Date - Start_Date)

**Dates:**
- Format: YYYY-MM-DD (ISO 8601)
- Start_Date: Day of planting (day 0)
- End_Date: Last day of observation window

### Optional Columns

Additional columns are allowed and will be ignored:
- `planting_group`: Numeric planting group
- `time1`, `time2`: Interval censoring bounds
- `Rep`, `Block`, `Plot`: Field design info
- Any other metadata

## Weather Data

### File: `weather.csv`

**Location:** `pipeline/1_data/weather/{LOCATION}/weather.csv`

Weather files are organized by location in subdirectories:

```
pipeline/1_data/weather/
├── FL_Hasting/
│   └── weather.csv
├── CIT/
│   └── weather.csv
└── ...
```

The location directory name (e.g., `FL_Hasting`) is automatically extracted and used in the timestamped output folder name.

**Required columns:**

| Column | Type | Description | Example | Units |
|--------|------|-------------|---------|-------|
| `Period` | date | Calendar date | 2020-05-01 | YYYY-MM-DD |
| `tmx` | float | Daily maximum temperature | 22.5 | C |
| `tmn` | float | Daily minimum temperature | 12.3 | C |

**Example:**

```csv
Period,tmx,tmn
2020-05-01,22.5,12.3
2020-05-02,24.1,13.8
2020-05-03,21.7,11.9
```

### Data Requirements

**Period (Date):**
- Format: YYYY-MM-DD
- Must be continuous (no gaps)
- Must cover all phenotype observation windows (all planting Start_Date to End_Date ranges)

**Temperature:**
- Units: Degrees Celsius (C)
- Daily mean temperature computed internally as: `tmean = (tmx + tmn) / 2`
- Constraint: `tmx` >= `tmn` for each day
- Missing values: Not currently supported (use interpolation if needed)

### Optional Columns

Additional weather variables are allowed but not used:
- `DAYLhr`: Daylength in hours (used directly by Joint photoperiod coupling if present; otherwise computed from latitude via Spencer 1971)
- `prec`: Precipitation (mm)
- `srad`: Solar radiation
- `rhum`: Relative humidity
- Any other meteorological variables

## Genomic Data

### Marker Matrix

**Location:** `pipeline/2_gmatrix/`

- Rows: Genotypes
- Columns: SNP markers
- Values: 0, 1, 2 (allele dosage) or -9 (missing)
- Genotype IDs must match phenotype file

### G-matrix (Genomic Relationship Matrix)

**Location:** `pipeline/2_gmatrix/EVD.rda` (eigenvalue decomposition)

- Square matrix: n_genotypes x n_genotypes
- Symmetric
- Computed using VanRaden method via AGHmatrix R package

## Structured Output

All outputs are written to a timestamped structured run folder under `pipeline/output/runs/`:

```
pipeline/output/runs/{crop}/{location-or-cv-scheme}/{YYYY-MM-DD_HH-MM-SS}/
├── meta.json                      # Run metadata (crop, location, label, tags, git_sha, metrics)
├── params/                        # Fitted per-genotype parameters
│   ├── fitted.csv                 # Mechanistic alpha/beta
│   └── joint_params.csv           # CGM-WGP Joint per-genotype Theta, S, Pc, alpha_fixed, beta_fixed
├── predictions/                   # Per-method DAP predictions
│   ├── mechanistic.csv            # Mechanistic, dual-threshold combiner (thermal AND photo clocks)
│   ├── raw_mechanistic.csv        # Mechanistic, multiplicative coupling (cardinal_beta(T) × F_photo(P))
│   ├── gblup.csv                  # GBLUP DAP predictions with PEV
│   ├── jarquin.csv                # RN-GBLUP (Jarquin 2014) with conditional variance
│   └── joint.csv                  # CGM-WGP Joint predictions
├── diagnostics/                   # Run diagnostics, metrics, and convergence
│   ├── metrics.csv                # Per-planting accuracy metrics
│   ├── validation_comparison.csv  # Side-by-side metrics for the four methods
│   ├── progress.csv               # Cumulative progress H per planting
│   └── run_config.json            # Frozen run configuration
└── plots/                         # Per-run diagnostic PNGs
```

A `latest` symlink at `pipeline/output/runs/{crop}/{location}/latest` points at the most recent run.

### Catalog: `params/fitted.csv`

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Genotype identifier |
| `alpha`, `beta` | float | Fitted cardinal-beta shape parameters |
| `loss` | float | Final mechanistic loss |
| `nit`, `nfev` | integer | Optimizer iteration / function-evaluation counts |
| `n_training_plantings` | integer | Training plantings used for this genotype |

### Catalog: `params/joint_params.csv`

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Genotype identifier |
| `Theta` | float | Per-genotype thermal-time-to-flower threshold |
| `S` | float | Per-genotype photoperiod rate-penalty (>= 0) |
| `Pc` | float | Per-genotype critical daylength (hours) |
| `alpha_fixed`, `beta_fixed` | float | Population-level cardinal-beta shape (derived from Tb/Topt/Tc) |

### Catalog: `predictions/*.csv`

Each method's predictions file shares a common schema:

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Genotype identifier |
| `planting` | string | Prediction planting (environment) |
| `observed_dap` | integer/NA | Observed flowering DAP (NA if censored or missing) |
| `predicted_dap` | integer/NA | Predicted DAP (NA if threshold not reached) |
| `error` | float/NA | predicted - observed |
| `pred_var` | float (where applicable) | Per-genotype prediction variance / PEV (GBLUP, RN-GBLUP) |

Method-specific extras:

- `predictions/mechanistic.csv` — dual-threshold predictions; `predicted_dap = max(day_thermal, day_photo)` where each clock has its own per-genotype threshold (H_thermal_g, H_photo_g). Also `target_H`, `alpha`, `beta`. Enabled when `photoperiod.coupling: dual_threshold` is set in the crop YAML (default for Beans + Broccoli).
- `predictions/raw_mechanistic.csv` — multiplicative-coupling predictions; rates accumulate as `cardinal_beta(T) × F_photo(DL)` per day. Same column schema as `mechanistic.csv`. Pass `--mech-no-photo` at runtime to skip photoperiod entirely and write cardinal-β-only predictions to this file.
- `predictions/gblup.csv` — `breeding_value`, `pred_var`
- `predictions/jarquin.csv` — `pred_var` (conditional variance from G + E + GxE)
- `predictions/joint.csv` — Joint per-genotype Theta/S/Pc baked into the prediction

### Catalog: `diagnostics/`

- `metrics.csv` — per-planting and overall MAE / RMSE / bias / within-N-day counts.
- `validation_comparison.csv` — single-table cross-method comparison (Mechanistic, GBLUP, RN-GBLUP, CGM-WGP Joint).
- `progress.csv` — cumulative progress at flowering per genotype × planting (variance is the mechanistic loss).
- `run_config.json` — frozen CLI/config snapshot for reproducibility.

### Run Registry

- `pipeline/output/index.json` — global JSON registry of every run.
- `pipeline/output/runs/{crop}/{location}/latest` — symlink to most recent run.

Manage runs with `python -m cgm_wgp.runs` (list, tag, label, archive, backfill).

## Data Preparation Checklist

Before running analysis:

- [ ] Phenotype file has all required columns
- [ ] Genotype IDs are consistent across files
- [ ] Planting names follow "Planting N" format
- [ ] Dates are in YYYY-MM-DD format
- [ ] Weather data covers all phenotype date ranges
- [ ] Weather file is in `pipeline/1_data/weather/{LOCATION}/weather.csv`
- [ ] No gaps in weather data
- [ ] Temperature values are reasonable for crop/location
- [ ] At least some uncensored observations (censored=0)
- [ ] Flowering times are positive integers
- [ ] Genomic data available and matches genotype IDs

---

For usage instructions, see [USAGE.md](USAGE.md).
