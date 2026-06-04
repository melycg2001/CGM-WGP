# Usage Guide - CGM-WGP

Examples and workflows for using the CGM-WGP framework.

## Table of Contents

1. [YAML Config System](#yaml-config-system)
2. [Basic Workflows](#basic-workflows)
3. [Adding a New Crop](#adding-a-new-crop)
4. [Validation Methods](#validation-methods)
5. [CV Schemes](#cv-schemes)
6. [Parameter Tuning](#parameter-tuning)
7. [Interpreting Results](#interpreting-results)
8. [Common Scenarios](#common-scenarios)
9. [Structured Output System](#structured-output-system)

## YAML Config System

The config-driven pipeline replaces long CLI argument lists with one YAML file per crop. A config captures all crop-specific settings (phenotype layout, weather format, genomic data tier, cardinal temperatures) so data preparation and model runs are reproducible with a short command.

### Config File Location

Configs live in `configs/`. Beans and Broccoli configs ship with the public repo:

```
configs/
├── broccoli.yaml
└── beans.yaml
```

### Config Schema Reference

A config YAML has five top-level sections:

```yaml
crop: MyCrop                   # Crop name (used for output directories)

phenotypes:
  source: path/to/phenotype_file.csv
  format: csv                  # pipeline_ready | csv | excel
  layout: long                 # long | wide  (only for csv/excel)
  # Additional fields depend on format/layout (see below)

weather:
  source: path/to/weather
  format: single_location      # single_location | multi_location_dir | combined_csv
  # Additional fields depend on format (see below)

genomics:
  tier: gmatrix                # markers | gmatrix | evd
  # Additional fields depend on tier (see below)

model:
  Tb: 5.0                     # Base temperature
  Topt: 20.0                  # Optimum temperature
  Tc: 35.0                    # Ceiling temperature
  alpha_bounds: [0.5, 8.0]    # Optional (default [0.5, 8.0])
  beta_bounds: [0.5, 8.0]     # Optional (default [0.5, 8.0])
```

#### `phenotypes` section

| Field | Required | Values | Description |
|-------|----------|--------|-------------|
| `source` | Yes | file path | Path to raw phenotype data (relative to project root) |
| `format` | Yes | `pipeline_ready`, `csv`, `excel` | Input format. `pipeline_ready` means the file already has `id, Planting, ft, Start_Date, End_Date` columns |
| `layout` | For csv/excel | `long`, `wide` | Long: one row per observation. Wide: locations as columns |
| `genotype_col` | For csv/excel | column name | Column containing genotype IDs |
| `location_col` | For long layout | column name | Column containing location/environment |
| `ft_col` | For long layout | column name | Column containing flowering time (DAP) |
| `ft_columns` | For wide layout | mapping | Map of `Location: column_name` for flowering time columns |
| `sheet` | For excel | sheet name | Excel sheet to read |
| `missing_value` | Optional | string | String representing missing data (default `"-"`) |
| `planting_dates` | For csv/excel | mapping | Map of `"Location": "YYYY-MM-DD"` planting dates |

#### `weather` section

| Field | Required | Values | Description |
|-------|----------|--------|-------------|
| `source` | Yes | path | Weather file or directory |
| `format` | Yes | `single_location`, `multi_location_dir`, `combined_csv` | How weather data is organized |
| `date_format` | For combined_csv | `calendar`, `year_doy` | Date representation in source file |
| `site_column` | For combined_csv | column name | Column identifying the site/location |
| `year_col` | For year_doy | column name | Column containing year |
| `doy_col` | For year_doy | column name | Column containing day of year |
| `tmax_col` | Optional | column name | Max temperature column (default `"Tmax"`) |
| `tmin_col` | Optional | column name | Min temperature column (default `"Tmin"`) |

#### `genomics` section

| Field | Required | Values | Description |
|-------|----------|--------|-------------|
| `tier` | Yes | `markers`, `gmatrix`, `evd` | Starting point for genomic data. `markers` computes G-matrix then EVD. `gmatrix` computes EVD. `evd` uses pre-computed EVD |
| `source` | For markers/gmatrix | path | Path to marker or G-matrix file |
| `evd_path` | For evd tier | path | Path to pre-computed EVD.rda |
| `gmatrix_path` | Optional | path | Path to pre-computed Gmatrix.rda |
| `format` | Optional | `csv`, `excel`, `rds` | File format of source |
| `sheet` | For excel | sheet name | Excel sheet to read |
| `marker_coding` | For markers | `biallelic_12`, `biallelic_01`, `dosage` | How alleles are encoded in the marker matrix |
| `genotype_col` | For markers | column name | Column with genotype IDs in the marker file |
| `metadata_rows` | For markers | integer | Number of header/metadata rows to skip before genotype data |
| `missing_value` | For markers | string | String representing missing marker data |

#### `model` section

| Field | Default | Description |
|-------|---------|-------------|
| `Tb` | 5.0 | Base temperature (no development below this) |
| `Topt` | 20.0 | Optimum temperature (maximum development rate) |
| `Tc` | 35.0 | Ceiling temperature (no development above this) |
| `alpha_bounds` | [0.5, 8.0] | Search bounds for the alpha shape parameter |
| `beta_bounds` | [0.5, 8.0] | Search bounds for the beta shape parameter |
| `joint_fit.enabled` | false | Run CGM-WGP Joint method (equivalent to `--run-joint`) |
| `joint_fit.photo_enabled` | true | Use Hadley additive photoperiod coupling inside Joint |
| `joint_fit.photo_direction` | short_day | `short_day` (beans) or `long_day` (broccoli) — sign convention for the rate-penalty |
| `joint_fit.Theta_bounds` | [5, 200] | Search bounds for per-genotype Theta (thermal-time threshold) |
| `joint_fit.S_bounds` | [0.0, 0.3] | Search bounds for per-genotype S (photoperiod rate-penalty, >= 0) |
| `joint_fit.Pc_bounds` | [11, 15] | Search bounds for per-genotype Pc (critical daylength, hours) |
| `joint_fit.max_em_iters` | 50 | Maximum EM iterations for Joint |
| `joint_fit.em_tol` | 1e-4 | Relative convergence tolerance for EM |
| `joint_fit.lbfgsb_maxiter` | 1000 | L-BFGS-B inner-step max iterations |

Joint photoperiod coupling uses the Hadley additive rate-penalty form — see
[TECHNICAL.md](TECHNICAL.md) for the forward-model math.

The Mechanistic baseline fits per-genotype (α, β) to cardinal-β only (no
photoperiod in the loss), then combines that thermal fit with a population-
level photoperiod fit at prediction time. `predictions/mechanistic.csv` uses
the dual-threshold combiner (a plant flowers when both the thermal and photo
clocks have completed), and `predictions/raw_mechanistic.csv` uses
multiplicative coupling (`cardinal_beta(T) × F_photo(P)`). Pass
`--mech-no-photo` to drop the photoperiod step entirely and predict from
cardinal-β alone. See [TECHNICAL.md § Mechanistic Model](TECHNICAL.md).

#### `locations` section (optional)

Maps location names to latitudes for daylength computation when the weather
CSV does not already contain a `DAYLhr` column.

```yaml
locations:
  CIT:
    latitude: 10.0
  ND:
    latitude: 46.9
```

If the weather CSV contains a `DAYLhr` column, latitude is not required — the
observed daylength values are used directly.

### `prepare_pipeline.py` — Config-Driven Data Preparation

`prepare_pipeline.py` reads a YAML config and prepares all pipeline inputs (phenotypes, weather, genomics) in one command.

**CLI arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `--config` | Yes | Path to the crop YAML config file |
| `--validate-only` | No | Check that all expected pipeline files are in place without preparing anything |
| `--force` | No | Overwrite existing output files. Without this flag, existing files are skipped |

**What it does (5 steps):**

1. **Creates directories** — `pipeline/1_data/{crop}/` and `pipeline/2_gmatrix/{crop}/`
2. **Prepares phenotypes** — Converts raw data to `phenotypes_dated.csv` according to the `format` and `layout` in the config
3. **Prepares weather** — For `single_location`, validates the file exists. For `multi_location_dir`, checks subdirectories. For `combined_csv`, splits the file into per-location `weather.csv` files
4. **Prepares genomics** — Depending on `tier`: `markers` computes VanRaden G-matrix then EVD; `gmatrix` computes EVD from an existing G-matrix; `evd` validates and copies pre-computed files
5. **Validates** — Confirms all expected outputs exist

**Examples:**

```bash
# Prepare all data for a crop
python -m cgm_wgp.prepare_pipeline --config configs/beans.yaml

# Check if data is already in place (no changes made)
python -m cgm_wgp.prepare_pipeline --config configs/broccoli.yaml --validate-only

# Re-prepare, overwriting existing files
python -m cgm_wgp.prepare_pipeline --config configs/beans.yaml --force
```

### `app.py` `--config` Flag

The pipeline orchestrator (`app.py`) accepts a `--config` flag that populates default values from the YAML config. Explicit CLI arguments override config values.

**Fields populated from config:**

| Config field | CLI argument populated |
|--------------|----------------------|
| `crop` | `--crop` |
| `model.Tb` | `--Tb` |
| `model.Topt` | `--Topt` |
| `model.Tc` | `--Tc` |
| `model.alpha_bounds` | `--alpha-bounds` |
| `model.beta_bounds` | `--beta-bounds` |
| `model.joint_fit.enabled` | `--run-joint` |
| `weather.source` + `weather.format` | `--weather` |
| `phenotypes` (derived) | `--phenotypes` (defaults to `pipeline/1_data/{crop}/phenotypes_dated.csv`) |
| `genomics` (derived) | `--evd-path` (defaults to `pipeline/2_gmatrix/{crop}/EVD.rda`) |

**Example — config-only invocation:**

```bash
# Minimal: config supplies crop, temperatures, weather path
python -m cgm_wgp.app --config configs/broccoli.yaml \
  --train-plantings 1,2,3,4 \
  --predict-plantings 5,6
```

**Example — config with CLI overrides:**

```bash
# Override cardinal temperatures
python -m cgm_wgp.app --config configs/broccoli.yaml \
  --Tb 3.0 --Topt 18.0 --Tc 32.0 \
  --run-joint
```

## Basic Workflows

### Workflow 1: Quick Test on Single Genotype

Test the mechanistic model on one genotype to verify data and setup:

```bash
venv/bin/python3 -m cgm_wgp.main_fit \
  --phenotypes pipeline/1_data/Broccoli/phenotypes_dated.csv \
  --weather pipeline/1_data/weather/FL_Hasting/weather.csv \
  --genotype FA002 \
  --crop broccoli \
  --maxiter 500
```

Output goes to `pipeline/output/runs/broccoli/FL_Hasting/{timestamp}/params/fitted.csv`:
```csv
id,alpha,beta,loss,nit,nfev
FA002,3.988,3.999,118.76,1000,5840
```

### Workflow 2: Fit All Genotypes (Mechanistic Only)

Run the mechanistic model on all genotypes without genomic information:

```bash
venv/bin/python3 -m cgm_wgp.main_fit \
  --phenotypes pipeline/1_data/Broccoli/phenotypes_dated.csv \
  --weather pipeline/1_data/weather/FL_Hasting/weather.csv \
  --crop broccoli \
  --maxiter 1000
```

**What happens:**
- Fits alpha, beta for each genotype independently via dual annealing
- Creates output files in structured run folder:
  - `params/fitted.csv` — parameter estimates
  - `diagnostics/progress.csv` — cumulative progress by planting

### Workflow 3: Train/Predict Split (Mechanistic Only)

Fit parameters on a subset of plantings and predict flowering in others (no GBLUP):

```bash
venv/bin/python3 -m cgm_wgp.main_fit \
  --phenotypes pipeline/1_data/Broccoli/phenotypes_dated.csv \
  --weather pipeline/1_data/weather/FL_Hasting/weather.csv \
  --crop Broccoli \
  --train-plantings 1,2,3,4 \
  --predict-plantings 5,6 \
  --maxiter 1000
```

**What happens:**
1. Fits MAP alpha/beta using only the training plantings via dual annealing
2. Predicts DAP on held-out plantings via cardinal-beta integration
3. Outputs `predictions/mechanistic.csv` and `predictions/raw_mechanistic.csv`
4. Prints prediction summary (MAE, RMSE, bias, per-planting breakdown)

**Output prediction file format:**
```csv
id,planting,observed_dap,predicted_dap,error,target_H,alpha,beta
FA002,Planting 5,52,49,-3.0,15.2340,3.988000,3.999000
FA002,Planting 6,39,41,2.0,15.2340,3.988000,3.999000
```

### Workflow 4: Full Pipeline (4 Methods)

`app.py` orchestrates the four-method registry: Mechanistic, GBLUP, RN-GBLUP,
and CGM-WGP Joint (Joint is enabled by `--run-joint`).

```bash
venv/bin/python3 -m cgm_wgp.app \
  --maxiter 1000 \
  --crop Broccoli \
  --phenotypes pipeline/1_data/Broccoli/phenotypes_dated.csv \
  --weather pipeline/1_data/weather/FL_Hasting/weather.csv \
  --train-plantings 1,2,3,4 \
  --predict-plantings 5,6 \
  --run-joint
```

All output goes to a structured timestamped folder:
`pipeline/output/runs/Broccoli/FL_Hasting/{timestamp}/` (see
[Structured Output System](#structured-output-system)).

**What runs:**

1. **Mechanistic**: dual annealing fits per-genotype (alpha, beta); writes `predictions/mechanistic.csv` and `predictions/raw_mechanistic.csv`
2. **GBLUP**: REML + BLUP on EVD, with PEV; writes `predictions/gblup.csv`
3. **RN-GBLUP** (Jarquin 2014): G + E_K + GxE kernels with conditional variance; writes `predictions/jarquin.csv`
4. **CGM-WGP** (with `--run-joint`): EM + L-BFGS-B fitting per-genotype (Theta, S, Pc) under MVN genomic prior with Hadley additive photoperiod coupling (Messina et al. 2006, Technow et al. 2015, Cooper et al. 2016); writes `predictions/joint.csv` and `params/joint_params.csv`
5. **Validation comparison table**: `diagnostics/validation_comparison.csv` summarises all four methods side-by-side

### Workflow 5: Debug Daily Rates

Inspect detailed daily temperature and rate calculations:

```bash
venv/bin/python3 -m cgm_wgp.main_fit \
  --phenotypes pipeline/1_data/Broccoli/phenotypes_dated.csv \
  --weather pipeline/1_data/weather/FL_Hasting/weather.csv \
  --genotype FA002 \
  --crop broccoli \
  --debug \
  --maxiter 500
```

**Outputs in run folder:**
- `params/fitted.csv` — fitted parameters
- `diagnostics/debug_daily_rates.csv` — detailed daily data

**Daily rates file format:**
```csv
id,planting,day_index,date,temp,rate
FA002,Planting 1,0,2020-05-01,17.4,0.456
FA002,Planting 1,1,2020-05-02,18.2,0.489
FA002,Planting 1,2,2020-05-03,16.1,0.398
```

### Workflow 6: CGM-WGP Joint

CGM-WGP Joint (Messina/Technow framework; see TECHNICAL.md § Joint CGM-WGP)
fits all genotypes simultaneously under an MVN genomic prior via EM +
L-BFGS-B. Runs alongside the other three methods on the same CV split,
producing `predictions/joint.csv` and `params/joint_params.csv` (per-genotype
Theta, S, Pc).

**Enable via config** (preferred):

```yaml
# configs/<crop>.yaml
model:
  joint_fit:
    enabled: true
    photo_enabled: true
    photo_direction: short_day        # or long_day for Brassicas, etc.
    Theta_bounds: [5, 200]
    S_bounds: [0.0, 0.3]              # rate-penalty per hour (>= 0 always)
    Pc_bounds: [11, 15]               # critical daylength
    max_em_iters: 50
    em_tol: 0.0001
    lbfgsb_maxiter: 1000
```

**Or via CLI flag** for a one-off run:

```bash
venv/bin/python3 -m cgm_wgp.app \
  --config configs/beans.yaml \
  --cv-scheme balanced_training_subset \
  --run-joint
```

**When to enable Joint**

| Scenario | Use Joint? |
|---|---|
| **CV0** (new environments, known genotypes) | **Yes** — best method on this scheme for both Beans and Broccoli |
| **CV00** (new env + new genotype) | **Yes** — best method |
| CV2 (sparse testing within trained envs) | No — RN-GBLUP wins (kernel BLUP interpolates better when training and test share envs) |
| CV1 (new varieties in trained envs) | No — RN-GBLUP wins for the same reason |
| Single-latitude trial | Optional — photoperiod signal is weak; effect is small either way |

## Adding a New Crop

### Config-Driven Workflow (Recommended)

To add a new crop, create a YAML config and let `prepare_pipeline.py` handle data preparation.

**Step 1: Create a config file**

Create `configs/newcrop.yaml` describing your data:

```yaml
crop: NewCrop

phenotypes:
  source: pipeline/1_data/NewCrop/raw_phenotypes.csv
  format: csv
  layout: long
  genotype_col: GenoID
  location_col: Site
  ft_col: FlowerDAP
  planting_dates:
    "Site A": "2024-05-10"
    "Site B": "2024-06-01"

weather:
  source: pipeline/1_data/weather
  format: multi_location_dir

genomics:
  tier: gmatrix
  source: pipeline/1_data/NewCrop/Gmatrix.rds
  format: rds

model:
  Tb: 8.0
  Topt: 25.0
  Tc: 38.0
```

**Step 2: Place raw data files**

Put your raw phenotype and genomic files at the paths referenced in the config. Place weather files under `pipeline/1_data/weather/{location}/weather.csv` (one per location).

**Step 3: Prepare all pipeline data**

```bash
python -m cgm_wgp.prepare_pipeline --config configs/newcrop.yaml
```

**Step 4: Validate (optional)**

```bash
python -m cgm_wgp.prepare_pipeline --config configs/newcrop.yaml --validate-only
```

**Step 5: Run the pipeline**

```bash
python -m cgm_wgp.app --config configs/newcrop.yaml --run-joint
```

### Inspecting Raw Data (Before Writing a Config)

When onboarding a new crop from external files, use `inspect_data.py` to analyze what you have:

```bash
venv/bin/python3 -m cgm_wgp.inspect_data --crop Beans
```

This scans `pipeline/1_data/Beans/` and reports:
- **File classification**: phenotype, markers, weather, metadata, or unknown
- **Column mapping**: which columns are genotype IDs, locations, flowering time, etc.
- **Marker detection**: coding scheme (dosage, biallelic), metadata rows to skip
- **Metadata extraction**: planting dates and coordinates from supplementary Excel files
- **Cross-file validation**: genotype overlap between phenotype and marker files
- **Weather coverage**: which environments have weather data and which are missing

To also generate a draft YAML config:

```bash
venv/bin/python3 -m cgm_wgp.inspect_data --crop Beans --generate-config
```

This writes `configs/beans.yaml` with detected values filled in and TODO comments for anything requiring manual input.

### Auto-Preparation (`--prepare`)

The `--prepare` flag extracts raw data into pipeline-ready formats and generates a complete config YAML with no TODO comments. Only generates files that don't already exist, so safe to run multiple times (idempotent).

**What `--prepare` does:**

1. Scans all raw files (Excel sheets, CSV, TXT) and auto-detects phenotype, weather, and metadata
2. Generates `pipeline/1_data/{Crop}/phenotypes_dated.csv` from raw phenotype data (averages reps, maps planting dates)
3. Generates per-environment `pipeline/1_data/weather/{env}/weather.csv` from embedded weather data (auto-converts Fahrenheit to Celsius when median TMAX > 50)
4. If no embedded weather but lat/long coordinates are available: fetches daily weather from the NASA POWER API (T2M_MAX, T2M_MIN, ALLSKY_SFC_SW_DWN; no authentication; 1-second rate limit; daylength computed from latitude via Spencer 1971)
5. Generates `configs/{crop}.yaml` with all fields populated

**`inspect_data.py` CLI arguments:**

| Flag | Description |
|------|-------------|
| `--crop NAME` | Crop name (scans `pipeline/1_data/{NAME}/`) |
| `--generate-config` | Generate a draft config YAML (may contain TODO comments) |
| `--prepare` | Extract and prepare data into pipeline formats. Implies config generation. Only generates missing files. |
| `--cardinal-temps JSON` | Override default cardinal temperatures. Example: `'{"Tb": 8, "Topt": 34, "Tc": 44}'` |

**Example: Beans end-to-end with `--prepare`**

```bash
# Step 1: Place raw data files in pipeline/1_data/Beans/

# Step 2: Inspect and auto-prepare
venv/bin/python3 -m cgm_wgp.inspect_data --crop Beans --prepare \
  --cardinal-temps '{"Tb": 8, "Topt": 25, "Tc": 38}'

# Step 3: Review the generated config
cat configs/beans.yaml

# Step 4: Run the pipeline
venv/bin/python3 -m cgm_wgp.app --config configs/beans.yaml \
  --train-plantings "CIT,ND,PAL" --predict-plantings "POP,PR" \
  --run-joint --maxiter 1000
```

**Generated files (only if they do not already exist):**

| File | Description |
|------|-------------|
| `pipeline/1_data/{Crop}/phenotypes_dated.csv` | Pipeline-format phenotypes (id, Planting, ft, Start_Date, End_Date, censored) |
| `pipeline/1_data/weather/{env}/weather.csv` | Per-environment daily weather (Period, tmx, tmn, DAYLhr) |
| `configs/{crop}.yaml` | Complete crop config with all fields populated |

### Manual Workflow (Step-by-Step)

To run each preparation step individually, follow the steps below.

### Step 1: Prepare Phenotype Data

Use `cgm_wgp.prepare_crop_data` to convert raw phenotype data to the pipeline format:

```bash
venv/bin/python3 -m cgm_wgp.prepare_crop_data \
  --input pipeline/1_data/Beans/raw_phenotypes.csv \
  --crop Beans \
  --genotype-col RIL \
  --location-col Site \
  --ft-col DTF \
  --planting-dates "CIT=2018-11-15,ND=2019-05-25,..." \
  --location-map "CIT=CIT,ND=ND,..."
```

**Key arguments:**

| Argument | Description |
|----------|-------------|
| `--input` | Path to raw phenotype CSV |
| `--crop` | Crop name (determines output directory) |
| `--genotype-col` | Column containing genotype IDs |
| `--location-col` | Column containing location/environment |
| `--ft-col` | Column containing flowering time (days after planting) |
| `--planting-dates` | Per-location planting dates: `"Loc1=2024-05-10,Loc2=2024-05-20"` or path to CSV |
| `--season` | Default planting date for all locations (fallback) |
| `--location-map` | Explicit mapping of raw location names to weather folder names |
| `--normalize-locations` | Convert location names to filesystem-safe format (spaces → underscores) |

Output: `pipeline/1_data/{crop}/phenotypes_dated.csv` with columns `id, Planting, ft, Start_Date, End_Date, censored`.

### Step 2: Generate EVD from G-matrix

```bash
venv/bin/python3 -m cgm_wgp.generate_evd \
  --gmatrix pipeline/1_data/Beans/Gmatrix.rds \
  --output pipeline/2_gmatrix/Beans/EVD.rda \
  --save-gmatrix
```

Supports `.rds` (R serialized) and `.rda` (R data) formats. Genotype IDs are taken from row/column names of the matrix.

### Step 3: Place Weather Files

Weather data is shared across all crops in `pipeline/1_data/weather/`. Each location has its own subfolder:

```
pipeline/1_data/weather/              # Shared weather directory
├── FL_Hasting/weather.csv            # Used by Broccoli (single-location)
├── CIT/weather.csv                   # Used by Beans (multi-location)
├── ND/weather.csv
├── PAL/weather.csv
└── .../weather.csv
```

Each `weather.csv` needs columns: `Period` (YYYY-MM-DD), `tmx` (max temp), `tmn` (min temp).

The folder names must match the Planting column values in `phenotypes_dated.csv`. Use `--location-map` to map raw location names to weather folder names.

### Step 4: Run the Pipeline

Pass the shared weather **directory** instead of a file:

```bash
venv/bin/python3 -m cgm_wgp.main_fit \
  --phenotypes pipeline/1_data/Beans/phenotypes_dated.csv \
  --weather pipeline/1_data/weather/ \
  --crop Beans \
  --Tb 8 --Topt 25 --Tc 38 \
  --train-plantings "CIT,ND,PAL" \
  --predict-plantings "POP,PR" \
  --maxiter 1000
```

**Multi-location weather mode**: When `--weather` points to a directory, the pipeline loads `{dir}/{Planting}/weather.csv` for each unique location. When it points to a file, all observations use that single weather file.

### Cardinal Temperatures (paper crops)

| Crop | Tb | Topt | Tc |
|------|-----|------|-----|
| Broccoli | 5.0 | 25.0 | 40.0 |
| Beans | 10.0 | 25.0 | 38.0 |

---

## Validation Methods

When `--predict-plantings` (or `--cv-scheme`) is specified, the framework runs the
four-method registry and compares them in a validation table.

| Method | Description |
|--------|-------------|
| **Mechanistic** | Per-genotype (α, β) from temp-only dual annealing on cardinal-β progress; prediction uses dual-threshold combiner (thermal clock AND photoperiod clock must complete). `predictions/mechanistic.csv` is the dual-threshold output; `predictions/raw_mechanistic.csv` is the multiplicative variant. Pass `--mech-no-photo` to drop photoperiod entirely. |
| **GBLUP** | EnvCov GBLUP (REML + BLUP on EVD); linear regression of training planting means on mean temperature plus per-genotype breeding value + PEV. |
| **RN-GBLUP** | Reaction Norm model (Jarquin 2014) with G, E_K, and GxE kernels: `y = mu + g + e + ge + epsilon`. Provides conditional variance per prediction. |
| **CGM-WGP** | Per-genotype (Theta, S, Pc) under MVN genomic prior; EM + L-BFGS-B; Hadley additive photoperiod coupling (Messina et al. 2006, Technow et al. 2015, Cooper et al. 2016). Enable with `--run-joint`. |

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--evd-path` | `pipeline/2_gmatrix/{crop}/EVD.rda` | Path to eigenvalue decomposition of G-matrix |
| `--no-gblup-dap` | False | Skip GBLUP DAP prediction |
| `--run-joint` | False | Run CGM-WGP Joint alongside the baselines |
| `--cv-scheme` | None | Cross-validation partition: `balanced_training_subset`, `new_varieties`, `individual`, `cv00_double_novelty` |

### How Each Method Works

**Mechanistic** fits per-genotype (alpha, beta) via dual annealing on a temperature-only variance-of-cumulative-progress loss. A separate population-level photoperiod fit (`fit_photo_only`) then derives a per-genotype photoperiod threshold `H_photo_g`. Two prediction tracks are written:

- `predictions/mechanistic.csv` — dual-threshold combiner: `predicted_DAP = max(day_thermal, day_photo)` where each day is the first to cross its respective threshold. Enabled by `photoperiod.coupling: dual_threshold` in the crop YAML (default for Beans + Broccoli).
- `predictions/raw_mechanistic.csv` — multiplicative coupling: rates accumulate as `cardinal_beta(T) × F_photo(DL)` per day. No GBLUP penalty in the (α, β) fit.

Pass `--mech-no-photo` to drop the photoperiod step entirely (predictions become cardinal-β only).

**GBLUP DAP** (EnvCov GBLUP):
1. Loads the eigenvalue decomposition (EVD) of the G-matrix from `pipeline/2_gmatrix/{crop}/EVD.rda`
2. Computes adjusted phenotypes: `y_adj = observed_DAP - planting_mean`
3. Estimates variance components via REML (spectral decomposition)
4. Computes genomic breeding values: `u_hat = V diag(d/(d+lambda)) V' y_adj`
5. Computes PEV: `PEV_i = sigma_g^2 * sum_k(v_ik^2 * lambda / (d_k + lambda))`
6. Predicts: `predicted_DAP = center + u_hat` with `pred_var = PEV`

**RN-GBLUP** (Jarquin 2014 Reaction Norm):
1. Constructs genomic (G), environmental (E_K), and GxE (G # E_K) kernel matrices
2. Fits the model: `y = mu + g + e + ge + epsilon` with REML variance component estimation
3. Predicts DAP for new genotype-environment combinations using the fitted model
4. Computes conditional variance: `PEV = Var(y_new) - k_cross' V^{-1} k_cross`
5. Predictions written to `predictions/jarquin.csv` with `pred_var` column

**CGM-WGP** fits per-genotype (Theta, S, Pc) under an MVN genomic prior (`theta ~ N(mu_k 1, sigma^2_k G)`) via EM with L-BFGS-B inner loop. Hadley additive photoperiod coupling penalises daily rate by `S_g * deviation_from_Pc_g` rather than multiplying by a g(P) factor. Predictions: `predictions/joint.csv`; per-genotype parameters: `params/joint_params.csv`. See [TECHNICAL.md](TECHNICAL.md) for the forward model.

### Unseen Genotype Predictions (CV1, CV00)

The pipeline predicts for all genotypes in the G-matrix, including those with
no phenotype data in any training planting. This matters for cross-maturity
prediction where genotype panels may be disjoint.

**How it works:**

1. GBLUP DAP and RN-GBLUP predict for all G-matrix genotypes using genomic, environmental, and GxE relationships
2. Joint uses the spectral BLUP form to predict Theta from genomic relationships for held-out genotypes; S and Pc fall back to the population mean
3. For Mechanistic, a G-matrix proxy uses GBLUP of fitted genotypes' mechanistic predictions

No configuration needed; unseen genotype prediction runs automatically when a G-matrix/EVD is available.

### Validation Comparison Table

When `--predict-plantings` is specified, a validation comparison CSV
(`diagnostics/validation_comparison.csv`) is produced with side-by-side
metrics for all four methods:

```
method,center_used,rmse,mae,bias,n_eval,...
Mechanistic,"P5=48.2, P6=39.1 (mech MAP)",11.4,8.4,3.8,250,...
GBLUP,"P5=47.0, P6=38.8 (temp regression)",10.1,7.5,2.8,250,...
RN-GBLUP,"Jarquin 2014 Reaction Norm (G+E+GxE)",9.5,7.0,2.2,250,...
CGM-WGP,"per-genotype (Theta, S, Pc)",8.6,6.3,1.5,250,...
```

The `center_used` column shows what planting mean each method used, making it easy to diagnose whether centering or breeding values are the bottleneck.

## CV Schemes

`--cv-scheme` selects the four-scheme CV partition logic used in the paper.
Each scheme runs the full four-method registry on the resulting train/test
split.

| Scheme | Holdout unit | Description |
|--------|--------------|-------------|
| `balanced_training_subset` | Planting subsets | GxE Sparse Testing (CV2) within trained envs |
| `new_varieties` | 20% genotypes | New Varieties (CV1): whole-genotype holdout across all envs, 5 reps |
| `individual` | One environment | New Location (CV0): leave-one-environment-out |
| `cv00_double_novelty` | Env + 20% genotypes | New Program (CV00): joint novelty in env *and* genotype |

```bash
# CV1 (New Varieties)
venv/bin/python3 -m cgm_wgp.app \
  --config configs/beans.yaml \
  --cv-scheme new_varieties \
  --run-joint

# CV0 (New Location, leave-one-environment-out)
venv/bin/python3 -m cgm_wgp.app \
  --config configs/beans.yaml \
  --cv-scheme individual \
  --run-joint

# CV00 (joint env + genotype novelty)
venv/bin/python3 -m cgm_wgp.app \
  --config configs/beans.yaml \
  --cv-scheme cv00_double_novelty \
  --run-joint
```

CV outputs land under `pipeline/output/runs/{Crop}/{cv-scheme}/{timestamp}/`.

## Parameter Tuning

### Cardinal Temperature Selection

The cardinal temperatures define the temperature response curve:

```bash
venv/bin/python3 -m cgm_wgp.main_fit \
  --phenotypes pipeline/1_data/Broccoli/phenotypes_dated.csv \
  --weather pipeline/1_data/weather/FL_Hasting/weather.csv \
  --crop broccoli \
  --Tb 5.0 \
  --Topt 20.0 \
  --Tc 35.0
```

**Paper crop values:**
- **Broccoli**: Tb=5.0C, Topt=25.0C, Tc=40.0C
- **Beans**: Tb=10.0C, Topt=25.0C, Tc=38.0C

### Optimization Intensity

Control the thoroughness of the optimization:

```bash
# Quick test (fast but may not converge fully)
--maxiter 200

# Standard (good balance)
--maxiter 1000

# Intensive (best results, slower)
--maxiter 3000
```

### Alpha/Beta Parameter Bounds

```bash
# Default bounds
--alpha-bounds 0.5 8.0 --beta-bounds 0.5 8.0

# Wider bounds (if defaults are too restrictive for your crop)
--alpha-bounds 0.1 6.0 --beta-bounds 0.1 6.0
```

Tighter bounds constrain parameters to biologically plausible ranges and often improve results by preventing overfitting. Widen only if many genotypes hit the default bounds.

### Optimizer Restarts and Loss Normalization

**Multiple restarts** (`--n-restarts`): The dual annealing optimizer may find local minima. Running multiple restarts with different random seeds explores more of the loss landscape:

```bash
# Default: single run (fast)
--n-restarts 1

# Better global optima (3-5x slower)
--n-restarts 3
```

**Loss normalization** (`--normalize-loss`): By default, the loss is `sum(diffs^2)` across training plantings. Genotypes observed in 4 plantings have 4x higher raw loss than genotypes in 1 planting. Normalizing divides by the count:

```bash
--normalize-loss
```

**Guidelines:**
- `--n-restarts 1` (default): Fast, sufficient when the loss landscape is smooth
- `--n-restarts 3-5`: Use when many genotypes hit parameter bounds or RMSE is high
- `--normalize-loss`: Use when genotypes have varying numbers of training observations

## Interpreting Results

### Main Results File

**File**: `params/fitted.csv` (in run folder)

```csv
id,alpha,beta,loss,nit,nfev,n_training_plantings
FA002,3.988,3.999,118.76,1000,5840,4
FA003,3.176,3.937,111.52,1000,4095,4
```

**Column interpretations:**
- `alpha`, `beta`: Fitted cardinal-beta shape parameters
- `loss`: MAP mechanistic loss value (Var(H_e)). Lower = better fit
- `nit`, `nfev`: Optimizer iteration / function-evaluation counts
- `n_training_plantings`: Number of training plantings used for this genotype

**Warning signs:**
- alpha or beta near bounds (0.5 or 8.0): May need wider bounds
- Very high loss (>1000): Check data quality for that genotype
- `nit == maxiter`: Increase maxiter for better convergence

### Joint Per-Genotype Parameters

**File**: `params/joint_params.csv`

```csv
id,Theta,S,Pc,alpha_fixed,beta_fixed
FA002,38.4,0.018,13.7,4.0,1.0
```

- `Theta`: Per-genotype thermal-time-to-flower threshold
- `S`: Per-genotype photoperiod rate-penalty per hour of deviation (>= 0)
- `Pc`: Per-genotype critical daylength (hours)
- `alpha_fixed`, `beta_fixed`: Population-level cardinal-beta shape derived from Tb/Topt/Tc

### Prediction Results File

**File**: `predictions/mechanistic.csv` (in run folder, when using `--predict-plantings`)

```csv
id,planting,observed_dap,predicted_dap,error,target_H,alpha,beta
FA002,Planting 3,38,41,3.0,15.2340,3.988000,3.999000
FA002,Planting 5,52,49,-3.0,15.2340,3.988000,3.999000
FA002,Planting 6,60,NA,NA,15.2340,3.988000,3.999000
```

**Column interpretations:**
- `observed_dap`: Actual flowering time from data (NA if censored or missing)
- `predicted_dap`: Model prediction using fitted alpha/beta (NA if threshold not reached)
- `error`: predicted - observed
- `target_H`: Thermal maturity threshold (H*) used for prediction
- `alpha`, `beta`: Fitted shape parameters used for this genotype's predictions

### Prediction Summary Output

When `--predict-plantings` is used, the script prints a detailed prediction summary:

```
============================================================
  PREDICTION SUMMARY
============================================================
  Total predictions: 261  |  Evaluated: 250  |  Failed (no convergence): 0

  Overall Metrics:
    MAE  = 8.4 days    (mean absolute error)
    RMSE = 11.4 days   (root mean square error)
    Bias = +3.8 days   (mean signed error)

  Accuracy Breakdown:
    Within  5 days:  115/250  ( 46.0%)
    Within 10 days:  168/250  ( 67.2%)
    Within 15 days:  215/250  ( 86.0%)

  Per-Planting Breakdown:
    Planting          N     MAE    RMSE     Bias   Within5
    Planting 5      182    8.2d   11.7d    +3.3d    49.0%
    Planting 6       79    8.8d   10.7d    +5.0d    37.0%
============================================================
```

### Understanding Prediction Metrics

**MAE (Mean Absolute Error):**
The average of `|error|` across all predictions. **The most intuitive metric** — "on average, predictions miss by X days."

**RMSE (Root Mean Square Error):**
Like MAE, but squares the errors before averaging, then takes the square root. Large errors are penalized disproportionately.

**Bias (Mean Signed Error):**
The average of raw errors (not absolute values). Measures average tendency, not the size of individual misses.

### What Makes Good Results?

> **Note:** These thresholds are practical guidelines, not formally validated benchmarks.

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| MAE | < 5 days | 5-10 days | 10-15 days | > 15 days |
| RMSE | < 7 days | 7-12 days | 12-20 days | > 20 days |
| Bias | < \|2\| days | \|2\|-\|5\| days | \|5\|-\|10\| days | > \|10\| days |
| Within 5 days | > 60% | 40-60% | 25-40% | < 25% |

**Common causes of poor predictions:**
- Training on warm-season only, predicting cold-season (temperature regime mismatch)
- Too few training plantings (need 3+ for reliable H* estimation)
- Parameters at bounds — optimizer couldn't find good fit
- Genotype has censored observations in training plantings

### Progress File

**File**: `diagnostics/progress.csv`

```csv
id,planting,cumulative_progress
FA002,Planting 1,0.9823
FA002,Planting 2,1.0234
```

- Values should be similar across plantings for same genotype
- Variance in these values = mechanistic loss

### GBLUP DAP Prediction Results

**File**: `predictions/gblup.csv`

```csv
id,planting,observed_dap,predicted_dap,error,breeding_value,pred_var
FA002,Planting 5,52,51,-1.0,3.45,5.2
FA002,Planting 6,39,40,1.0,3.45,5.2
```

### RN-GBLUP Prediction Results

**File**: `predictions/jarquin.csv` — includes `pred_var` column (conditional variance from G + E + GxE kernels).

### Joint Prediction Results

**File**: `predictions/joint.csv` — Joint per-genotype Theta/S/Pc baked into the prediction.

## Common Scenarios

### Scenario 1: New Dataset - First Analysis

```bash
# 1. Verify data loads correctly with one genotype
venv/bin/python3 -m cgm_wgp.main_fit \
  --phenotypes my_phenotypes.csv \
  --weather pipeline/1_data/weather/MY_LOCATION/weather.csv \
  --genotype GENO001 \
  --crop my_crop \
  --maxiter 500 \
  --debug

# 2. If successful, fit all genotypes (no GBLUP yet)
venv/bin/python3 -m cgm_wgp.main_fit \
  --phenotypes my_phenotypes.csv \
  --weather pipeline/1_data/weather/MY_LOCATION/weather.csv \
  --crop my_crop \
  --maxiter 1000
```

### Scenario 2: Production Run (4 Methods)

```bash
# Run full pipeline with Joint enabled
venv/bin/python3 -m cgm_wgp.app \
  --maxiter 1000 \
  --crop Broccoli \
  --phenotypes pipeline/1_data/Broccoli/phenotypes_dated.csv \
  --weather pipeline/1_data/weather/FL_Hasting/weather.csv \
  --train-plantings 1,2,3,4 \
  --predict-plantings 5,6 \
  --run-joint

# Check predictions in the structured output folder:
# params/fitted.csv                          (Mechanistic alpha, beta)
# params/joint_params.csv                    (Joint Theta, S, Pc)
# predictions/mechanistic.csv                (Mechanistic, dual-threshold combiner)
# predictions/raw_mechanistic.csv            (Mechanistic, multiplicative coupling)
# predictions/gblup.csv                      (GBLUP DAP + PEV)
# predictions/jarquin.csv                    (RN-GBLUP + conditional variance)
# predictions/joint.csv                      (CGM-WGP Joint)
# diagnostics/validation_comparison.csv      (all four methods compared)
# meta.json                                  (run metadata, metrics, tags)
```

### Scenario 3: Quick Evaluation (No GBLUP)

```bash
# Mechanistic-only train/predict
venv/bin/python3 -m cgm_wgp.main_fit \
  --phenotypes pipeline/1_data/Broccoli/phenotypes_dated.csv \
  --weather pipeline/1_data/weather/FL_Hasting/weather.csv \
  --crop Broccoli \
  --train-plantings 1,2,3,4 \
  --predict-plantings 5,6 \
  --no-gblup-dap \
  --maxiter 1000
```

### Scenario 4: CV Run with Joint

```bash
venv/bin/python3 -m cgm_wgp.app \
  --config configs/beans.yaml \
  --cv-scheme cv00_double_novelty \
  --run-joint \
  --maxiter 1000
```

## Tips

### Data Preparation
- Phenotype dates must overlap weather data availability
- Check for missing values in flowering time (`ft` column)
- Verify `censored` column correctly indicates observed (0) vs censored (1)
- Use consistent date formats (YYYY-MM-DD)
- Place weather files in `pipeline/1_data/weather/{LOCATION}/weather.csv` (shared directory)

### Optimization
- Start with low `maxiter` (200-500) for testing
- Use `--genotype` to test individual fits before full runs
- Use `maxiter` 1000-3000 for production
- Check convergence via `nit` and `nfev` columns

### Train/Predict
- Start with an even train/predict split (e.g., 3 train, 3 predict)
- More training plantings give better parameter estimates
- Check prediction accuracy across different train/predict splits

### Output Organization
- Each run creates a structured timestamped folder under `pipeline/output/runs/{crop}/{location}/`
- A `latest` symlink points to the most recent run for each crop/location
- Use `--run-label` to attach a label to a run
- Old runs are preserved and never overwritten
- Use `python -m cgm_wgp.runs` to list, tag, label, archive, or backfill runs
- Compare across runs via the global `pipeline/output/index.json` registry

## Structured Output System

Pipeline runs use a structured directory layout under `pipeline/output/runs/`. Each run is self-contained, with subdirectories for parameters, predictions, diagnostics, and plots.

### Directory Layout

```
pipeline/output/
├── runs/
│   └── {crop}/
│       └── {location-or-cv-scheme}/
│           ├── latest -> 2026-03-17_12-04-52/   (symlink to most recent run)
│           ├── 2026-03-17_12-04-52/
│           │   ├── meta.json
│           │   ├── params/
│           │   │   ├── fitted.csv
│           │   │   └── joint_params.csv
│           │   ├── predictions/
│           │   │   ├── mechanistic.csv
│           │   │   ├── raw_mechanistic.csv
│           │   │   ├── gblup.csv
│           │   │   ├── jarquin.csv
│           │   │   └── joint.csv
│           │   ├── diagnostics/
│           │   │   ├── metrics.csv
│           │   │   ├── progress.csv
│           │   │   ├── validation_comparison.csv
│           │   │   ├── run_config.json
│           │   │   └── debug_daily_rates.csv
│           │   └── plots/
│           │       └── (per-run diagnostic PNGs)
│           └── 2026-03-16_15-50-37/
│               └── ...
└── index.json                (global registry of all runs)
```

### Run Metadata (`meta.json`)

Each run directory contains a `meta.json` file at the root with run metadata:

```json
{
  "crop": "Broccoli",
  "location": "FL_Hasting",
  "created": "2026-03-17T12:04:52",
  "label": "CV1 baseline",
  "tags": ["production"],
  "git_sha": "6ef162b",
  "metrics": {
    "joint_mae": 6.86,
    "n_genotypes": 158
  }
}
```

### Run CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--run-label TEXT` | none | Human-readable label for this run (stored in `meta.json`) |
| `--run-dir PATH` | auto-generated | Explicit run directory (used internally by `app.py`; not normally needed) |

### Global Run Index (`index.json`)

`pipeline/output/index.json` is a JSON array of all indexed runs, updated after each pipeline execution. Used for filtering and comparison without traversing the directory tree.

### Run Management Utility (`cgm_wgp.runs`)

CLI utility for managing pipeline runs:

```bash
# List all runs (filterable)
python -m cgm_wgp.runs list
python -m cgm_wgp.runs list --crop Broccoli
python -m cgm_wgp.runs list --tag production
python -m cgm_wgp.runs list --label "CV1 baseline"

# Tag a run
python -m cgm_wgp.runs tag pipeline/output/runs/Broccoli/FL_Hasting/2026-03-17_12-04-52 production v2

# Set a label
python -m cgm_wgp.runs label pipeline/output/runs/Broccoli/FL_Hasting/2026-03-17_12-04-52 "Final production run"

# Archive old runs (moves to pipeline/output/archive/)
python -m cgm_wgp.runs archive --before 2026-03-01

# Rebuild index.json from existing run directories
python -m cgm_wgp.runs backfill
```

### The `latest` Symlink

Under each `{crop}/{location}/` directory, `latest` points to the most recent run. Stable path for scripts:

```bash
# Always points to the newest Broccoli run
cat pipeline/output/runs/Broccoli/FL_Hasting/latest/diagnostics/validation_comparison.csv
```

---

For algorithm details see [TECHNICAL.md](TECHNICAL.md). For the data file
spec see [DATA_FORMAT.md](DATA_FORMAT.md).
