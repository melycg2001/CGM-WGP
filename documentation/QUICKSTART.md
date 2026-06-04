# Quick Start Guide - CGM-WGP

Get running with CGM-WGP in under 5 minutes.

## Prerequisites Check

```bash
# Check Python version (need 3.8+)
python --version

# Check required Python packages
python -c "import numpy, pandas, scipy; print('Python packages OK')"

# R is only needed for generate_evd.py (EVD from G-matrix)
# Rscript --version
```

## Installation

```bash
# Clone repository
git clone https://github.com/melycg2001/CGM-WGP.git
cd CGM-WGP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas scipy requests
```

## First Run in 5 Minutes

Run the full 4-method pipeline on Broccoli using the shipped config:

```bash
venv/bin/python3 -m cgm_wgp.app \
  --config configs/broccoli.yaml \
  --cv-scheme balanced_training_subset \
  --run-joint \
  --maxiter 30
```

Outputs land under `pipeline/output/runs/Broccoli/{cv-scheme}/{timestamp}/` with a
`latest` symlink for convenience. The `diagnostics/validation_comparison.csv`
file summarises all four methods side-by-side.

## Quick Test on a Single Genotype

Test the mechanistic fit on one genotype to verify data and setup:

```bash
venv/bin/python3 -m cgm_wgp.main_fit \
  --phenotypes pipeline/1_data/Broccoli/phenotypes_dated.csv \
  --weather pipeline/1_data/weather/FL_Hasting/weather.csv \
  --genotype FA002 \
  --crop broccoli \
  --maxiter 500
```

Output is written to a timestamped run folder:
```
pipeline/output/runs/broccoli/FL_Hasting/{timestamp}/params/fitted.csv
```

A `latest` symlink is also created at
`pipeline/output/runs/broccoli/FL_Hasting/latest`.

**Expected output format** (`params/fitted.csv`):
```csv
id,alpha,beta,loss,nit,nfev
FA002,3.988,3.999,118.76,500,2840
```

## Config-Driven Workflow

Configs live in `configs/` and bundle crop-specific defaults (cardinal
temperatures, phenotype paths, weather format, genomic tier), so you don't pass
them on every command. Example configs: `broccoli.yaml`, `beans.yaml`.

### Step 0a: Auto-prepare raw data (fastest for new crops)

Use `--prepare` to auto-extract raw data into pipeline-ready formats and
generate a complete config YAML in one step. Only generates files that don't
already exist:

```bash
venv/bin/python3 -m cgm_wgp.inspect_data --crop Beans --prepare \
  --cardinal-temps '{"Tb": 8, "Topt": 25, "Tc": 38}'
```

This scans all raw files in `pipeline/1_data/Beans/`, generates
`phenotypes_dated.csv`, per-environment weather CSVs (with Fahrenheit
auto-conversion and NASA POWER fetch when needed), and writes
`configs/beans.yaml`. Review the config, then continue to Step 1.

### Step 0b: Inspect raw data (manual config approach)

Scan raw data files to detect formats and column mappings. Generates
a draft YAML config with TODO comments:

```bash
venv/bin/python3 -m cgm_wgp.inspect_data --crop MyCrop --generate-config
```

Review the generated `configs/mycrop.yaml` and fill in any TODO comments
(cardinal temperatures, missing planting dates, etc.).

### Step 1: Prepare data

`prepare_pipeline.py` reads the config and converts raw phenotype, weather, and
genomic data into the pipeline-ready format:

```bash
venv/bin/python3 -m cgm_wgp.prepare_pipeline --config configs/beans.yaml
```

This creates `phenotypes_dated.csv`, per-location weather files, and EVD from
whatever genomic tier is specified in the config (markers, gmatrix, or evd).

Use `--force` to overwrite existing files, or `--validate-only` to check that
all required pipeline data is already in place.

### Step 2: Run the pipeline

```bash
venv/bin/python3 -m cgm_wgp.app --config configs/beans.yaml --run-joint
```

The `--config` flag populates `--crop`, `--Tb`, `--Topt`, `--Tc`,
`--phenotypes`, `--weather`, and `--evd-path` automatically. CLI overrides
take precedence.

### Genomic tiers

Configs specify one of three genomic tiers under `genomics.tier`:

| Tier | Input | What `prepare_pipeline.py` does |
|------|-------|--------------------------------|
| `markers` | Raw marker matrix (Excel/CSV) | Computes G-matrix, then EVD |
| `gmatrix` | G-matrix file (.rds) | Computes EVD |
| `evd` | Pre-computed EVD.rda | Validates / copies into place |

## Common Commands

### Fit All Genotypes (Mechanistic Only)

```bash
# Using a config:
venv/bin/python3 -m cgm_wgp.main_fit \
  --config configs/broccoli.yaml --maxiter 1000

# Or with explicit flags:
venv/bin/python3 -m cgm_wgp.main_fit \
  --phenotypes pipeline/1_data/Broccoli/phenotypes_dated.csv \
  --weather pipeline/1_data/weather/FL_Hasting/weather.csv \
  --crop broccoli \
  --maxiter 1000
```

### Train on Specific Plantings, Predict Others

```bash
venv/bin/python3 -m cgm_wgp.main_fit \
  --phenotypes pipeline/1_data/Broccoli/phenotypes_dated.csv \
  --weather pipeline/1_data/weather/FL_Hasting/weather.csv \
  --crop broccoli \
  --train-plantings "1,4" \
  --predict-plantings "2,3,5,6" \
  --maxiter 1000
```

### Run Full Pipeline (4 Methods)

```bash
# Using a config (recommended):
venv/bin/python3 -m cgm_wgp.app --config configs/broccoli.yaml --run-joint

# Or with explicit flags:
venv/bin/python3 -m cgm_wgp.app \
  --maxiter 1000 \
  --crop broccoli \
  --phenotypes pipeline/1_data/Broccoli/phenotypes_dated.csv \
  --weather pipeline/1_data/weather/FL_Hasting/weather.csv \
  --run-joint
```

### CV Schemes

`--cv-scheme` selects the four-scheme CV partition logic:

| Scheme | Description |
|--------|-------------|
| `balanced_training_subset` | GxE Sparse Testing (CV2) within trained envs |
| `new_varieties` | New Varieties (CV1): 20% genotype holdout across all envs |
| `individual` | New Location (CV0): leave-one-environment-out |
| `cv00_double_novelty` | New Program (CV00): held-out env AND 20% genotypes |

```bash
venv/bin/python3 -m cgm_wgp.app \
  --config configs/beans.yaml \
  --cv-scheme new_varieties \
  --run-joint
```

### Debug Single Genotype

```bash
venv/bin/python3 -m cgm_wgp.main_fit \
  --phenotypes pipeline/1_data/Broccoli/phenotypes_dated.csv \
  --weather pipeline/1_data/weather/FL_Hasting/weather.csv \
  --genotype FA002 \
  --crop broccoli \
  --debug \
  --maxiter 500
```

## File Locations

### Input Files
- `pipeline/1_data/{Crop}/phenotypes_dated.csv` - Flowering time observations
- `pipeline/1_data/weather/{LOCATION}/weather.csv` - Daily temperature data (shared, organized by location)
- `pipeline/2_gmatrix/{crop}/` - Genomic relationship matrix (for GBLUP)

### Output Files (structured run folder)

All outputs go to `pipeline/output/runs/{crop}/{location-or-cv-scheme}/{timestamp}/` with
structured subdirectories. A `latest` symlink always points to the most recent run.
You can also tag runs with `--run-label "my label"`.

```
pipeline/output/runs/{crop}/{location-or-cv-scheme}/{timestamp}/
  meta.json                          # Run metadata (timestamp, label, config hash)
  params/
    fitted.csv                       # Mechanistic alpha, beta, loss, nit, nfev
    joint_params.csv                 # CGM-WGP Joint per-genotype Theta, S, Pc
  predictions/
    mechanistic.csv                  # Mechanistic predictions, dual-threshold combiner
    raw_mechanistic.csv              # Mechanistic predictions, multiplicative coupling
    gblup.csv                        # GBLUP DAP predictions with pred_var / PEV
    jarquin.csv                      # RN-GBLUP (Jarquin 2014) predictions with pred_var
    joint.csv                        # CGM-WGP Joint predictions
  diagnostics/
    metrics.csv                      # Per-planting accuracy: MAE, RMSE, bias, within-N
    progress.csv                     # Cumulative progress by planting
    validation_comparison.csv        # All four methods compared side-by-side
    run_config.json                  # Run configuration snapshot for reproducibility
  plots/                             # Per-run diagnostic PNGs (auto-generated)
```

See [DATA_FORMAT.md](DATA_FORMAT.md) for the full output catalog.

## Validation Comparison

When `--predict-plantings` (or `--cv-scheme`) is used, the framework runs all
four methods and writes a side-by-side table to
`diagnostics/validation_comparison.csv`.

The four-method registry (see `src/cgm_wgp/methods.py`):

| Method | Description |
|--------|-------------|
| Mechanistic | Per-genotype (α, β) from temp-only cardinal-β fit; predictions combine the thermal fit with a population-level photoperiod fit (dual-threshold by default, or multiplicative in `raw_mechanistic.csv`). Pass `--mech-no-photo` for cardinal-β-only predictions. |
| GBLUP | EnvCov GBLUP (REML + BLUP on EVD), per-genotype PEV |
| RN-GBLUP | Jarquin (2014) Reaction Norm: G + E_K + GxE kernels + conditional variance |
| CGM-WGP | EM + L-BFGS-B per-genotype (Theta, S, Pc) under MVN genomic prior with Hadley additive photoperiod coupling (Messina et al. 2006, Technow et al. 2015, Cooper et al. 2016). Enable with `--run-joint` |

**Unseen genotype predictions (CV1, CV00):** All methods predict for G-matrix
genotypes with no phenotype data via genomic relationships. No extra
configuration needed.

## Parameter Reference

### Essential Parameters

| Parameter | Default | Description | When to Change |
|-----------|---------|-------------|----------------|
| `--maxiter` | 1000 | Dual annealing iterations | Increase to 2000-3000 for production |
| `--Tb` | 5.0 | Base temperature (C) | Adjust for crop type |
| `--Topt` | 20.0 | Optimal temperature (C) | Adjust for crop type |
| `--Tc` | 35.0 | Critical temperature (C) | Adjust for crop type |
| `--crop` | "unknown" | Crop name for output folder | Set to your crop name |
| `--run-label` | None | Human-readable label stored in `meta.json` | Tag runs for later reference |
| `--train-plantings` | all | Plantings to train on | Set to subset (e.g. "1,2") |
| `--predict-plantings` | none | Plantings to predict | Set to held-out plantings |
| `--cv-scheme` | none | CV partition: `balanced_training_subset`, `new_varieties`, `individual`, `cv00_double_novelty` | Set for cross-validation runs |
| `--run-joint` | off | Enable CGM-WGP Joint alongside baselines | Recommended for CV0 / CV00 |
| `--alpha-bounds` | 0.5 8.0 | Alpha parameter search range | Widen if many genotypes hit bounds |
| `--beta-bounds` | 0.5 8.0 | Beta parameter search range | Widen if many genotypes hit bounds |
| `--n-restarts` | 1 | Random restarts for optimization | Increase to 3-5 for better convergence |

Joint CGM-WGP uses Hadley additive photoperiod coupling internally — see
[TECHNICAL.md](TECHNICAL.md) for the forward-model details. Photoperiod
hyperparameters (`Theta_bounds`, `S_bounds`, `Pc_bounds`, `photo_direction`)
live under `model.joint_fit` in the YAML config.

### Cardinal Temperatures (paper crops)

| Crop | Tb | Topt | Tc |
|------|-------|--------|-------|
| Broccoli | 5.0 | 25.0 | 40.0 |
| Beans | 10.0 | 25.0 | 38.0 |

## Troubleshooting Quick Fixes

### Error: "No rows left after filtering"
```bash
# Check your phenotype file has uncensored observations
venv/bin/python3 -c "import pandas as pd; df = pd.read_csv('pipeline/1_data/Broccoli/phenotypes_dated.csv'); print((df['censored']==0).sum(), 'uncensored rows')"
```

### Error: "No weather found"
```bash
# Verify date coverage
venv/bin/python3 -c "
import pandas as pd
pheno = pd.read_csv('pipeline/1_data/Broccoli/phenotypes_dated.csv')
weather = pd.read_csv('pipeline/1_data/weather/FL_Hasting/weather.csv')
print('Pheno dates:', pheno['Start_Date'].min(), 'to', pheno['End_Date'].max())
print('Weather dates:', weather['Period'].min(), 'to', weather['Period'].max())
"
```

### Optimizer not converging
```bash
# Increase maxiter
venv/bin/python3 -m cgm_wgp.main_fit \
  --phenotypes pipeline/1_data/Broccoli/phenotypes_dated.csv \
  --weather pipeline/1_data/weather/FL_Hasting/weather.csv \
  --crop broccoli \
  --maxiter 3000
```

### Parameters hitting bounds (alpha or beta = 7.999)
```bash
# Check data quality for that genotype
venv/bin/python3 -m cgm_wgp.main_fit \
  --phenotypes pipeline/1_data/Broccoli/phenotypes_dated.csv \
  --weather pipeline/1_data/weather/FL_Hasting/weather.csv \
  --genotype PROBLEM_GENO \
  --crop broccoli \
  --debug \
  --maxiter 2000
```

## Adding a New Crop

### Config-driven approach (recommended)

Create a YAML config in `configs/` (see `configs/beans.yaml` for a full
example), then run the two-step workflow:

```bash
# 1. Prepare all pipeline data from the config
venv/bin/python3 -m cgm_wgp.prepare_pipeline --config configs/mycrop.yaml

# 2. Run the pipeline
venv/bin/python3 -m cgm_wgp.app --config configs/mycrop.yaml --run-joint
```

### Manual approach

```bash
# 1. Convert raw phenotype data to pipeline format
venv/bin/python3 -m cgm_wgp.prepare_crop_data \
  --input pipeline/1_data/Beans/raw_phenotypes.csv \
  --crop Beans \
  --genotype-col RIL --location-col Site --ft-col DTF \
  --planting-dates "CIT=2018-11-15,ND=2019-05-25,..." \
  --location-map "CIT=CIT,ND=ND,..."

# 2. Generate EVD from G-matrix
venv/bin/python3 -m cgm_wgp.generate_evd \
  --gmatrix pipeline/1_data/Beans/Gmatrix.rds \
  --output pipeline/2_gmatrix/Beans/EVD.rda --save-gmatrix

# 3. Weather is shared under pipeline/1_data/weather/{location}/

# 4. Run with multi-location weather (pass directory, not file)
venv/bin/python3 -m cgm_wgp.main_fit \
  --phenotypes pipeline/1_data/Beans/phenotypes_dated.csv \
  --weather pipeline/1_data/weather/ \
  --crop Beans --Tb 8 --Topt 25 --Tc 38 --maxiter 1000
```

## Next Steps

- **Full documentation**: See [../README.md](../README.md)
- **Detailed usage**: See [USAGE.md](USAGE.md)
- **Technical details**: See [TECHNICAL.md](TECHNICAL.md)
- **Data formats**: See [DATA_FORMAT.md](DATA_FORMAT.md)

## Getting Help

1. Check documentation files (README, USAGE, TECHNICAL)
2. Use `--debug` flag to inspect calculations
3. Open issue on GitHub: https://github.com/melycg2001/CGM-WGP/issues

## Quick Tips

**DO:**
- Start with low `maxiter` (200-500) for testing
- Use `--genotype` flag to test single genotypes first
- Use `--train-plantings` to limit training for quick tests
- Check data quality before full runs

**DON'T:**
- Skip testing on single genotypes
- Use default cardinal temps for very different crops
- Enable Joint photoperiod on single-latitude data — the optimizer cannot separate temperature and daylength effects at one latitude (needs multi-latitude data)
