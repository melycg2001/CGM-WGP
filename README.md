# CGM-WGP: Crop Growth Model with Whole Genome Prediction

**Status:** prototype, active development.

A Crop Growth Model–Whole Genome Prediction (CGM-WGP) framework for
flowering time. Two workflows:

1. **CGM-WGP** — a per-genotype mechanistic model (cardinal-β
   temperature and photoperiod response) fitted jointly with a
   genomic prior via EM + L-BFGS-B (Messina et al. 2006, Technow
   et al. 2015, Cooper et al. 2016).
2. **RN-GBLUP** — a Reaction Norm GBLUP that predicts flowering DAP
   from genomic (G), environmental (E), and G×E kernels (Jarquín et al. 2014).

Both methods run side by side from a single command, sharing the same
data-prep tooling. Each predicts a per-genotype flowering DAP in
held-out environments. Method list, cross-validation schemes, and
algorithm math live under [documentation/](documentation/).

## Install

```bash
git clone https://github.com/melycg2001/CGM-WGP.git
cd CGM-WGP

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

R is only needed if you have to compute an EVD from a raw G-matrix:

```r
install.packages("AGHmatrix")
```

For data preparation, layout, and supported genomic tiers see
[documentation/DATA_FORMAT.md](documentation/DATA_FORMAT.md).

## Running both workflows

One command runs the full pipeline for a crop and writes per-method
predictions to one timestamped run directory:

```bash
venv/bin/python -m cgm_wgp.app --config configs/{crop}.yaml \
    --run-joint \
    --cv-scheme balanced_training_subset
```

Outputs land under:

```
pipeline/output/runs/{Crop}/{location-or-cv-scheme}/{YYYY-MM-DD_HH-MM-SS}/
├── meta.json
├── params/
│   ├── joint_params.csv          # CGM-WGP per-genotype (Θ, S, Pc)
│   └── fitted.csv                # mechanistic baseline α, β
├── predictions/
│   ├── joint.csv                 # CGM-WGP predicted vs observed DAP
│   ├── jarquin.csv               # RN-GBLUP predicted vs observed DAP
│   └── ...                       # other methods (see USAGE.md)
└── diagnostics/
    └── validation_comparison.csv # side-by-side metrics for all methods
```

`configs/beans.yaml` and `configs/broccoli.yaml` ship as working
examples. To add a new crop, see the auto-prep walkthrough in
[documentation/USAGE.md](documentation/USAGE.md).

## Workflow 1: CGM-WGP

CGM-WGP fits, per genotype, a thermal-time-to-flower Θ\_g, a photoperiod
rate-penalty S\_g, and a critical daylength P\_{c,g} under an MVN genomic
prior built from G⁻¹. Population-level cardinal temperatures and a
population α, β are fixed; the genotype-specific reaction comes from
(Θ, S, P\_c). Optimization alternates an EM-style M-step for the
variance components and an L-BFGS-B step for the per-genotype parameters.

CGM-WGP runs as part of the standard pipeline (config `joint_fit.enabled:
true`, or pass `--run-joint`) alongside the Mechanistic, GBLUP, and
RN-GBLUP baselines:

```bash
venv/bin/python -m cgm_wgp.app --config configs/{crop}.yaml \
    --run-joint \
    --cv-scheme balanced_training_subset
```

**Where to look:**

| File | What it contains |
|---|---|
| `predictions/joint.csv` | predicted vs observed DAP per (genotype, planting) |
| `params/joint_params.csv` | per-genotype `Theta`, `S`, `Pc`, plus `alpha_fixed`, `beta_fixed` |
| `diagnostics/validation_comparison.csv` | aggregate metrics (MAE, RMSE, Pearson r) |

Joint photoperiod can be toggled (`--joint-photo-enabled`) and the
photoperiod direction set (`--joint-photo-direction short_day|long_day`).
Per-method overrides live under `model.joint_fit` in the crop config — see
[documentation/TECHNICAL.md § Joint CGM-WGP](documentation/TECHNICAL.md).

## Workflow 2: RN-GBLUP

RN-GBLUP (Jarquín 2014) predicts DAP as

```
y = μ + g + e + ge + ε
```

with `g` from a genomic kernel (G), `e` from an environmental kernel
(E, built from mean temperature, photothermal time, etc.), and `ge`
from the Hadamard product. Variance components are fit by REML, and
predictions for held-out cells fall out of the BLUP equations with a
per-cell conditional variance.

RN-GBLUP runs by default in every pipeline invocation — the same command
you'd use for CGM-WGP also writes `predictions/jarquin.csv`:

```bash
venv/bin/python -m cgm_wgp.app --config configs/{crop}.yaml \
    --cv-scheme balanced_training_subset
```

**Where to look:**

| File | What it contains |
|---|---|
| `predictions/jarquin.csv` | predicted DAP plus per-cell `pred_var` |
| `diagnostics/validation_comparison.csv` | RN-GBLUP row alongside other methods |

To extend RN-GBLUP with a photoperiod kernel (`y = μ +
g + e + p + ge + gp + ε`), pass `--rn-photoperiod`. To turn RN-GBLUP off
during a run (e.g. when iterating on the mechanistic side), pass
`--no-gblup-dap`.

## Cross-validation modes

Both workflows support four hold-out schemes via `--cv-scheme`:
sparse testing (`balanced_training_subset`), unseen genotypes
(`new_varieties`), unseen environments (`individual`), and joint novelty
(`cv00_double_novelty`). Partition logic and recommended scheme for each
research question are documented in
[documentation/USAGE.md § Cross-Validation](documentation/USAGE.md) and
[documentation/TECHNICAL.md § Cross-Validation](documentation/TECHNICAL.md).

## Project layout

```
CGM-WGP/
├── configs/                 # per-crop YAML configs (beans, broccoli)
├── src/cgm_wgp/             # package: app.py, main_fit.py, methods.py,
│                            # joint_fit.py, jarquin_loo.py, gblup.py, …
├── pipeline/
│   ├── 1_data/{Crop}/       # per-crop phenotypes + raw genomics
│   ├── 1_data/weather/      # shared weather files
│   ├── 2_gmatrix/{Crop}/    # EVD.rda per crop
│   └── output/runs/{Crop}/… # timestamped run output
└── documentation/           # full docs
```

## Documentation

| Doc | When to read |
|---|---|
| [QUICKSTART.md](documentation/QUICKSTART.md) | 5-minute path from install to first run |
| [USAGE.md](documentation/USAGE.md) | CLI flags and CV schemes |
| [TECHNICAL.md](documentation/TECHNICAL.md) | Math: cardinal-β, Joint forward model, RN-GBLUP kernels |
| [DATA_FORMAT.md](documentation/DATA_FORMAT.md) | Phenotype, weather, genomic file specs |

## Citation

```
[Citation to be added — manuscript in preparation]
```

## License

[License to be determined]

---

Prototype software. CLI flags and file layout may shift before the first
tagged release.
