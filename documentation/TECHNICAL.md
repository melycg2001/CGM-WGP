# Technical Documentation - CGM-WGP

Algorithms, mathematics, and implementation details of the four-method
CGM-WGP framework: Mechanistic, GBLUP, RN-GBLUP, and CGM-WGP.

## Table of Contents

1. [Mathematical Framework](#mathematical-framework)
2. [Mechanistic Model](#mechanistic-model)
3. [Additive Photoperiod Coupling (Hadley)](#additive-photoperiod-coupling-hadley)
4. [GBLUP](#gblup)
5. [RN-GBLUP (Jarquin 2014)](#rn-gblup-jarquin-2014)
6. [Joint CGM-WGP (Messina/Technow)](#joint-cgm-wgp-messinatechnow)
7. [CV Schemes](#cv-schemes)
8. [Optimization](#optimization)
9. [Genomic Data Pipeline](#genomic-data-pipeline)
10. [Implementation Details](#implementation-details)
11. [Structured Output System](#structured-output-system)
13. [References](#references)

## Mathematical Framework

### Cardinal Beta Temperature Response

The cardinal beta function models the relationship between temperature and developmental rate:

```
           { 0                                                    if T <= Tb or T >= Tc
rate(T) = {
           { [(T - Tb)/(Topt - Tb)]^alpha x [(Tc - T)/(Tc - Topt)]^beta  if Tb < T < Tc
```

**Parameters:**
- T: Daily mean temperature (C)
- Tb: Base temperature (no development below)
- Topt: Optimal temperature (maximum rate)
- Tc: Critical temperature (no development above)
- alpha, beta: Shape parameters

**Shape parameter effects:**

**Alpha:**
- Controls the rise of the curve from Tb to Topt
- alpha < 1: Concave curve (slow initial rise)
- alpha = 1: Linear rise
- alpha > 1: Convex curve (rapid initial rise)

**Beta:**
- Controls the decline from Topt to Tc
- beta < 1: Gradual decline
- beta = 1: Linear decline
- beta > 1: Steep decline

### Daylength Computation

When the weather CSV does not contain a `DAYLhr` column, daylength is computed from latitude and day of year using the Spencer (1971) astronomical equations:

```
theta = 2 * pi * (DOY - 1) / 365

delta = 0.006918
      - 0.399912 * cos(theta) + 0.070257 * sin(theta)
      - 0.006758 * cos(2*theta) + 0.000907 * sin(2*theta)
      - 0.002697 * cos(3*theta) + 0.00148 * sin(3*theta)

omega_s = arccos(-tan(lat_rad) * tan(delta))

daylength = (24 / pi) * omega_s
```

Latitude is provided via the `locations` section of the YAML config or via
`--latitude-map`. If the weather CSV contains a `DAYLhr` column, that column
is used directly.

## Mechanistic Model

The Mechanistic method has two stages: a temperature-only fit of per-genotype
(α, β), and a prediction step that combines the fitted thermal model with a
population-level photoperiod fit. Two prediction variants ship side by side:
`predictions/mechanistic.csv` (dual-threshold combiner) and
`predictions/raw_mechanistic.csv` (multiplicative coupling).

### Stage 1 — Temperature-only (α, β) fit

For genotype g in planting e with observed flowering at DAP_e:

```
H_{g,e} = sum(d=0 to DAP_e-1) cardinal_beta(T_{e,d}, alpha, beta, Tb, Topt, Tc)
```

Per-genotype (α, β) are fitted by minimising variance in cumulative cardinal-β
progress at observed flowering across the genotype's training plantings:

```
L_mech(alpha, beta; g) = sum(e in plantings of g) [H_{g,e}(alpha,beta) - H_bar_g]^2

H_bar_g = (1/n_e) sum(e) H_{g,e}
```

Solved by `scipy.optimize.dual_annealing` over (α, β) ∈ alpha_bounds × beta_bounds.
The fit uses cardinal_beta(T) only — photoperiod is **not** in the loss.
`mech_loss_for_genotype` in `src/cgm_wgp/mech_fit.py` is the entry point.

### Stage 2 — Population photoperiod fit (`fit_photo_only`)

After the (α, β) fit, a separate diagnostic pipeline fits population-level
photoperiod shape parameters and a per-genotype photoperiod threshold from
pure-photoperiod predictions of flowering DAP (no temperature). The default
shape on Beans + Broccoli is `logistic3`:

```
F_photo(DL; a, b, d) = 1 / (1 + a * exp(b * (DL - d)))
```

with population-level `(a, b, d)` fitted by `dual_annealing` and a per-genotype
threshold `H_photo_g` derived as the mean of `cum(F_photo(DL))` at the observed
flowering days across the genotype's training plantings.

### Stage 3 — Prediction (dual-threshold combiner, default)

When the crop config sets `photoperiod.coupling: dual_threshold` (the default
for Beans and Broccoli), `predictions/mechanistic.csv` is written using two
independent state variables and a `max()` combiner:

```
day_thermal_g(e) = first day d where cum(cardinal_beta(T_{e,d})) >= H_thermal_g
day_photo_g(e)   = first day d where cum(F_photo(DL_{e,d}))    >= H_photo_g
predicted_DAP    = max(day_thermal_g(e), day_photo_g(e))
```

where `H_thermal_g` is the per-genotype mean of cumulative cardinal-β at
observed DAP across training plantings, and `H_photo_g` is the per-genotype
photoperiod threshold from Stage 2. A plant only flowers when **both**
developmental clocks have completed. The dual-threshold path is wired in
`main_fit.py:4334-4445`.

`--mech-dual-threshold` is the CLI flag; the YAML config option
`photoperiod.coupling: dual_threshold` auto-enables it via `app.py:391-397`.

### Prediction variant: multiplicative coupling (`raw_mechanistic.csv`)

`predictions/raw_mechanistic.csv` is a separate prediction track that uses
multiplicative coupling instead of dual-threshold. It refits (α, β) with one
restart (no GBLUP penalty), then predicts via:

```
rate(t) = cardinal_beta(T_t, alpha, beta, Tb, Topt, Tc) * F_photo(DL_t; a, b, d)
predicted_DAP = first day d where cum(rate) >= H_g
```

That is, photoperiod modulates the thermal rate day-by-day instead of acting
as a separate clock. Implementation: `daily_development_rate` in `mech_fit.py:741`,
which returns `cardinal_beta(T) * photo(P)` when daylengths and photo params
are supplied. The raw_mechanistic block is wired in `main_fit.py:4941-5193`.

### Pure cardinal-β (opt-in)

To run the Mechanistic method with no photoperiod at all (cardinal-β only,
in both the fit and the prediction), pass `--mech-no-photo` to
`python -m cgm_wgp.app`. This skips Stage 2 and the dual-threshold/multiplicative
recompute; predictions use `cum(cardinal_beta(T))` reaching the per-genotype
thermal threshold:

```
predicted_DAP = first day d where cum(cardinal_beta(T_t)) >= H_thermal_g
```

This mode is useful as a diagnostic for separating thermal-only performance
from the photoperiod contribution; it is not the default for the shipped
Beans + Broccoli configs.

### Implementation

```python
def predict_flowering_dap(temps, target_H, alpha, beta,
                          temp_base, temp_optimal, temp_critical,
                          daylengths=None, photo_a=None, photo_b=None,
                          photo_d=None, photo_model="logistic3", ...):
    rates = daily_development_rate(
        temps, alpha, beta, temp_base, temp_optimal, temp_critical,
        daylengths=daylengths, photo_a=photo_a, photo_b=photo_b,
        photo_d=photo_d, photo_model=photo_model,
    )
    cumsum = np.cumsum(rates)
    indices = np.where(cumsum >= target_H)[0]
    return int(indices[0]) + 1 if len(indices) > 0 else -1
```

`daily_development_rate` returns `cardinal_beta(T) * F_photo(P)` when photo
parameters are supplied and pure `cardinal_beta(T)` otherwise.

## Additive Photoperiod Coupling (Hadley)

The Joint CGM-WGP method couples temperature and photoperiod through a Hadley
additive rate-penalty rather than the multiplicative `cardinal_beta(T) × g(P)`
form. This avoids a CV0 failure mode (multiplicative coupling can shut
development off entirely when photoperiod is unfavourable, missing 27% of
cells on beans CV0).

### Forward Model

For genotype `g` in environment `e` with daily cardinal-beta thermal rates
`T_t` and daylengths `DL_t`:

```
short_day:  deviation_t = max(0, DL_t − Pc_g)      # penalty on long days
long_day:   deviation_t = max(0, Pc_g − DL_t)      # penalty on short days
base_rate_t = max(0, T_t − S_g * deviation_t)
daily_rate_t = base_rate_t / Theta_g
flowering when sum(daily_rate) >= 1
```

- `photo_direction` (config: `joint_fit.photo_direction`) selects which side of `Pc_g` is penalised
- `S_g >= 0` always — sign of response is encoded in the direction, not the parameter
- alpha, beta (cardinal-β shape) are fixed at the population level from Tb/Topt/Tc:
  `alpha_fixed = (Topt − Tb) / (Tc − Topt)`, `beta_fixed = 1`

Per-genotype free parameters:
- **Theta_g** — thermal-time-to-flower threshold (cumulative cardinal-β rate, ~5–200)
- **S_g** — photoperiod rate-penalty per hour of deviation (~0–0.3)
- **Pc_g** — critical daylength in hours (~10–18)

Code: `forward_dap_fractional` in `src/cgm_wgp/joint_fit.py`.

## GBLUP

### Model

```
DAP_{g,e} = mu_e + u_g + epsilon_{g,e}
```

Where:
- DAP_{g,e}: Flowering time for genotype g in environment e
- mu_e: Environment (planting) mean — for GBLUP DAP, mu_e is estimated by regressing training planting means on mean temperature
- u_g: Genomic breeding value, u ~ N(0, G * sigma^2_g)
- epsilon: Residuals

### Breeding Value Estimation via Spectral Decomposition

The G-matrix is decomposed as `G = V D V'`. Breeding values are estimated from the eigenvalue decomposition (EVD):

```
u_hat = V diag(d_i / (d_i + lambda)) V' y_adj
```

Where:
- d_i: Eigenvalues of G
- lambda = sigma^2_e / sigma^2_g (variance ratio)
- y_adj: Adjusted phenotypes (DAP - planting mean), averaged per genotype

### REML Estimation of Lambda

The variance ratio lambda is estimated via Restricted Maximum Likelihood using the spectral decomposition:

```
z = V' y_adj    (rotated adjusted phenotypes)

L(lambda) = -0.5 sum_i [log(d_i + lambda) + z_i^2 / (d_i + lambda)]
```

Optimized using `scipy.optimize.minimize_scalar` (bounded method, lambda in [1e-6, 1e6]).

### Prediction Error Variance (PEV)

PEV quantifies uncertainty in genomic breeding value predictions:

```
PEV_i = sigma_g^2 * sum_k(v_ik^2 * lambda / (d_k + lambda))
```

PEV is reported in `predictions/gblup.csv` as the `pred_var` column.

**Implementation:** `src/cgm_wgp/gblup.py` performs REML and GBLUP in pure
Python (numpy/scipy).

## RN-GBLUP (Jarquin 2014)

The Reaction Norm GBLUP combines genomic, environmental, and GxE kernels:

```
y = mu + g + e + ge + epsilon
```

Where:
- g ~ N(0, G * sigma^2_g): genomic main effect, G is the genomic relationship matrix
- e ~ N(0, E_K * sigma^2_e): environmental main effect, E_K is the environmental kernel
- ge ~ N(0, (G # E_K) * sigma^2_ge): GxE interaction, # is the Hadamard product
- epsilon ~ N(0, sigma^2_eps): residual

### Kernel Construction

- **G**: genomic relationship matrix from VanRaden Method 1 (see [Genomic Data Pipeline](#genomic-data-pipeline))
- **E_K**: environmental kernel built from environmental covariates (means + day-of-year features per training environment)
- **G # E_K**: Hadamard product of G and E_K, blocked across (genotype, environment) cells

### Variance Component Estimation

Variance components (sigma^2_g, sigma^2_e, sigma^2_ge, sigma^2_eps) are
estimated via REML on the joint kernel structure.

### Conditional Variance

For a new genotype-environment combination, the conditional predictive variance is:

```
PEV = Var(y_new) - k_cross' V^{-1} k_cross
```

Where:
- Var(y_new): Prior variance of the new observation under the RN-GBLUP model
- k_cross: Cross-covariance vector between the new observation and training data
- V: Variance-covariance matrix of the training data (incorporating G, E_K, and GxE)

`pred_var` is reported in `predictions/jarquin.csv`.

## Joint CGM-WGP (Messina/Technow/Cooper)

The Joint method (`src/cgm_wgp/joint_fit.py`) fits all genotypes
simultaneously under an explicit multivariate-normal genomic prior,
integrating crop-growth model and whole-genome prediction in one optimization
(Messina et al. 2018; Technow et al. 2015; Cooper et al. 2016).

The current implementation uses Hadley-additive photoperiod coupling (see
[Additive Photoperiod Coupling (Hadley)](#additive-photoperiod-coupling-hadley))
rather than the multiplicative form in the original Messina paper.

### Genomic Prior (Multivariate Normal in Eigenspace)

Each parameter vector (Theta, S, Pc) is treated as a draw from a population MVN distribution with covariance `sigma^2_k · G`:

```
theta_k ~ N(mu_k · 1, sigma^2_k · G)
```

Working in the eigenspace `G = V D V'` makes the prior cost diagonal:

```
prior_k = (1/sigma^2_k) · sum_j z^2_{k,j} / d_j     with z_k = V' (theta_k − mu_k · 1)
logdet_k = G · log(sigma^2_k)
```

Total objective minimized at the inner step:

```
L(x) = SSE_data(x)  +  sum_k prior_k(x)  +  sum_k logdet_k(x)
```

Code: `compute_genomic_prior` in `src/cgm_wgp/joint_fit.py`.

### EM with L-BFGS-B Inner Loop

The full algorithm is REML-style EM; closed-form M-step on hyperparameters, gradient-based E-step on per-genotype values:

```
Initialize Theta_g, S_g, Pc_g per genotype (population means or warm-start)
Repeat until rel_change(loss) < em_tol or max_em_iters reached:
    INNER (E-step): minimize L(x) by L-BFGS-B with analytic gradients,
                    bounds = [Theta_bounds]*G + [S_bounds]*G + [Pc_bounds]*G
    OUTER (M-step): update population params closed-form via REML on V D V':
                    mu_k = mean(theta_k)
                    sigma^2_k = (1/G) · sum_j z^2_{k,j} / d_j     (with floor 1e-6)
```

Convergence: `|delta L| / |L| < em_tol` (default 1e-4). Defaults: `max_em_iters = 50`, `lbfgsb_maxiter = 1000`.

### Analytic Gradients

The crossing day `pred_dap = idx + (target − cumsum[idx-1]) / base_rate[idx]` yields closed-form gradients via the implicit function theorem at the crossing:

```
∂pred/∂Theta_g  = 1 / base_rate[idx]
∂pred/∂S_g      = −(sum_{t<=idx} ∂base_t/∂S_g) / base[idx]    (active region: base > 0 AND deviation > 0)
∂pred/∂Pc_g     = −(sum_{t<=idx} ∂base_t/∂Pc_g) / base[idx]
```

The `photo_direction` flip becomes a sign on `∂base/∂Pc`: +1 for short_day, −1 for long_day.

Gradients are validated against `scipy.optimize.check_grad` finite differences (relative error ≤ 1e-4 in unit tests).

### Held-Out Genotype Prediction (BLUP)

For genotypes never seen in training (CV1, CV00), Joint uses the spectral BLUP form to predict Theta from genomic relationships:

```
Theta_new = mu_Theta + G[new, train] · (G[train, train] + lambda*I)^-1 · (Theta_train − mu_Theta)
```

`G[train,train]` is reconstructed from the EVD as `V · diag(d) · V'` and inverted with a small ridge (`1e-4 · mean(diag(G_tt))`). For S and Pc, the population mean is used directly since per-genotype variance collapses (see "Heritability findings" below).

### Parameter Layout in the Optimization Vector

```
x = [Theta_1, …, Theta_G,  S_1, …, S_G,  Pc_1, …, Pc_G]
```

Total dimensionality: `3G` when photoperiod is enabled, `G` otherwise. For Beans (G=187), this is 561 free parameters per fit.

## CV Schemes

Cross-validation is driven by `--cv-scheme` on `python -m cgm_wgp.app`. The four paper schemes:

| Scheme | Holdout unit | Description |
|--------|--------------|-------------|
| `balanced_training_subset` | Planting subsets | GxE Sparse Testing within trained envs (CV2) |
| `new_varieties` | 20% genotypes | New Varieties in trained envs (CV1); whole-genotype holdout across all envs, 5 reps |
| `individual` | One environment | New Location / new environment (CV0); leave-one-environment-out |
| `cv00_double_novelty` | Env + 20% genotypes | New Program (CV00); joint novelty in env *and* genotype |

Each scheme runs the full four-method registry (Mechanistic, GBLUP, RN-GBLUP, CGM-WGP Joint). The CV partitioning logic lives in `src/cgm_wgp/cv_schemes.py`.

### Unseen Genotype Handling

Genotypes present in the G-matrix but with no phenotype data receive predictions from:

- **GBLUP / RN-GBLUP**: breeding values via genomic, environmental, and GxE relationships
- **Joint**: spectral BLUP for Theta; S and Pc use the population mean
- **Mechanistic**: G-matrix proxy — GBLUP of fitted genotypes' mechanistic DAP predictions

Minimum 5 fitted predictions per planting required for the Mechanistic proxy.

## Optimization

### Dual Annealing (Mechanistic)

The mechanistic loss is minimized using SciPy's `dual_annealing` optimizer:

```python
from scipy.optimize import dual_annealing

result = dual_annealing(
    func=objective_function,
    bounds=[(0.5, 8.0), (0.5, 8.0)],  # alpha and beta bounds
    maxiter=1000,
    initial_temp=5230.0,
    visit=2.62,
    accept=-5.0,
    minimizer_kwargs={
        'method': 'Nelder-Mead',
        'options': {
            'maxiter': 150,
            'xatol': 1e-4,
            'fatol': 1e-6,
            'adaptive': True
        }
    },
    seed=42
)
```

### Multi-Restart Strategy

When `n_restarts > 1`, the optimizer runs multiple times with different random seeds and keeps the best result:

```python
for restart_i in range(n_restarts):
    result = dual_annealing(func=objective, bounds=bounds,
                            seed=seed + restart_i, ...)
    if result.fun < best_loss:
        best_loss = result.fun
        best_result = result
```

This helps escape local minima on multimodal loss landscapes. Runtime scales linearly with `n_restarts`.

### L-BFGS-B (Joint)

The Joint inner E-step uses L-BFGS-B with analytic gradients and box bounds
over `(Theta, S, Pc)` for each genotype. The outer M-step updates population
hyperparameters in closed form via REML on the eigenspace.

### Why Dual Annealing for Mechanistic?

The cardinal beta loss landscape is multimodal (multiple local minima), nonsmooth (discrete daily time steps create small discontinuities), bounded (parameter range constraints), and genotype-specific. Dual annealing combines global exploration (simulated annealing) with local exploitation (Nelder-Mead).

### Convergence Criteria

Dual annealing terminates when:
1. **Function tolerance**: `|f_new - f_best| < fatol`
2. **Parameter tolerance**: `||x_new - x_best|| < xatol`
3. **Iteration limit**: `iterations >= maxiter`

Set `maxiter` high enough (1000-3000) for reliable convergence.

### Loss Normalization

By default, the mechanistic loss is `sum((H_e - H_bar)^2)` across training plantings. When `normalize_loss=True`, the loss is divided by the number of plantings:

```
L_normalized = sum((H_e - H_bar)^2) / n_plantings
```

Useful when genotypes have varying numbers of training observations.

## Genomic Data Pipeline

Genomic data enters at three tiers of pre-processing. Each tier produces the same final artifact: `EVD.rda` (eigenvalue decomposition of the genomic relationship matrix), consumed by all GBLUP-based methods.

```
Tier 1: Markers   ──> G-matrix (VanRaden Method 1) ──> EVD
Tier 2: G-matrix  ────────────────────────────────────> EVD
Tier 3: EVD       ──────────────────────────────────────(copy)
```

The tier is selected by the `genomics.tier` field in the crop config. Output files are stored in `pipeline/2_gmatrix/{crop}/`.

### Tier 1: Markers — VanRaden Method 1 G-matrix

When raw marker data is provided (`tier: markers`), the pipeline computes the genomic relationship matrix using VanRaden Method 1 (VanRaden, 2008).

**Implementation:** `src/cgm_wgp/prepare_pipeline.py:compute_vanraden_gmatrix()`

**Supported marker codings:**

| `marker_coding` | Raw values | Interpretation |
|-----------------|-----------|----------------|
| `biallelic_12` | 1, 2 | Homozygous AA / BB; converted to {0, 1} before computation |
| `biallelic_01` | 0, 1 | Homozygous AA / BB; used directly |
| `dosage` | 0, 1, 2 | Allele dosage (0 = AA, 1 = AB, 2 = BB) |

**Step 1 — Recoding.** For `biallelic_12` input, values are shifted by subtracting 1.

**Step 2 — Missing value imputation.** `NaN` entries are replaced with the column (marker) mean over non-missing individuals:

```
M[i, j] = mean(M[:, j])   for all i where M[i, j] is missing
```

**Step 3 — Allele frequency estimation.**

```
p_j = mean(M[:, j]) / 2
```

**Step 4 — Monomorphic marker removal.** Markers with `p_j < 0.001` or `p_j > 0.999` are removed.

**Step 5 — Centering.**

```
Z = M - 2p
```

**Step 6 — G-matrix computation.**

```
G = ZZ' / (2 * sum_j(p_j * (1 - p_j)))
```

The denominator scales G so that diagonal elements approximate `1 + f_i` (one plus the inbreeding coefficient).

### Tier 2: G-matrix

When a pre-computed G-matrix is provided (`tier: gmatrix`), the pipeline skips marker processing and computes the EVD directly. Source can be `.rds`, `.rda` (read via R subprocess), or CSV.

### Tier 3: EVD

When EVD is already available (`tier: evd`), the pipeline copies `EVD.rda` (and optionally `Gmatrix.rda`) to `pipeline/2_gmatrix/{crop}/`. No computation.

### EVD Generation

For tiers 1 and 2, the eigenvalue decomposition is computed by an R subprocess:

```r
EVD <- eigen(G)
rownames(EVD$vectors) <- genotype_ids
save(EVD, file = "pipeline/2_gmatrix/{crop}/EVD.rda")
```

`EVD.rda` contains `EVD$values` (eigenvalues) and `EVD$vectors` (eigenvectors with genotype ID rownames).

## Implementation Details

### Path Configuration (`src/cgm_wgp/config.py`)

All paths are centralized in `src/cgm_wgp/config.py`:

```python
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
DATA_DIR = PIPELINE_DIR / "1_data"
OUTPUT_DIR = PIPELINE_DIR / "output"  # Timestamped run folders
```

Key constants:
- `WEATHER_DIR` -> `pipeline/1_data/weather/` (shared across all crops)
- `RUN_SUBDIRS` -> `("params", "predictions", "diagnostics", "plots")`

Key helper functions:
- `create_run_dir(crop, location, label=None)` — creates `pipeline/output/runs/{crop}/{location}/{timestamp}/` with structured subdirs, `meta.json`, `latest` symlink, and index entry
- `get_output_path(run_dir, category, filename)` — returns the full path for an output file routed to the correct subdirectory
- `finalize_run(run_dir, metrics)` — updates `meta.json` with post-run metrics and refreshes the global index entry
- `resolve_run(crop, location, tag, label)` — looks up a run directory from `index.json`

### Data Structures

**Temperature dictionary (`T_dict`):**
```python
T_dict[genotype_id][planting_id] = pandas.Series(
    data=[temp1, temp2, ...],
    index=[date1, date2, ...]
)
```

Includes weather for ALL plantings (training, prediction, and censored).

**Multi-location weather:** When `weather_csv` is a directory, `load_and_build_dicts()` loads `{dir}/{planting_id}/weather.csv` per unique Planting value. Single-file mode (default) — all observations share one weather file.

**Observation dictionary (`DAP_obs_dict`):**
```python
DAP_obs_dict[genotype_id][planting_id] = days_to_flowering
```

Only includes uncensored observations.

### Cardinal Beta Implementation

```python
def cardinal_beta(temps, temp_base, temp_optimal, temp_critical, alpha, beta):
    t = np.asarray(temps, dtype=float)
    out = np.zeros_like(t)
    mask = (t > temp_base) & (t < temp_critical)
    if np.any(mask):
        x = (t[mask] - temp_base) / (temp_optimal - temp_base)
        y = (temp_critical - t[mask]) / (temp_critical - temp_optimal)
        out[mask] = (x ** alpha) * (y ** beta)
    return out
```

### Method Registry

The four methods are registered in `src/cgm_wgp/methods.py`:

```python
METHODS = {
    "Mechanistic": ...,
    "GBLUP": ...,
    "RN-GBLUP": ...,
    "CGM-WGP": ...,
}
```

`app.py` iterates over the active subset of this registry on each run.

## Structured Output System

Organizes run artifacts into a predictable directory hierarchy with machine-readable metadata.

### Directory Layout

Each call to `create_run_dir(crop, location, label=None)` creates:

```
pipeline/output/runs/{crop}/{location-or-cv-scheme}/{YYYY-MM-DD_HH-MM-SS}/
    meta.json
    params/
        fitted.csv                      # Mechanistic alpha, beta per genotype
        joint_params.csv                # CGM-WGP Joint per-genotype Theta, S, Pc
    predictions/
        mechanistic.csv                 # Mechanistic, dual-threshold combiner
        raw_mechanistic.csv             # Mechanistic, multiplicative coupling
        gblup.csv                       # GBLUP DAP with PEV
        jarquin.csv                     # RN-GBLUP with conditional variance
        joint.csv                       # CGM-WGP Joint
    diagnostics/
        metrics.csv                     # Per-planting accuracy metrics
        progress.csv                    # Cumulative progress H per planting
        validation_comparison.csv       # All four methods compared side-by-side
        run_config.json                 # Frozen run configuration
        debug_daily_rates.csv           # Daily rate debug output (with --debug)
    plots/
        *.png                           # Per-run diagnostic PNGs
```

### meta.json

Written at run creation with initial metadata:

```json
{
  "crop": "Broccoli",
  "location": "FL_Hasting",
  "created": "2026-03-21_14-30-00",
  "label": null,
  "tags": [],
  "git_sha": "abc1234",
  "metrics": {}
}
```

After run completion, `finalize_run(run_dir, metrics)` updates `metrics` (MAE, RMSE, etc.) and refreshes the corresponding entry in the global index.

### latest Symlink

Each `pipeline/output/runs/{crop}/{location}/` contains a `latest` symlink pointing to the most recent run's timestamp folder. Updated atomically by `create_run_dir()`. Scripts reference stable paths like `latest/predictions/joint.csv`.

### Global Index (index.json)

`pipeline/output/index.json` is a JSON array; each entry records one run:

```json
[
  {
    "crop": "Broccoli",
    "location": "FL_Hasting",
    "created": "2026-03-21_14-30-00",
    "label": "baseline",
    "tags": ["best"],
    "git_sha": "abc1234",
    "metrics": {"mech_mae": 8.2, "joint_mae": 6.9},
    "path": "runs/Broccoli/FL_Hasting/2026-03-21_14-30-00"
  }
]
```

`resolve_run(crop, location, tag, label)` queries this index and returns the `Path` to the most recent matching run directory.

### Lifecycle

1. Creation: `create_run_dir()` creates the directory tree, writes initial `meta.json`, updates `latest` symlink, appends to `index.json`.
2. During run: `get_output_path(run_dir, category, filename)` routes each output file to the correct subdirectory.
3. Finalization: `finalize_run(run_dir, metrics)` writes final metrics to `meta.json` and updates the index entry.

## References

**Cardinal Temperatures:**
- Yan W, Hunt LA (1999). An equation for modelling the temperature response of plants using only the cardinal temperatures. *Annals of Botany*.

**Photoperiod:**
- Hadley P, Roberts EH, Summerfield RJ, Minchin FR (1984). Effects of temperature and photoperiod on flowering in soya bean (*Glycine max* (L.) Merrill): a quantitative model. *Annals of Botany*, 53(5), 669–681.
- Spencer JW (1971). Fourier series representation of the position of the sun. *Search* 2(5).

**Dual Annealing:**
- Xiang Y, Sun DY, Fan W, Gong XG (1997). Generalized simulated annealing algorithm. *Physics Letters A*.

**GBLUP:**
- VanRaden PM (2008). Efficient methods to compute genomic predictions. *Journal of Dairy Science*.

**Reaction Norm GBLUP:**
- Jarquin D, Crossa J, Lacaze X, Du Cheyron P, Daucourt J, Lorgeou J, Piraux F, Guerreiro L, Perez P, Calus M, Burgueno J, de los Campos G (2014). A reaction norm model for genomic selection using high-dimensional genomic and environmental data. *Theoretical and Applied Genetics*.

**Joint CGM-WGP:**
- Messina CD, Technow F, Tang T, et al. (2018). Leveraging biological insight and environmental variation to improve phenotypic prediction: Integrating crop growth models (CGM) with whole genome prediction (WGP). *European Journal of Agronomy*, 100, 151–162.
- Technow F, Messina CD, Totir LR, Cooper M (2015). Integrating crop growth models with whole genome prediction through approximate Bayesian computation. *PLoS ONE*, 10(6), e0130855.

**NASA POWER:**
- Stackhouse PW et al. (2018). POWER Release 8 (with GIS Applications) Methodology. NASA Technical Report. https://power.larc.nasa.gov/

---

For practical usage examples, see [USAGE.md](USAGE.md).
