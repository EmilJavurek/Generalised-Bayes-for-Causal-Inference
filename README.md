# Generalized Bayes for Causal Inference

Research code for generalized Bayesian inference of average treatment effects (ATE) and conditional average treatment effects (CATE). The experiments compare causal pseudo-outcomes, learning-rate calibration, prior information, and nuisance estimators across synthetic settings and the IHDP benchmark.

## Setup and first run

Use Python 3.13 and [uv](https://docs.astral.sh/uv/). Run commands from the repository root:

```bash
uv sync --locked
uv run --locked python -m experiments.ate
uv run --locked python -m reporting.figures ate
```

Dependencies are locked in `uv.lock`, including a CPU-only PyTorch build. Synthetic datasets are generated during each run. Benchmark data and generated results are not included in the repository.

## Experiments

Each experiment loads its matching JSON file from [`experiments/configs/`](experiments/configs/).

| Module | Comparison |
|---|---|
| `ate` | AIPW, IPW, and regression adjustment across nine settings and sample sizes |
| `cate` | Exact and variational Gaussian-process CATE inference |
| `likelihood_comparison` | Standard likelihood versus generalized Bayes under misspecification |
| `calibration` | Learning-rate calibration and scale sensitivity |
| `prior_strength` | Prior strength and prior misspecification |
| `cross_fitting` | Sensitivity to fold counts and partitions |
| `nuisance_learners` | Ridge, tree, neural-network, and Bayesian nuisance estimators |
| `bart_comparison` | BART versus generalized Bayes |
| `ihdp` | ATE estimation on the IHDP benchmark |

Run a study with `uv run --locked python -m experiments.<module>`, then generate its figures and LaTeX summary table with `uv run --locked python -m reporting.figures <module>`.

### Configuration

Edit the JSON files to change sample sizes, repetitions, seeds, learners, priors, or inference settings. To keep a separate configuration, copy an existing file, edit it, and select it with `--config`:

```bash
uv run --locked python -m experiments.ate --config experiments/configs/my_ate.json
```

A selected file supplies the complete configuration. Each run saves the settings it used alongside its results.

Default configurations run the full studies. Reduce repetition counts for exploratory runs; CATE also exposes the number of optimization epochs. Refitted-bootstrap calibration is enabled with `"refit_bootstrap": true` in `experiments/configs/calibration.json`.

### BART

Install the optional dependencies when running the BART comparison:

```bash
uv run --locked --extra bart python -m experiments.bart_comparison
```

### IHDP

Place the following benchmark files in `datasets/IHDP/` before running `experiments.ihdp`:

```text
ihdp_npci_1-100.train.npz
ihdp_npci_1-100.test.npz
ihdp_npci_1-1000.train.npz
ihdp_npci_1-1000.test.npz
```

Training bundles contain `x`, `t`, and `yf`; test bundles contain `mu0` and `mu1`. The two overlapping bundles are analyzed separately.

## Outputs

Each run creates `results/<module>/` containing replication-level `raw.csv`, aggregate `summary.csv`, and `settings.json`. Reporting adds PDF plots in `figures/` and a LaTeX table in `summary.tex`. Some experiments also save additional comparison summaries.

Existing experiment directories are not overwritten. Remove an old run deliberately, or set `GBCI_RESULTS_DIR` to another output root for both the experiment and reporting commands. Inputs, outputs, and local environments are gitignored.

## Implementation

- [`gbci/simulation.py`](gbci/simulation.py) defines the nine synthetic settings and their true treatment effects.
- [`gbci/estimation.py`](gbci/estimation.py) implements nuisance fitting, cross-fitting, causal pseudo-outcomes, and learning-rate calibration.
- [`gbci/posterior.py`](gbci/posterior.py) implements exact Gaussian ATE inference and exact/inducing-point variational GP CATE inference.
- [`experiments/`](experiments/) contains the study scripts and JSON configurations; [`reporting/`](reporting/) summarizes and plots their outputs.

Calculations use float64. CATE intervals describe the latent treatment-effect function and exclude observation noise. Fixed-score and refitted bootstrap calibrations estimate different quantities; their definitions are documented in `gbci/estimation.py`. Fold-count and nuisance-learner comparisons hold the baseline learning rate fixed to isolate the component being varied.
