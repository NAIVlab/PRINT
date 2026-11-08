# Simulation study: Figure 1

This folder contains the simulation data generator, PRINT estimator, direct
regression comparator, Kallus--Zhou comparator, and the per-replicate policy
values used by the manuscript's current Figure 1. It works independently of the
original manuscript repository. No external dataset is needed.

Install the dependencies in the release's `requirements.txt` using Python
3.12, the tested environment. The simulation portion uses NumPy, SciPy,
scikit-learn, and Matplotlib; it does not require a GPU or PyTorch.

## Redraw the manuscript figure from the reference results

From this directory:

```sh
python bin/make_fig1.py
```

This reads `reference_results/` and writes
`results/figs_out/sim_regret_vs_n.{png,eps}`. To put the figure elsewhere:

```sh
python bin/make_fig1.py --output-dir /path/to/figures
```

The five reference CSV files are copied without changes from the original
`code/simulations/results_new/` directory, whose Figure 1 PNG is byte-identical
to the image included in the manuscript. Each of the four scenarios and ten
sample sizes has 50 replicates per method. `plugin_results.csv` contains PRINT
and direct regression; the four `kz_*.csv` files contain Kallus--Zhou. These are
aggregate policy values, not simulated individual records. Alternative
experimental result directories are excluded.

The figure plots the median of `ora_value - value` and shades its 25th--75th
percentiles across replicates. PNG preserves the translucent shaded bands; the
EPS backend renders transparency as opaque. Rendering can vary with Matplotlib
and font versions.

## Regenerate simulated data, fit all methods, and draw the figure

```sh
bash bin/run_simulations.sh
```

The launcher runs all four scenarios sequentially and actually fits all three
methods. It does not substitute cached baseline results. The default settings
are:

| Setting | Value |
| --- | --- |
| Sample sizes | 1,000; 2,000; 5,000; 10,000; 20,000; 50,000; 100,000; 200,000; 500,000; 1,000,000 |
| Replicates per scenario/sample size | 50, with seeds 0 through 49 |
| Fixed evaluation set | 20,000 observations, seed 32768 |
| Direct-effect severity `c4` | 1 (mild), 5 (severe) |
| Instrument-dependent dispersion `c_het` | 3 (weak), 4.5 (strong) |
| Price interval | [0, 60] |
| PRINT curvature bound | Estimated quadratic coefficient at most -0.1 |
| First-stage cross-fitting | 5 folds, fixed partition seed 12345 |
| Kallus--Zhou training cap | 50,000 observations |

All replicate results are retained. The seed loop defines Monte Carlo
replicates; it never selects a preferred seed or ranks runs. Cross-fitting and
the Kallus--Zhou subsampling also use fixed random states.

For a small end-to-end check, use a fresh output directory:

```sh
NS=1000 N_RUNS=1 RESULTS_DIR=/tmp/print-simulation-check \
  bash bin/run_simulations.sh
```

Set `PYTHON=/path/to/python` to select an interpreter. `NS`, `N_RUNS`,
`KZ_CAP`, and `RESULTS_DIR` are optional environment settings; see
`bash bin/run_simulations.sh --help`. Relative `RESULTS_DIR` paths are resolved
from the caller's current directory. The launcher refuses to reuse existing
result CSVs or run metadata, preventing accidental duplication of replicates.
Failures stop the run instead of silently removing failed replicates.

The default experiment involves 6,000 policy fits and sample sizes up to one
million. Allow hours or longer depending on CPU/BLAS performance, and several
GB of RAM. The Kallus--Zhou density estimation is relatively expensive. No full
runtime benchmark is asserted here; the small check and reference-figure
redraw are intended to validate the installation before a full run.

Outputs in `RESULTS_DIR` are `plugin_results.csv`, four `kz_*.csv` files,
`run_config.json` (settings and package versions), and `figs_out/`. Simulated
individual records remain in memory and are not exported.

## Reproduction details and limitations

The reference CSVs reproduce the plotted statistics exactly. A new simulation
run uses the same DGP formulas, parameter settings, and fitted methods, but
need not reproduce the historical CSVs bit for bit. The original scripts
seeded the covariate/instrument draws while price and revenue noise used a
global random stream. This release explicitly seeds that stream at the start
of every replicate, so rerunning the release with the same software produces
the same results. The generator's original use of the same integer random
state for the separate covariate and instrument/latent draws is retained; this
release does not redesign those streams. Numerical optimizer and linear
algebra versions can also affect results.

In the historical CSVs, the column named `regret` used the stronger oracle
that also observes the latent variable (`opt_value - value`). Figure 1 has
always recomputed regret against the X-only oracle (`ora_value - value`). The
plotting script continues to do so, and newly generated CSVs use this X-only
quantity in their `regret` column. `opt_value` remains available as an unused
diagnostic for the latent-variable oracle.

For nominal sample sizes above 50,000, the reference implementation fits
Kallus--Zhou on a reproducible random subsample of 50,000 rows (seed
`10000 + run`). The plot's horizontal coordinate is the nominal generated
sample size. This cap is a computational approximation, not an exact full-data
fit. Set `KZ_CAP` at least as large as the largest requested `NS` to fit that
comparator on every generated observation; such results differ from the
reference experiment.

The released implementation follows the code that produced the reference
plot. A few descriptions in the manuscript are broader than this code:

- Both comparator policies use X. Direct regression fits a quadratic basis in
  X, interacted with price and squared price; it does not include G. The
  Kallus--Zhou generalized propensity score conditions on X, rather than
  (X, G). Its two density bandwidths are 0.2, its propensity floor is 0.03,
  and it searches a linear policy with coefficient bounds [-60, 60].
- The PRINT implementation uses the two residual moment blocks with instrument
  multipliers `G - h1` and `G * (G - h1)`. It does not add a third block with
  `G**2 * (G - h1)`. Its price-response functions use the quadratic basis
  `[1, X1, X2, X1**2, X2**2, X1*X2]`.
- The historical generated price is clipped to [0, 60] before revenue is
  generated. 
  not explicitly show this implementation step.

These details are documented rather than silently changing the experiment or
the manuscript during code packaging.

## Files

| Path | Purpose |
| --- | --- |
| `src/data_gen.py` | Four DGP scenarios, oracle policy, policy evaluation |
| `bin/print_estimator.py` | Cross-fitted PRINT and direct regression |
| `bin/run_print.py` | PRINT and direct-regression simulation runner |
| `bin/run_kz18.py` | Kallus--Zhou simulation runner |
| `bin/run_simulations.sh` | Full reproduction entry point |
| `bin/make_fig1.py` | Regret plot from reference or freshly generated CSVs |
| `core/kz2018aistats/` | Continuous-action comparator and its small utility |
| `reference_results/` | Five immutable reference CSVs, 6,000 aggregate rows |
