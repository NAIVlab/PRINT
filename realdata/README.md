# Auto-loan analysis: Table 1 and Figures 2–3

This directory contains a standalone reproduction workflow and one optional
fixed-seed PRINT training run. It includes the fitted PRINT policy, the logistic
demand model used to evaluate revenue, and the fitted Kallus–Zhou policy. The
auto-loan dataset is not included; see [Data](#data). Saved per-customer
predictions, seed searches, hyperparameter searches, cluster launchers, and
historical run directories are excluded.

Install the dependencies in the release root's `requirements.txt`. The original
demand-model pickle records scikit-learn 1.1.2; the release uses the modern
`OneHotEncoder(sparse_output=False)` interface (scikit-learn 1.2 or newer). Loading
the older model on a newer version may issue a version warning. The original
training environment was not archived, so exact retraining across software or
GPU versions is not guaranteed.

## Data

The original *On-Line Auto Lending* dataset (CPRM-12-001), referenced in the
manuscript, is not included. It can be obtained from the Center for Pricing and
Revenue Management at Columbia Business School by emailing
[cprm@gsb.columbia.edu](mailto:cprm@gsb.columbia.edu). Put the CSV, with its
original filename and row order (208,085 records), in the `AutoLoan` folder:

```
realdata/data/
  AutoLoan/
    CPRM_AutoLOan_OnlineAutoLoanData.csv
```

The scripts use this `data/` directory by default. Optionally set
`DATA_DIR=/path/to/data` to use another copy, retaining the `AutoLoan/` subdirectory,
CSV filename, and original row order. Required columns are `rate`, `rate1`, `Competition_rate`,
`rel_compet_rate`, `mp`, `mp_rto_amtfinance`, `onemonth`, `Amount_Approved`,
`Term`, `Primary_FICO`, `days`, `Tier`, `termclass`, `Type`, `State`, `CarType`,
`partnerbin`, and `apply`.

`common.py` reconstructs price as
`mp * (1+r)/r * ((1+r)^Term - 1) - Amount_Approved`, where `r=onemonth/1200`.
Prices and revenues are in thousands of dollars. It preserves the original
analysis's feature transformer, fitted to all records before taking the row-order
split: first 50,000 for PRINT training, next 10,000 for validation, and remaining
records for testing. Regression uses the first 60,000 records. The APR (`rate`)
is the instrument. The first six price/instrument-derived feature columns are
excluded from the PRINT policy inputs. Altering the order, categories, or
preprocessing invalidates the supplied checkpoint's feature representation.

## Reproduce from the supplied PRINT policy

From the release root:

```bash
bash realdata/run.sh
# Select a specific Python environment or CPU explicitly if needed:
PYTHON=/path/to/python DEVICE=cpu bash realdata/run.sh
```

The default is `DEVICE=auto` (CUDA if available, otherwise CPU). This runs:

1. `make_table1.py`: loads the supplied PRINT and Kallus–Zhou policies, evaluates
   Oracle and Firm, and fits the Regression baseline **once at seed 0 for 1,000
   epochs**. All five policies must succeed; failures stop the workflow.
2. `make_figs.py`: draws Figure 2's policy-price densities and Figure 3's four
   partial-dependence panels from the same PRINT checkpoint. It verifies that
   PRINT predictions agree with the prices used in the table.

PRINT is never implicitly retrained. A missing checkpoint is an error. This
workflow runs directly in the current environment and submits no scheduler jobs.
A GPU is recommended for the regression fit; CPU execution is supported but
slower. Set `OUT_DIR=/path/to/output` to override `realdata/out/`.

For separate steps:

```bash
export OUT_DIR=/path/to/output
python realdata/make_table1.py
python realdata/make_figs.py
```

To redraw Figure 3 alone from the supplied PRINT policy, without fitting the
Regression baseline or generating Table 1, run:

```bash
python realdata/make_figs.py --pdp-only
```

## Optional: retrain PRINT once

```bash
OUT_DIR=/path/to/new_output bash realdata/run.sh retrain
```

`train_print.py` uses **seed 1**, one hidden layer of width 256, zero dropout,
curvature floor 0.05, weight decay `1e-5`, learner/adversary learning rates
`1e-3`/`1e-2`, batch size 500, and 700 epochs. First-stage nuisance fits use
five-fold cross-fitted histogram gradient boosting (fixed split/model seeds 0).
The second stage uses validation-moment early stopping, evaluated every two
epochs after a 100-epoch adversary probe. It does not compare multiple seeds,
compare hyperparameter configurations, or select an iterate by revenue.

The supplied policy's recorded early-stop epoch is 656 (zero-based).
Retraining may yield different numerical values across hardware/software.
Training checkpoints under `OUT_DIR/print_training/` allow the same interrupted
run to resume. Use a new `OUT_DIR` for a fresh fit or changed data. The `retrain`
launcher explicitly passes the newly fitted policy to the table and figures;
running individual scripts requires setting `PRINT_NET=/path/to/print_net.pt`
to use a policy other than the bundled one.

## Outputs

- `table1.csv`, `table1_metadata.json`: computed revenues and checkpoint metadata.
- `policy_prices.npz`, `print_policy.npz`: regenerated per-record predictions
  for local analysis; these are not included in the release.
- `data_density{,_zoom}.{eps,png,pdf}`: Figure 2 and its alternative y-axis view.
- `data_partialdep_{libor,term,fico,days}.{eps,png,pdf,npz}`: Figure 3 panels and
  aggregate plotting values. Each curve uses a fixed seed-0 background of up to
  2,000 test records, a 120-point grid between the first and 99th feature
  percentiles, and mean policy predictions over that background.

`models/reference.json` records fixed training settings and reference table values.
The full data-dependent analysis has not been rerun as part of packaging.

`core/neuralnet/genius_quad_twostage.py` retains its module name so the saved
`CurvatureHead` checkpoint can load. `checkpointed_print.py` implements a single
resumable fit; `reg_quad.py` implements Regression; `oadam.py` supplies the
optimistic Adam optimizer with its original attribution header.
