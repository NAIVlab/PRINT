"""Fit one PRINT model at fixed seed 1 and save its moment-early-stopped policy.

The supplied checkpoint supports reproduction without retraining PRINT. This
optional script repeats the fixed configuration: no seed or hyperparameter
search, and no revenue-based choice between early-stopped and final iterates.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

import common
from core.neuralnet.checkpointed_print import CheckpointedGeniusQuad
from core.neuralnet.genius_quad_twostage import CurvatureHead

SEED = 1
EPOCHS = 700


def first_stage(D):
    """Five-fold cross-fits on TRAIN; full-TRAIN nuisance fits predict VAL."""
    X, G, P = D["X_obs"], D["G"], D["P"]
    Xtr, Xval = X[common.TRAIN], X[common.VAL]
    XGtr = np.column_stack([Xtr, G[common.TRAIN]])
    XGval = np.column_stack([Xval, G[common.VAL]])
    tasks = [("h1", Xtr, Xval, G[common.TRAIN].ravel()),
             ("h2", XGtr, XGval, P[common.TRAIN].ravel()),
             ("h3", Xtr, Xval, G[common.TRAIN].ravel() ** 2)]
    Htr, Hval = np.empty((len(Xtr), 3)), np.empty((len(Xval), 3))
    folds = KFold(n_splits=5, shuffle=True, random_state=0)
    for j, (name, features, validation, target) in enumerate(tasks):
        oof = np.empty(len(features))
        for train_idx, heldout_idx in folds.split(features):
            fit = HistGradientBoostingRegressor(random_state=0).fit(features[train_idx], target[train_idx])
            oof[heldout_idx] = fit.predict(features[heldout_idx])
        full = HistGradientBoostingRegressor(random_state=0).fit(features, target)
        Htr[:, j] = np.clip(oof, target.min(), target.max())
        Hval[:, j] = np.clip(full.predict(validation), target.min(), target.max())
        print("First stage %s: out-of-fold R^2 %.4f" %
              (name, 1 - np.mean((oof - target) ** 2) / np.var(target)), flush=True)
    return Htr, Hval


def make_net(d_in, d_out):
    # Retain the supplied checkpoint's exact module layout, including inactive dropout.
    return nn.Sequential(nn.Dropout(0.0), nn.Linear(d_in, 256), nn.ReLU(),
                         nn.Dropout(0.0), nn.Linear(256, d_out))


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    out, dev = common.output_dir(), common.device()
    D = common.load_features()
    Htr, Hval = first_stage(D)
    common.set_seeds(SEED)
    nX = D["X_obs"].shape[1]

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device=dev)

    args = {}
    for key, field in (("G", "G"), ("X", "X_obs"), ("P", "P"), ("Y", "Y")):
        args[key] = tensor(D[field][common.TRAIN])
        args[key + "_val"] = tensor(D[field][common.VAL])
    work = out / "print_training"
    print(f"Training one PRINT model: seed {SEED}, {EPOCHS} epochs, {dev}", flush=True)
    fit = CheckpointedGeniusQuad(
        make_net(nX, 3), CurvatureHead(make_net(nX, 2), c_p2=0.05),
        make_net(nX, CheckpointedGeniusQuad.N_W)).fit(
            **args, H=tensor(Htr), H_val=tensor(Hval), eval_freq=2,
            beta_learner_weight_decay=1e-5, adversary_weight_decay=1e-5,
            beta_learner_lr=1e-3, adversary_lr=1e-2,
            n_epochs=EPOCHS, batch_size=500, train_learner_every=1,
            train_adversary_every=1, warm_start=False, logger=None,
            model_dir=str(work), resume_dir=str(work), device=dev)
    model = torch.load(work / "beta_earlystop.pt", map_location=dev, weights_only=False).eval()
    torch.save(model, out / "print_net.pt")
    prices = common.policy_price(model, D["X_obs"][common.TEST], dev)
    reward = common.revenue(common.load_demand(), D["X"][common.TEST], prices)
    np.savez(out / "print_policy.npz", price=prices, reward=reward, seed=SEED)
    (out / "training_metadata.json").write_text(json.dumps({
        "seed": SEED, "epochs": EPOCHS, "iterate": "validation-moment earlystop",
        "heldout_revenue": reward, "device": str(dev),
        "earlystop_epoch_zero_based": int(np.argmin(fit.eval_history)) * 2,
    }, indent=2) + "\n")
    print(f"Saved {out / 'print_net.pt'}; held-out revenue {reward:.4f}", flush=True)


if __name__ == "__main__":
    main()
