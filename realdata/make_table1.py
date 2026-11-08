"""Regenerate Table 1 and the per-policy prices needed for Figure 2.

PRINT, demand evaluation, and KZ18 use supplied models. The regression baseline
is fitted once at fixed seed 0 (1,000 epochs), matching the original analysis.
"""
import argparse
import json
import pickle
import sys
import types

import numpy as np
import scipy.special
import torch
import torch.nn as nn

import common
from core.neuralnet.reg_quad import VanillaRegQuad

ORDER = ["Oracle", "Firm", "PRINT", "KZ18", "Regression"]


def kz_prices(G, X_obs):
    # The archived pickle contains only the fitted linear policy's coefficients.
    package = types.ModuleType("core.kz2018aistats")
    sys.modules.setdefault("core.kz2018aistats", package)
    module = types.ModuleType("core.kz2018aistats.kz18aistats")
    module.linear_policy = type("linear_policy", (), {})
    sys.modules[module.__name__] = module
    with (common.MODELS / "kz18_policy.pkl").open("rb") as stream:
        policy = pickle.load(stream)
    features = np.concatenate((G, X_obs), axis=1)[common.TEST]
    weights = next(value.ravel().astype(float) for value in policy.__dict__.values()
                   if isinstance(value, np.ndarray) and
                   value.size in (features.shape[1], features.shape[1] + 1))
    price = (features @ weights if weights.size == features.shape[1]
             else features @ weights[1:] + weights[0])
    return np.clip(price.reshape(-1, 1), 0, 40)


def regression_prices(D, dev, out):
    common.set_seeds(0)
    X_obs, G, P, Y = (D[k] for k in ("X_obs", "G", "P", "Y"))
    nX = X_obs.shape[1]

    def net(d_in, d_out):
        return nn.Sequential(nn.Dropout(0.1), nn.Linear(d_in, 64), nn.ReLU(),
                             nn.Dropout(0.1), nn.Linear(64, d_out))

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device=dev)

    fit = VanillaRegQuad(net(nX, 2), net(nX + 1, 1)).fit(
        G=tensor(G[:60000]), X=tensor(X_obs[:60000]), P=tensor(P[:60000]),
        Y=tensor(Y[:60000]), learner_weight_decay=1e-3, learner_lr=1e-3,
        n_epochs=1000, batch_size=512, model_dir=str(out / "regression_work"),
        device=dev, verbose=0)
    beta = fit.predict(tensor(X_obs[common.TEST]), "final")
    # Release the temporary epoch checkpoints immediately after prediction.
    fit.tempdir.cleanup()
    return np.clip((-0.5 * beta[:, 0] / beta[:, 1]).reshape(-1, 1), 0, 40)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    dev, out = common.device(), common.output_dir()
    net = common.load_print(dev)
    D, demand = common.load_features(), common.load_demand()
    X, X_obs, G, P = (D[k] for k in ("X", "X_obs", "G", "P"))
    X_test = X[common.TEST]
    half = demand.coef_.shape[1] // 2
    alpha, beta = demand.coef_[0, :half], demand.coef_[0, half:]
    a, b = X_test @ alpha + demand.intercept_, X_test @ beta
    prices = {
        "Oracle": np.clip((-1. - np.real(scipy.special.lambertw(np.exp(-1 + a)))) / b,
                          0, 40).reshape(-1, 1),
        "Firm": np.clip(P[common.TEST], 0, 40),
        "PRINT": common.policy_price(net, X_obs[common.TEST], dev).reshape(-1, 1),
        "KZ18": kz_prices(G, X_obs),
    }
    print("Fitting Regression baseline: seed 0, 1,000 epochs", flush=True)
    prices["Regression"] = regression_prices(D, dev, out)
    revenues = {key: common.revenue(demand, X_test, value) for key, value in prices.items()}
    with (out / "table1.csv").open("w") as stream:
        stream.write("policy,revenue\n")
        for key in ORDER:
            print(f"{key:12s} {revenues[key]:.4f}", flush=True)
            stream.write(f"{key},{revenues[key]:.4f}\n")
    np.savez(out / "policy_prices.npz", **{key: value.ravel() for key, value in prices.items()})
    np.savez(out / "print_policy.npz", price=prices["PRINT"].ravel(), reward=revenues["PRINT"])
    (out / "table1_metadata.json").write_text(json.dumps({
        "print_checkpoint": str(common.policy_path()), "revenue": revenues,
        "regression_seed": 0, "regression_epochs": 1000, "device": str(dev)
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
