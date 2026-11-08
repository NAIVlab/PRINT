"""Shared auto-loan preprocessing, saved policies, and held-out revenue."""
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import load
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
# Required for the class path stored in the supplied PRINT checkpoint.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

NUMERIC = ["rate", "rate1", "Competition_rate", "rel_compet_rate", "mp",
           "mp_rto_amtfinance", "onemonth", "Amount_Approved", "Term",
           "Primary_FICO", "days"]
N_PRICE_COLS = 6
TRAIN, VAL, TEST = slice(0, 50000), slice(50000, 60000), slice(60000, None)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def device():
    requested = os.environ.get("DEVICE", "auto")
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def output_dir():
    path = Path(os.environ.get("OUT_DIR", HERE / "out")).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def policy_path():
    return Path(os.environ.get("PRINT_NET", MODELS / "print_net.pt")).resolve()


def load_print(dev=None):
    path = policy_path()
    if not path.is_file():
        raise FileNotFoundError(f"PRINT checkpoint missing: {path}. Set PRINT_NET or run train_print.py explicitly.")
    # The supplied trusted model serializes CurvatureHead, not only its weights.
    net = torch.load(path, map_location=dev or device(), weights_only=False)
    return net.eval()


def policy_price(net, features, dev=None):
    dev = dev or next(net.parameters()).device
    with torch.no_grad():
        beta = net(torch.as_tensor(np.asarray(features), dtype=torch.float32,
                                   device=dev)).cpu().numpy()
    return np.clip(-0.5 * beta[:, 0] / beta[:, 1], 0, 40)


def load_features(data_dir=None):
    """Reconstruct price and preserve the original row-order split/preprocessing.

    As in the source analysis, the transformer is fitted to the full dataset,
    then rows 0:50000 / 50000:60000 / 60000: are used for train / validation /
    test. Changing that preprocessing would require refitting the saved models.
    """
    data_dir = data_dir or os.environ.get("DATA_DIR") or HERE / "data"
    csv = Path(data_dir) / "AutoLoan" / "CPRM_AutoLOan_OnlineAutoLoanData.csv"
    if not csv.is_file():
        raise FileNotFoundError(f"Required auto-loan data not found: {csv}. Request the CPRM-12-001 dataset from cprm@gsb.columbia.edu and place the CSV there, or set DATA_DIR.")
    data = pd.read_csv(csv)
    if len(data) <= 60000:
        raise ValueError("The original row-order split requires more than 60,000 records.")
    rate = data["onemonth"] / 1200.
    data["price"] = (data["mp"] * (1 + rate) / rate *
                     ((1 + rate) ** data["Term"] - 1) - data["Amount_Approved"])
    ct = make_column_transformer(
        (StandardScaler(), NUMERIC),
        (FunctionTransformer(feature_names_out="one-to-one"), ["Tier", "termclass"]),
        (OrdinalEncoder(), ["Type"]),
        (OneHotEncoder(sparse_output=False), ["State", "CarType", "partnerbin"]))
    X = ct.fit_transform(data)
    names = list(ct.get_feature_names_out())
    P = (np.asarray(data["price"]) / 1000.).reshape(-1, 1)
    Y = (np.asarray(data["price"]) * np.asarray(data["apply"]) / 1000.).reshape(-1, 1)
    return dict(X=X, X_obs=X[:, N_PRICE_COLS:], G=X[:, 0:1], P=P, Y=Y,
                feat_cols=names[N_PRICE_COLS:])


def load_demand():
    return load(MODELS / "demand_model.joblib")


def revenue(demand, X_full_test, price_col):
    """Mean modeled acceptance probability times price, in thousands of dollars."""
    prices = np.clip(np.asarray(price_col, float).reshape(-1, 1), 0, 40)
    features = np.concatenate((X_full_test, X_full_test * prices), axis=1)
    accept = demand.predict_proba(features)[:, 1].reshape(-1, 1)
    return float(np.mean(accept * prices))
