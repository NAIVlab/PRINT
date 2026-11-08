"""Regenerate Figures 2 and 3 from the Table 1 policies and PRINT checkpoint."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import seaborn as sns

import common

PDP_FEATURES = {
    "libor": ("standardscaler__onemonth", "LIBOR (standardized)"),
    "term": ("standardscaler__Term", "Term (standardized)"),
    "fico": ("standardscaler__Primary_FICO", "FICO score (standardized)"),
    "days": ("standardscaler__days", "Days until approval (standardized)"),
}
ORDER = ["Firm", "Oracle", "PRINT", "KZ18", "Regression"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdp-only", action="store_true",
                        help="Regenerate Figure 3 directly from PRINT; omit Table 1 and Figure 2 inputs.")
    args = parser.parse_args()
    dev, out = common.device(), common.output_dir()
    price_path = out / "policy_prices.npz"
    if not args.pdp_only and not price_path.is_file():
        raise FileNotFoundError(f"Run make_table1.py first: {price_path} is missing.")
    net = common.load_print(dev)
    D, demand = common.load_features(), common.load_demand()
    X_test, X_test_full = D["X_obs"][common.TEST], D["X"][common.TEST]
    pnet = common.policy_price(net, X_test, dev)
    print("Policy net held-out revenue: %.4f" % common.revenue(demand, X_test_full, pnet), flush=True)

    if not args.pdp_only:
        with np.load(price_path) as archive:
            missing = set(ORDER) - set(archive.files)
            if missing:
                raise ValueError(f"Table 1 policy archive is incomplete: {sorted(missing)}")
            prices = {key: archive[key] for key in ORDER}
        if prices["PRINT"].shape != pnet.shape or not np.allclose(prices["PRINT"], pnet, atol=1e-5, rtol=1e-5):
            raise ValueError("PRINT checkpoint differs from the Table 1 policy. Rerun make_table1.py with the same PRINT_NET.")

    colors = {"Firm": "C0", "Oracle": "C1", "PRINT": "C2", "KZ18": "C3", "Regression": "C4"}
    for tag, ylim in (() if args.pdp_only else (("", None), ("_zoom", (0, 0.65)))):
        plt.figure(figsize=(4.2, 3.2))
        for key in ORDER:
            sns.kdeplot(prices[key], label="Kallus&Zhou'18" if key == "KZ18" else key,
                        color=colors[key], lw=1.8, clip=(0, 40))
        plt.xlim(-0.3, 10.5)
        if ylim:
            plt.ylim(*ylim)
        plt.xlabel("Price ($1,000)")
        plt.ylabel("Density")
        plt.legend(title="Pricing Policy", fontsize=7, title_fontsize=7.5)
        plt.tight_layout()
        for suffix in ("eps", "png", "pdf"):
            plt.savefig(out / f"data_density{tag}.{suffix}", dpi=200)
        plt.close()

    rng = np.random.RandomState(0)
    background = X_test[rng.choice(len(X_test), min(2000, len(X_test)), replace=False)]
    columns = list(D["feat_cols"])
    for name, (column, xlabel) in PDP_FEATURES.items():
        idx = columns.index(column)
        values = X_test[:, idx]
        lo, hi = np.percentile(values, [1, 99])
        grid = np.linspace(lo, hi, 120)
        raw_curve = []
        for value in grid:
            changed = background.copy()
            changed[:, idx] = value
            raw_curve.append(float(np.mean(common.policy_price(net, changed, dev))))
        raw_curve = np.asarray(raw_curve)
        curve = gaussian_filter1d(raw_curve, sigma=6)
        fig, ax = plt.subplots(figsize=(4, 3))
        effect, feature_mean = curve.mean(), values.mean()
        histogram = ax.twinx()
        counts, edges, _ = histogram.hist(values[(values >= lo) & (values <= hi)], bins=40, color="0.85")
        histogram.set_ylim(0, histogram.get_ylim()[1] * 5)
        histogram.set_yticks([])
        for spine in ("top", "right"):
            histogram.spines[spine].set_visible(False)
        ax.set_zorder(histogram.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.plot(grid, curve, color="#1f77b4", lw=2.2)
        ax.axhline(effect, ls=(0, (6, 4)), color="0.55", lw=0.8)
        ax.axvline(feature_mean, ls=(0, (6, 4)), color="0.55", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean policy price ($1,000)")
        ax.text(feature_mean, ax.get_ylim()[1], "mean", ha="center", va="bottom", fontsize=8)
        ax.spines["top"].set_visible(False)
        fig.tight_layout()
        for suffix in ("eps", "png", "pdf"):
            fig.savefig(out / f"data_partialdep_{name}.{suffix}", dpi=200)
        plt.close(fig)
        np.savez(out / f"data_partialdep_{name}.npz", grid=grid, raw_mean_price=raw_curve,
                 mean_price=curve, feature_mean=feature_mean, hist_counts=counts,
                 hist_edges=edges, background_size=len(background), smoothing_sigma=6)
        print(f"Figure 3 {name}: {curve[0]:.4f} -> {curve[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
