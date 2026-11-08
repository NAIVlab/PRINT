"""Redraw Figure 1 from per-replicate policy values (no simulation fitting)."""
import argparse
import csv, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
root = Path(__file__).resolve().parents[1]
parser.add_argument('--results-dir', type=Path, default=root / 'reference_results')
parser.add_argument('--output-dir', type=Path, default=root / 'results' / 'figs_out')
args = parser.parse_args()

# Regret vs the X-only oracle, by (violation, IV, n, method).
d = collections.defaultdict(list)
NS = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]


def load(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        v = r.get("value", "")
        if v == "" or v == "nan":
            continue
        reg = float(r["ora_value"]) - float(v)
        d[(r["Violation"], r["IV"], int(r["n"]), r["method"])].append(reg)


load(args.results_dir / "plugin_results.csv")
for cell in ["mild_strong", "mild_weak", "severe_strong", "severe_weak"]:
    load(args.results_dir / f"kz_{cell}.csv")

# Also permit deliberately smaller sample sizes for a quick end-to-end check.
NS = sorted({key[2] for key in d})

cells = [("mild", "strong"), ("mild", "weak"), ("severe", "strong"), ("severe", "weak")]
methods = ["PRINT", "KZ18", "Naive"]
colors = {"PRINT": "#1f77b4", "KZ18": "#2ca02c", "Naive": "#d62728"}
labels = {"PRINT": "PRINT", "KZ18": "Kallus & Zhou (2018)",
          "Naive": "Direct regression"}
mk = {"PRINT": "-o", "KZ18": "-^", "Naive": "-s"}

fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
for ax, (viol, iv) in zip(axes.ravel(), cells):
    for method in methods:
        ns = [n for n in NS if d[(viol, iv, n, method)]]
        if not ns:
            continue
        med = np.array([np.median(d[(viol, iv, n, method)]) for n in ns])
        q1 = np.array([np.percentile(d[(viol, iv, n, method)], 25) for n in ns])
        q3 = np.array([np.percentile(d[(viol, iv, n, method)], 75) for n in ns])
        ax.plot(ns, med, mk[method], color=colors[method], label=labels[method], lw=2, ms=5)
        ax.fill_between(ns, q1, q3, color=colors[method], alpha=0.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{viol} violation, {iv} IV", fontsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.set_ylim(bottom=0.3, top=60)
for ax in axes[-1]:
    ax.set_xlabel("sample size $n$")
for ax in axes[:, 0]:
    ax.set_ylabel("regret  $\\mathcal{V}(\\pi^*)-\\mathcal{V}(\\widehat\\pi)$")
axes[0, 0].legend(fontsize=9, loc="upper right")
fig.tight_layout()
args.output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output_dir / "sim_regret_vs_n.png", dpi=600)
fig.savefig(args.output_dir / "sim_regret_vs_n.eps")
plt.close(fig)
print(f"saved {args.output_dir}/sim_regret_vs_n.{{png,eps}}")
