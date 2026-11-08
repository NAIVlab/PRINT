"""
Full simulation runner using the TWO-STAGE PRINT estimator (cross-fitted
first-stage regressions h1,h2,h3 + closed-form second-stage moment solve for
(beta1,beta2,h4,h5,h6) -- the finite-basis instance of the manuscript's
Section-4 min-max criterion) vs the biased Naive regression.
Produces per-run regret across sample sizes -> regret-vs-n convergence figure.

Run:  python run_print.py            (writes plugin_results.csv)
Env:  N_RUNS (default 50), NS (comma list of n), PARAM_SET ('all' or one key), OUT
"""
import sys, os, csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.data_gen import (sim_data, sim_eval,
                          params_mildViolation_strongIV, params_mildViolation_weakIV,
                          params_severeViolation_strongIV, params_severeViolation_weakIV)
from print_estimator import print_beta, naive_beta

PMAP = {'mild_strong': ('mild', 'strong', params_mildViolation_strongIV),
        'mild_weak': ('mild', 'weak', params_mildViolation_weakIV),
        'severe_strong': ('severe', 'strong', params_severeViolation_strongIV),
        'severe_weak': ('severe', 'weak', params_severeViolation_weakIV)}


def policy(b1, b2, pmax):
    return lambda Xn: np.clip(-0.5 * b1(Xn) / b2(Xn), 0, pmax)


def main():
    n_runs = int(os.environ.get('N_RUNS', 50))
    ns = [int(x) for x in os.environ.get('NS', '1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000').split(',')]
    if n_runs < 1 or not ns or min(ns) < 10:
        raise ValueError('N_RUNS must be positive and every sample size must be at least 10')
    pf = os.environ.get('PARAM_SET', 'all')
    keys = list(PMAP) if pf == 'all' else [pf]
    out = os.environ.get('OUT', 'plugin_results.csv')
    # Refuse accidental duplicate results; use a fresh output path for each run.
    f = open(out, 'x', newline=''); w = csv.writer(f)
    w.writerow(['Violation', 'IV', 'n', 'run', 'ora_value', 'opt_value', 'method', 'value', 'regret'])
    for key in keys:
        viol, ivs, params = PMAP[key]
        ev = sim_eval(params=params, samp_size=20000, seed=32768)
        ora, optU = ev.opt_val, ev.opt_val_U
        pmax = ev.pmax
        for n in ns:
            for run in range(n_runs):
                # Fixed Monte Carlo replicates, all retained; this also seeds
                # price/revenue noise that the original generator leaves global.
                np.random.seed(run)
                G, X, P, Y, *_ = sim_data(params=params).gen(n, seed=run)
                G, P, Y = G.ravel(), P.ravel(), Y.ravel()
                pb1, pb2 = print_beta(G, X, P, Y)
                vp = ev.eval(policy=policy(pb1, pb2, pmax))
                nb1, nb2 = naive_beta(X, P, Y)
                vn = ev.eval(policy=policy(nb1, nb2, pmax))
                for method, val in [('PRINT', vp), ('Naive', vn)]:
                    if not np.isfinite(val):
                        raise RuntimeError(f'{key}, n={n}, run={run}: nonfinite {method} value')
                    w.writerow([viol, ivs, n, run, ora, optU, method, val, ora - val])
                f.flush()
            # progress line
            print(f'{key} n={n} done ({n_runs} runs)', flush=True)
    f.close()


if __name__ == '__main__':
    main()
