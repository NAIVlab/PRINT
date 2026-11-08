"""
Kallus & Zhou (2018, AISTATS) continuous-action policy-learning baseline on the
DGP.  KZ18 does direct policy search that maximises a kernel
off-policy-value estimate under a *generalized propensity score*; it assumes no
UNOBSERVED confounding (ignorability given the conditioning covariates).  Here it
is given the SAME information as PRINT/Naive -- the personalization features X
only (not the instrument G) -- and a linear-in-[1,X] policy whose coefficient
bounds are scaled to the price range so the policy class CAN reach the interior
optimum. Its unconfoundedness assumption need not hold in this simulation.

Run:  python run_kz18.py            (writes kz18_results.csv)
Env:  N_RUNS (default 50), NS (comma list of n), PARAM_SET ('all' or one key), OUT
"""
import sys, os, csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.data_gen import (sim_data, sim_eval,
                          params_mildViolation_strongIV, params_mildViolation_weakIV,
                          params_severeViolation_strongIV, params_severeViolation_weakIV)
from core.kz2018aistats.kz18aistats import Gpropscore, policy_search_kernel

PMAP = {'mild_strong': ('mild', 'strong', params_mildViolation_strongIV),
        'mild_weak': ('mild', 'weak', params_mildViolation_weakIV),
        'severe_strong': ('severe', 'strong', params_severeViolation_strongIV),
        'severe_weak': ('severe', 'weak', params_severeViolation_weakIV)}


def kz18_value(ev, params, n, run, pmax):
    np.random.seed(run)
    G, X, P, Y, *_ = sim_data(params=params).gen(n, seed=run)
    # Preserve the reference experiment's computational approximation: fit on
    # at most KZ_CAP observations. This is not a full-n fit when n > KZ_CAP.
    cap = int(os.environ.get('KZ_CAP', 50000))
    if cap < 10:
        raise ValueError('KZ_CAP must be at least 10')
    if len(X) > cap:
        idx = np.random.RandomState(10000 + run).choice(len(X), cap, replace=False)
        X, P, Y = X[idx], P[idx], Y[idx]
    P = P.reshape(-1, 1)
    Y = Y.ravel()
    Xint = np.concatenate([np.ones((len(X), 1)), X], axis=1)   # [1, X1, X2]
    # generalized propensity score f(P|X) (kernel density), X-only conditioning
    Qs = Gpropscore(0.2, 0.2).fit(X=X, T=P)
    Q_est = Qs.predict(X=X, T=P)
    px = Xint.shape[1]
    # start at the mean observed price (intercept), zero slopes; bounds = price range
    beta_init = np.zeros(px); beta_init[0] = float(np.mean(P))
    data_sim = {'p_x': px, 'p_a': 1, 'y': Y, 'X': Xint, 'T': P,
                'threshold': 0.03, 'Q_known': Q_est, 'h_factor': 1.,
                'bounds': tuple((-pmax, pmax) for _ in range(px)),
                'beta_init': beta_init}
    pi = policy_search_kernel(data_sim)

    def policy_kz(Xq):
        Xq2 = np.concatenate([np.ones((len(Xq), 1)), Xq], axis=1)
        Pr = pi.sample_actions(Xq2, seed=run)
        return np.clip(Pr.ravel(), 0, pmax)

    return ev.eval(policy=policy_kz)   # useG=False -> policy is a function of X


def main():
    n_runs = int(os.environ.get('N_RUNS', 50))
    ns = [int(x) for x in os.environ.get('NS', '1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000').split(',')]
    if n_runs < 1 or not ns or min(ns) < 10:
        raise ValueError('N_RUNS must be positive and every sample size must be at least 10')
    pf = os.environ.get('PARAM_SET', 'all')
    keys = list(PMAP) if pf == 'all' else [pf]
    out = os.environ.get('OUT', 'kz18_results.csv')
    f = open(out, 'x', newline=''); w = csv.writer(f)
    w.writerow(['Violation', 'IV', 'n', 'run', 'ora_value', 'opt_value', 'method', 'value', 'regret'])
    for key in keys:
        viol, ivs, params = PMAP[key]
        ev = sim_eval(params=params, samp_size=20000, seed=32768)
        ora, optU, pmax = ev.opt_val, ev.opt_val_U, ev.pmax
        for n in ns:
            for run in range(n_runs):
                val = kz18_value(ev, params, n, run, pmax)
                if not np.isfinite(val):
                    raise RuntimeError(f'{key}, n={n}, run={run}: nonfinite KZ18 value')
                w.writerow([viol, ivs, n, run, ora, optU, 'KZ18', val, ora - val])
            f.flush()
            print(f'{key} n={n} done ({n_runs} runs)', flush=True)
    f.close()


if __name__ == '__main__':
    main()
