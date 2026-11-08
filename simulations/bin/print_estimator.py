"""
TWO-STAGE PRINT estimator -- the finite-basis instance of the manuscript's
Section-4 procedure (Corollary 2 case, beta_pg=0).

First stage (cross-fitted, clipped; Assumption "First stage"):
  h1(X)   = E[G   | X]      -- OLS on the quadratic basis phi(X)
  h2(X,G) = E[P   | G, X]   -- OLS on [phi(X), G, G*X1, G*X2, G^2]
  h3(X)   = E[G^2 | X]      -- OLS on phi(X)
Each observation's fitted nuisance is computed from the other folds
(cross-fitting), and fitted values are clipped to the observed range of the
response, as required by the theory.

Second stage (the linear-adversary / finite-basis min-max, which reduces to a
closed-form moment solve; Section 4 of the manuscript):
  alpha = (beta1, beta2, h4, h5, h6), with beta_j(X) = phi(X)' th_j and
  h4(X) = E[(P-h2)Y|X], h5(X) = E[P(P-h2)|X], h6(X) = E[P^2(P-h2)|X]
  estimated by OLS of the residual products on phi(X) (moment components 4-6
  of W are exactly these regressions when tested against the basis).
  (beta1, beta2) then solve the stacked conditional-moment restrictions of the
  last two components of W, tested against phi(X):
    w7 = rho1 - rho2*beta1(X) - rho3*beta2(X)
    w8 = rho4 - s*h4 - {rho5 - s*h5}*beta1(X) - {rho6 - s*h6}*beta2(X)
  with rG = G - h1, rP = P - h2, s = h3 - h1^2 (= E[G(G-h1)|X]) and
    rho1 = rG*rP*Y,    rho2 = rG*rP*P,    rho3 = rG*rP*P^2,
    rho4 = G*rG*rP*Y,  rho5 = G*rG*rP*P,  rho6 = G*rG*rP*P^2.
  E[w7|X]=0 and E[w8|X]=0 are the Lemma-1 and Lemma-2 (k=1) identities
  (Omega/Upsilon systems of Corollary 2).

Curvature restriction (Theorem "Regret bound" / Algorithm 1): the class is
restricted so that -beta2_hat >= c_{p,2} > 0; implemented as the floor
beta2_hat(X) <= -B2_FLOOR (the truth is beta_{p,2} = -0.2 <= -B2_FLOOR).

Weighting: the moment blocks are variance-weighted (diagonal weighting from a
first pass), the finite-basis analogue of the ||f||^2_{n,alpha~}-weighted
criterion; W_DIAG=0 disables it (identity weighting).
"""
import os
import numpy as np


def phi(X):
    # quadratic basis [1, X1, X2, X1^2, X2^2, X1*X2] (beta_tilde(X) is quadratic in X)
    X1, X2 = X[:, 0], X[:, 1]
    return np.column_stack([np.ones(len(X)), X1, X2, X1**2, X2**2, X1 * X2])


# Curvature restriction -beta2_hat >= c_{p,2} (theory-backed: Theorem "Regret
# bound" and Algorithm 1 restrict H so the estimated curvature is bounded away
# from zero; the DGP truth is beta_{p,2} = -0.2, so B2_FLOOR=0.1 <= c is valid).
B2_FLOOR = float(os.environ.get('B2_FLOOR', '0.1'))

# First-stage cross-fitting folds (Assumption "First stage"; K_FOLDS>=2).
K_FOLDS = int(os.environ.get('K_FOLDS', 5))
# Diagonal variance weighting of the stacked moments (finite-basis analogue of
# the alpha~-weighted norm in the min-max criterion). 1=on (default), 0=off.
W_DIAG = int(os.environ.get('W_DIAG', 1))


def _floor_b2(v):
    return np.minimum(v, -B2_FLOOR)


def _ols_pred(Ftr, ttr, Fte):
    return Fte @ np.linalg.lstsq(Ftr, ttr, rcond=None)[0]


def _crossfit(F, t, folds):
    """Cross-fitted OLS predictions of t on features F, clipped to the observed
    range of t (the theory's clipping of first-stage fits)."""
    out = np.empty(len(t))
    for te in folds:
        tr = np.concatenate([f for f in folds if f is not te])
        out[te] = _ols_pred(F[tr], t[tr], F[te])
    return np.clip(out, t.min(), t.max())


def first_stage(G, X, P):
    """Cross-fitted h1=E[G|X], h2=E[P|G,X], h3=E[G^2|X] (clipped)."""
    n = len(G)
    pX = phi(X)
    pGX = np.column_stack([pX, G, G * X[:, 0], G * X[:, 1], G**2])
    rng = np.random.RandomState(int(os.environ.get('CF_SEED', 12345)))
    idx = rng.permutation(n)
    folds = np.array_split(idx, max(K_FOLDS, 2))
    h1 = _crossfit(pX, G, folds)
    h2 = _crossfit(pGX, P, folds)
    h3 = _crossfit(pX, G**2, folds)
    return h1, h2, h3


def print_beta(G, X, P, Y):
    """Two-stage PRINT estimator; returns (beta1_hat(.), beta2_hat(.))."""
    # ---- first stage ----
    h1, h2, h3 = first_stage(G, X, P)
    rG = G - h1
    rP = P - h2
    s = np.maximum(h3 - h1**2, 0.0)          # E[G(G-h1)|X] = Var(G|X) >= 0

    # ---- second stage ----
    F = phi(X); q = F.shape[1]
    # h4, h5, h6: regressions of the price-residual products on the basis
    # (moment components 4-6 of W tested against phi)
    beta_h4 = np.linalg.lstsq(F, rP * Y, rcond=None)[0]
    beta_h5 = np.linalg.lstsq(F, rP * P, rcond=None)[0]
    beta_h6 = np.linalg.lstsq(F, rP * P**2, rcond=None)[0]
    h4, h5, h6 = F @ beta_h4, F @ beta_h5, F @ beta_h6

    # w7 and w8 pieces: w = a - b*beta1(X) - c*beta2(X)
    a7, b7, c7 = rG * rP * Y, rG * rP * P, rG * rP * P**2
    Gr = G * rG
    a8 = Gr * rP * Y - s * h4
    b8 = Gr * rP * P - s * h5
    c8 = Gr * rP * P**2 - s * h6

    # Stacked moments E[phi(X) w_j] = 0, j=7,8: 2q equations, 2q unknowns
    def _blocks(a, b, c):
        cvec = (F * a[:, None]).mean(0)
        Ab1 = ((F * b[:, None])[:, :, None] * F[:, None, :]).mean(0)
        Ab2 = ((F * c[:, None])[:, :, None] * F[:, None, :]).mean(0)
        return cvec, Ab1, Ab2

    c7v, A71, A72 = _blocks(a7, b7, c7)
    c8v, A81, A82 = _blocks(a8, b8, c8)
    A = np.block([[A71, A72], [A81, A82]])
    cv = np.concatenate([c7v, c8v])
    th = np.linalg.lstsq(A, cv, rcond=None)[0]

    if W_DIAG:
        # variance-weight each of the 2q moments (alpha~-weighting analogue)
        th1, th2 = th[:q], th[q:]
        e7 = a7 - b7 * (F @ th1) - c7 * (F @ th2)
        e8 = a8 - b8 * (F @ th1) - c8 * (F @ th2)
        gi = np.concatenate([F * e7[:, None], F * e8[:, None]], axis=1)  # (n, 2q)
        wts = 1.0 / np.maximum(gi.var(axis=0), 1e-12)
        Wm = np.sqrt(wts)
        Aw = A * Wm[:, None]
        cw = cv * Wm
        th = np.linalg.lstsq(Aw.T @ Aw + 1e-10 * np.eye(2 * q), Aw.T @ cw, rcond=None)[0]

    th1, th2 = th[:q], th[q:]
    return (lambda Xn: phi(Xn) @ th1), (lambda Xn: _floor_b2(phi(Xn) @ th2))


def naive_beta(X, P, Y):
    F = phi(X); q = F.shape[1]
    feat = np.hstack([F, F * P[:, None], F * (P**2)[:, None]])
    co = np.linalg.lstsq(feat, Y, rcond=None)[0].ravel()
    th1 = co[q:2*q]; th2 = co[2*q:3*q]
    return (lambda Xn: phi(Xn) @ th1), (lambda Xn: _floor_b2(phi(Xn) @ th2))
