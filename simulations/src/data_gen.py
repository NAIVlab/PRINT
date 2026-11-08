#%%
import scipy
import numpy as np
from scipy.stats import multivariate_normal

# ---------------------------------------------------------------------------
# Simulation DGP retained from the manuscript experiment.
#   * Instrument channel of the price is free of U:  alpha_g(X,G)  (no U2).
#   * Identification comes from HETEROSKEDASTICITY: the price noise variance
#     depends on the instrument G (c_het * G * noise), independent of U; this
#     makes Var(P|G,X) shift with G (Omega_2 != 0) so beta_{p,1} is identified.
#   * beta_p1, beta_p2, beta_g depend on U only through U1 (=U^a); the price
#     (alpha_ux, price moments) depends on U only through U2 (=U^p); U1 ⟂ U2 | X.
#   * Confounding: alpha_ux carries U2 into the price and beta_ux depends
#     strongly on U2, so naive regression is biased; PRINT corrects it via G.
#   * beta_p2(U1,X) is a negative constant so the optimal price is
#     interior (P* ~ 8) and pricing is non-trivial.
# IV strength is the heteroskedasticity magnitude c_het; violation severity is
# the direct instrument effect c4 in beta_g.
# ---------------------------------------------------------------------------

#%% cases
_UXG = {
    'x_mean': [0.25, 0.25],
    'x_cov': [[1.0, 0], [0, 1.0]],
    'gu_mean_coef': [[2.,   0.5, 0.3],    # [mu_g, mu_u1, mu_u2]
                     [0.25, 0.3, 0.2],    # X1 coefs
                     [0.25, 0.4, 0.2]],   # X2 coefs
    'gu_cov': [[1., 0., 0.], [0., 3., 0.], [0., 0., 1.]]   # var_g, var_u1, var_u2
}
_REV = {
    'c_beta_p1': [0.3, 0.2],              # beta_p1 = U1^2 - X@c
    'c_beta_p2': [0.2, 0.0],              # beta_p2 = -b0 (negative CONSTANT, P* interior ~8)
    'c_beta_g': {'c3': [0.2, -0.1], 'c4': 1.},
    'c_beta_ux': {'cX': [0.4, 0.1], 'k_ux': 50., 'k_ux2': 5.}   # strong U2 confounding
}
_PRICE = {
    'c_alpha_g': {'c6': [1.2, 0.4], 'c7': 4.},   # higher historical price -> little clipping
    'c_alpha_ux': {'a_ux': 3.},           # alpha_ux = a_ux*U2
    'c_het': 2.5,                         # heteroskedasticity / IV strength
    'pmax': 60.
}


def _mk(c4, c_het):
    import copy
    p = {'UXG_params': copy.deepcopy(_UXG),
         'revenue_params': copy.deepcopy(_REV),
         'pricing_params': copy.deepcopy(_PRICE)}
    p['revenue_params']['c_beta_g']['c4'] = c4
    p['pricing_params']['c_het'] = c_het
    return p


# violation (c4): mild=1, severe=5.   IV/het strength (c_het): weak=3.0, strong=4.5
params_mildViolation_weakIV   = _mk(c4=1., c_het=3.0)
params_mildViolation_strongIV = _mk(c4=1., c_het=4.5)
params_severeViolation_weakIV   = _mk(c4=5., c_het=3.0)
params_severeViolation_strongIV = _mk(c4=5., c_het=4.5)


#%%
class sim_data:
    def __init__(self, params):
        self.params = params

    def gen(self, samp_size, seed=None):
        G, X, U1, U2 = gen_UXG(samp_size, self.params['UXG_params'], seed=seed)
        # Heteroskedastic price noise: Var depends on G, independent of U.
        c_het = self.params['pricing_params'].get('c_het', 1.0)
        pmax = self.params['pricing_params'].get('pmax', 60.)
        eps_p = c_het * G * np.random.normal(0., 1., samp_size) \
            + (np.random.rand(samp_size) - 0.5) * 2.
        P = pricing_model(G, X, U2, self.params['pricing_params']) + eps_p
        P = P.clip(0, pmax)
        Y = revenue_model(P, G, X, U1, U2, self.params['revenue_params']) \
            + (np.random.rand(samp_size) - 0.5) * 2.
        beta_p1_t = beta_p1_tilde(X, self.params)
        beta_p2_t = beta_p2_tilde(X, self.params)
        return G.reshape(-1, 1), X, P.reshape(-1, 1), Y.reshape(-1, 1), \
            U1.reshape(-1, 1), U2.reshape(-1, 1), \
            beta_p1_t.reshape(-1, 1), beta_p2_t.reshape(-1, 1)


#%% Policy evaluator
class sim_eval:
    def __init__(self, params, samp_size, seed=None):
        self.params = params
        self.pmax = params['pricing_params'].get('pmax', 60.)
        self.G, self.X, self.U1, self.U2 \
            = gen_UXG(samp_size, self.params['UXG_params'], seed=seed)
        self.GX = np.concatenate((self.G.reshape(-1, 1), self.X), axis=1)
        self.GXU2 = np.concatenate(
            (self.G.reshape(-1, 1), self.X, self.U2.reshape(-1, 1)), axis=1)

        P_opt = -0.5 * beta_p1_tilde(self.X, self.params) / beta_p2_tilde(self.X, self.params)
        P_opt = P_opt.clip(0, self.pmax)
        Y_opt = revenue_model(P_opt, self.G, self.X, self.U1, self.U2,
                              self.params['revenue_params'])

        P_opt_U = -0.5 * beta_p1(self.U1, self.X, self.params['revenue_params']['c_beta_p1']) \
            / beta_p2(self.U1, self.X, self.params['revenue_params']['c_beta_p2'])
        P_opt_U = P_opt_U.clip(0, self.pmax)
        Y_opt_U = revenue_model(P_opt_U, self.G, self.X, self.U1, self.U2,
                                self.params['revenue_params'])

        self.opt_val = np.mean(Y_opt)
        self.opt_val_U = np.mean(Y_opt_U)

    def eval(self, policy, useG=False, useU=False):
        if useG:
            P = policy(self.GXU2) if useU else policy(self.GX)
        else:
            P = policy(self.X)
        Y = revenue_model(P, self.G, self.X, self.U1, self.U2, self.params['revenue_params'])
        return np.mean(Y)


#%%
def gen_UXG(samp_size, params, seed=None):
    X = multivariate_normal.rvs(mean=params['x_mean'], cov=params['x_cov'],
                                size=samp_size, random_state=seed)
    GU1U2 = multivariate_normal.rvs(size=samp_size, cov=params['gu_cov'],
                                    random_state=seed)
    mean_coef = np.concatenate((np.ones((samp_size, 1)), X), axis=1)
    GU1U2 = GU1U2 + mean_coef @ np.array(params['gu_mean_coef'])
    G = GU1U2[:, 0]
    U1 = GU1U2[:, 1]
    U2 = GU1U2[:, 2]
    return G, X, U1, U2


#%% Coefficient functions (revenue/price models)
def beta_p1(U1, X, params):
    return U1**2 - X @ np.array(params)

def beta_p2(U1, X, params):
    # NEGATIVE CONSTANT price-curvature coefficient (matches the manuscript's
    # running example beta_{p,2}=eta_1<0).  A constant beta_{p,2} keeps the
    # heteroskedasticity-based identification focused on the single coefficient
    # beta_{p,1}(X) (classic Lewbel), which is robust; the optimal price still
    # varies with X through beta_{p,1}(X).
    return -params[0] * np.ones_like(np.asarray(U1, dtype=float))

def beta_g(U1, X, G, params):
    return (U1**2 + X @ np.array(params['c3'])) * G + params['c4'] * G

def beta_ux(U1, U2, X, params):
    return params['k_ux'] * U2 + params['k_ux2'] * U2**2 \
        + np.cos(U1 * U2 + X @ np.array(params['cX']))

def alpha_g(X, G, params):
    # instrument channel of the price model, free of U
    return (X @ np.array(params['c6']) + params['c7']) * G

def alpha_ux(U2, X, params):
    # Linear latent contribution before the historical price is clipped.
    return params['a_ux'] * U2

def pricing_model(G, X, U2, params):
    return alpha_g(X, G, params['c_alpha_g']) + alpha_ux(U2, X, params['c_alpha_ux'])

def revenue_model(P, G, X, U1, U2, params):
    return beta_p1(U1, X, params['c_beta_p1']) * P \
        + beta_p2(U1, X, params['c_beta_p2']) * P**2 \
        + beta_g(U1, X, G, params['c_beta_g']) \
        + beta_ux(U1, U2, X, params['c_beta_ux'])


#%% policy-relevant (X-only) coefficients:  E[beta_{p,j}(U,X) | X]
def _EU1sq(X, params):
    one = np.ones((X.shape[0], 1))
    mu_u1 = np.concatenate((one, X), axis=1) @ np.array(params['UXG_params']['gu_mean_coef'])[:, 1]
    return mu_u1**2 + params['UXG_params']['gu_cov'][1][1]

def beta_p1_tilde(X, params):
    return _EU1sq(X, params) - X @ np.array(params['revenue_params']['c_beta_p1'])

def beta_p2_tilde(X, params):
    cb2 = params['revenue_params']['c_beta_p2']
    return -cb2[0] * np.ones(X.shape[0])    # negative constant curvature
