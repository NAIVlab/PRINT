#%%
import numpy as np
import scipy
import sklearn
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import RobustScaler as Scaler

# For policy optimization
from scipy.optimize import minimize

# from _util import *
from core.kz2018aistats._util import *

#%%
class Gpropscore(BaseEstimator):
    """
    To Obtain Generalized Propensity Score Q_i for all data point
    """
    def __init__(self, bw_tx, bw_x):
        """
        bw_tx: bandwidth for kernel density estimation for f(t,x)
        bw_x:  bandwidth for kernel density estimation for f(x)
        """
        self.bw_tx = bw_tx
        self.bw_x = bw_x

    def fit(self, X, T):
        """
        Fit Kernel density estimation for f(t|x) by f(t,x)/f(x)
        """
        # X, T = check_X_y(X, T, accept_sparse=True)

        # scaling X and T and save the scalers to the class
        self.transX = Scaler()
        self.transX.fit(X)
        X = self.transX.transform(X)

        self.transT = Scaler()
        self.transT.fit(T)
        T = self.transT.transform(T)

        TX = np.concatenate((T,X), axis=1)

        self.kdeX  = KernelDensity(kernel='epanechnikov', bandwidth=self.bw_x).fit(X)
        self.kdeTX = KernelDensity(kernel='epanechnikov', bandwidth=self.bw_tx).fit(TX)

        return self

    def predict(self, X, T):
        X = self.transX.transform(X)
        T = self.transT.transform(T)
        TX = np.concatenate((T,X), axis=1)

        GPS = np.exp(self.kdeTX.score_samples(TX) - self.kdeX.score_samples(X))

        return GPS


class linear_policy():
    """
    Linear Policy Class
    """
    @autoargs()
    def __init__(self, beta = None, sigma = None, bias = 0, eps = 0,
                 uniform_exploration = False):
        """
        N(beta^T X, sigma^2) * (1-eps) + eps * Uniform()
        """
        pass


    def sample_actions(self, X, seed, return_Q = False):
        np.random.seed(seed)
        # behav policy?
        n = len(X)
        A_mean = X.dot(self.beta) + self.bias
        d_A = self.beta.shape[1]

        ## N(beta^T X, sigma^2)
        # change to uniform so the shift is more clear (if gaussian, actually this is still abs. cont. )
        A_noise = np.random.uniform(-self.sigma, self.sigma,
                                    (len(X), A_mean.shape[1]))
        sampled_A_0 = A_mean + A_noise

        """
        Below for data generation
        """
        if self.eps > 0:
            if self.eps <= 0.01:
                A_max = np.sum(np.abs(X), 1)
                A_min = -A_max
                randomly_sampled_A = np.random.uniform(A_min, A_max, size = (d_A, len(A_max))).T
                Q1 = (1 / (A_max * 2)) ** d_A
            else:
                # try to avoid a too large exploration space (during epsilon-greedy). Is this correct?
                A_noise = np.random.uniform(-1, 1
                                            , (len(X), A_mean.shape[1])
                                           ) # change to uniform so the shift is more clear (if gaussian, actually this is still abs. cont. )
                randomly_sampled_A = A_mean + A_noise
                Q1 = (1 / 2) ** d_A
            ## Final action
            choice = np.random.binomial(1, p = self.eps, size = n) # if 1, then 1
            sampled_A = randomly_sampled_A * choice[:, np.newaxis] + sampled_A_0 * (1 - choice)[:, np.newaxis]
        else:
            sampled_A = sampled_A_0

        if return_Q:
            ## Q with normal
            # true_Q = norm.pdf( T - T_mean, loc = 0, scale = T_sigma )
            if self.sigma == 0:
                Q0 = np.ones(len(X))
            else:
                Q0 = np.repeat((1 / (2 * self.sigma)) ** d_A
                          , len(X))
            if self.eps > 0:
                Q = self.eps * Q1 * choice + (1 - self.eps) * Q0 * (1 - choice)
                return sampled_A, Q
            else:
                return sampled_A, Q0
        else:
            return sampled_A



"""
Policy optimization over linear policy
"""


def policy_search_kernel(args):
    p_x = args['p_x']
    p_a = args['p_a']
    # bounds / initial guess default to the original [-1,1] scaling, but can be
    # overridden via args so the linear policy class can be scaled to the action
    # range of the problem (e.g. interior prices ~ O(10)).
    bounds = args.get('bounds', tuple((-1, 1) for _ in range(p_x * p_a)))
    beta_initial_guess = args.get('beta_init', np.random.uniform(-1, 1, p_x * p_a))

    oracle_actual_oracle_res = minimize(eval_wrapper_kernel,
                                        x0 = beta_initial_guess,
                                        bounds = bounds,
                                        options={'disp':False, 'gtol':1e-4},
                                        args= args)
    pi = linear_policy(beta = oracle_actual_oracle_res['x'].reshape((p_x, p_a)),
                       sigma = 0,
                       bias = 0)
    return pi

def eval_wrapper_kernel(beta, args):
    """ OPO: used in solving the policy pi(; \beta) after regr is learned, via Direct Policy Search

    Input:
        beta: the policy params to be evaluated
        args:
            X_shift: the covariate distribtuion to evaluate (optimize) on
            r_func:

    Return: -v as we are using a minimization tool
    """
    pi = linear_policy(beta = beta.reshape((args['p_x'], args['p_a'])), sigma = 0) # deterministic ones
    tau = pi.sample_actions(X = args['X'], seed = 0)
    args['tau'] = tau
    kernel_ope = kernel_OPE_known_Q()
    v = kernel_ope.run(args, args['h_factor'])
    vv = -v
    return vv


class kernel_OPE_known_Q():
    """
    Calculate kernel OPE with known Q (obtained by Gpropscore)
    """
    def __init__(self):
        pass
    def run(self, params, h_factor = 1):
        """ given known propensities
        Input:

            \tau = target policy: a vector of treatment values from the target policy (assumed given)
            ##
            threshold: max(Q_i, threshold) -> lower bound for the propensity Q
            Q_known: seems the same with Q, i.e., the generalized propensity

            ##
            y: outcome
            y_samp : NOT EXIST
            ##
            x_full: NOT EXIST
            x:
            x_samp: NOT EXIST
            ##
            T: action = treatment
            T_samp: NOT EXIST
            ##
            range of the treatment/action:
                t_lo:
                t_hi:
            ##
            h: Not sure how to specify

        h_factor:

            # 1. why h is so important?
            # rate
            # median heuristic does not work
            # kernel trick v.s. density v.s. those I used
            # purpose different?

            Optimal choice in the paper - but complex

        """
        THRESH = params['threshold']

    #     h = params['h']
        #n = params['n']
    #     kernel = epanechnikov_kernel #params['kernel_func']
    #     kernel_int =  epanechnikov_int # params['kernel_int_func']
        Q = params['Q_known']
        n = len(Q)
        x = params['X']
        d = x.shape[1]
        ######     If y_samp, T_samp is present, use that instead.
        if ('y_samp' in params.keys()):
            y_out = params['y_samp']
        else:
            y_out = params['y']
        if ('T_samp' in params.keys()):
            T = params['T_samp']
        else:
            T = params['T']
        if ('x_samp' in params.keys()):
            x = params['x_samp']
        else:
            x = params['X']
        ##
        #
        t_lo = np.min(T, axis = 0)
        t_hi = np.max(T, axis = 0)
        #
        tau = params['tau']
        self.clip_tau = tau.clip(max = t_hi, min = t_lo)
        #clip_tau = np.clip(tau, t_lo, t_hi)

        #
        """
        K(x, y) = exp(-gamma ||x-y||_1) = exp(- np.sum( |x1-x2| / median))
        -> gamma = 1 / median_of_L1
        """
        pairwise_diff = self.clip_tau - T #np.repeat(clip_tau, len(T), axis=0) - np.tile(T, [len(clip_tau), 1])
        pairwise_diff_l2 = np.sqrt(np.sum(pairwise_diff ** 2, axis = 1))
        median = np.median(pairwise_diff_l2) * h_factor
        kernel_values = np.exp(-pairwise_diff_l2 / median)
        # self.pairwise_diff = pairwise_diff
        # self.pairwise_diff_l2 = pairwise_diff_l2

        #K(x, y) = exp(-gamma ||x-y||_1)

#         gamma = 1 / median
#         kernel_values = np.diag(sk.metrics.pairwise.laplacian_kernel(self.clip_tau, T, gamma = gamma))

        # weights = []
        # self.Q_clip = []
        # self.kernel_values = kernel_values
        # for i in np.arange(n):
        #     ##
        #     kernel_value = kernel_values[i] #kernel( (clip_tau[i] - T[i])/h )
        #     ## Clip Q
        #     Q_i = Q[i]
        #     kernel_div_Q = kernel_value / max(Q_i, THRESH)
        #     #
        #     self.Q_clip.append(max(Q_i, THRESH))
        #     weights.append(kernel_div_Q)
        # self.weights = weights

        weights = kernel_values / Q.clip(max = None, min = THRESH)

        # self-normalized
        # v_vec = np.array(y_out) * np.array(weights) / np.sum(weights) * n
        # v_vec = np.array(y_out) * weights / np.sum(weights)

        # value = np.mean(v_vec)
        value = np.mean(np.array(y_out) * weights / np.sum(weights))
        # pessimistic
#         value = np.mean(v_vec) - np.std(v_vec) / np.sqrt(n)
        return value
