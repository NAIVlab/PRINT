"""Two-stage PRINT residuals and training primitives.

The second-stage game contains the five parameter-dependent moments (h4, h5,
h6, w7, w8). The three first-stage residuals are fitted separately. The module
path is retained because the supplied policy checkpoint uses CurvatureHead.
"""
import os
import random
import copy
import tempfile

import numpy as np
import torch


def load_module(path):
    return torch.load(path, weights_only=False)

import torch.nn as nn
import torch.nn.functional as F_nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from .oadam import OAdam


class CurvatureHead(nn.Module):
    """Wraps a 2-output net so that beta2 <= -c_p2 always (the manuscript's
    curvature restriction -beta_{p,2} >= c_{p,2}, enforced in Algorithm 1):
        beta1 = raw[:,0],   beta2 = -c_p2 - softplus(raw[:,1]).
    """
    def __init__(self, net, c_p2=0.05):
        super().__init__()
        self.net = net
        self.c_p2 = c_p2

    def forward(self, x):
        raw = self.net(x)
        beta1 = raw[:, 0:1]
        beta2 = -self.c_p2 - F_nn.softplus(raw[:, 1:2])
        return torch.cat([beta1, beta2], dim=1)


def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': l2_value}]


def add_multi_weight_decay(net_list, l2_value, skip_list=()):
    decay, no_decay = [], []
    for net in net_list:
        for name, param in net.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': l2_value}]


def reinit_weights(layer):
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(layer.weight.data)


class TwoStageGeniusQuad:
    def __init__(self, h456, beta_learner, adversary):
        """
        Parameters
        ----------
        h456         : pytorch net X -> R^3   (h4, h5, h6)
        beta_learner : pytorch net X -> R^2   (beta1, beta2)
        adversary    : pytorch net X -> R^5   (test function f)
        """
        self.h456 = h456
        self.beta_learner = beta_learner
        self.adversary = adversary
        self.skip_list = []

    # ------------------------------------------------------------------ W ----
    # Second-stage residual: the FIVE alpha-dependent components of the
    # manuscript's W (the h4, h5, h6 components and w7, w8).  The three
    # first-stage residual components (G - h1-hat, G^2 - h3-hat, P - h2-hat)
    # involve no second-stage parameter: including them only lets the
    # adversary spend capacity on noise the learner cannot reduce, so they
    # are omitted from the second-stage game (their conditional moments are
    # satisfied by the first stage up to its estimation error).
    N_W = 5

    def _psi_fun(self, G, X, P, Y, H, adv_out=None):
        if adv_out is not None:
            W = torch.zeros_like(adv_out)
        else:
            W = torch.zeros([Y.shape[0], self.N_W], device=Y.device)
        h1, h2, h3 = H[:, 0:1], H[:, 1:2], H[:, 2:3]
        P_h2 = P - h2
        rG = G - h1

        pred_h456 = self.h456(X)                      # (n,3)
        W[:, 0] = pred_h456[:, 0] - (P_h2 * Y).view(-1)
        W[:, 1] = pred_h456[:, 1] - (P * P_h2).view(-1)
        W[:, 2] = pred_h456[:, 2] - (P**2 * P_h2).view(-1)

        beta_pred = self.beta_learner(X)              # (n,2)
        base = rG * P_h2
        rho1 = (base * Y).view(-1)
        rho2 = (base * P).view(-1)
        rho3 = (base * P**2).view(-1)
        rho4 = (G.view(-1)) * rho1
        rho5 = (G.view(-1)) * rho2
        rho6 = rho5 * P.view(-1)
        s = torch.clamp(h3 - h1**2, min=0.).view(-1)

        W[:, 3] = rho1 - rho2 * beta_pred[:, 0] - rho3 * beta_pred[:, 1]
        W[:, 4] = rho4 - s * pred_h456[:, 0] \
            - (rho5 - s * pred_h456[:, 1]) * beta_pred[:, 0] \
            - (rho6 - s * pred_h456[:, 2]) * beta_pred[:, 1]
        return W

    def _approx_sup_moment_eval(self, G, X, P, Y, H, f_of_x_collection):
        eval_list = []
        for f_of_x in f_of_x_collection:
            with torch.no_grad():
                mean_moment = torch.mean(self._psi_fun(G, X, P, Y, H, f_of_x) * f_of_x)
                eval_list.append(mean_moment.cpu().numpy())
        return float(np.max(eval_list))

    # ------------------------------------------------------------- training ----
    def _pretrain(self, G, X, P, Y, H,
                  beta_learner_weight_decay, adversary_weight_decay,
                  beta_learner_lr, adversary_lr, n_epochs, batch_size,
                  warm_start, model_dir, device, verbose):
        self.verbose = verbose
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.tempdir = tempfile.TemporaryDirectory(dir=model_dir)
        self.model_dir = self.tempdir.name
        self.n_epochs = n_epochs

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)
        self.train_ds = TensorDataset(G, X, P, Y, H)
        self.train_dl = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=0, worker_init_fn=seed_worker, generator=g)

        self.h456 = self.h456.to(device)
        self.beta_learner = self.beta_learner.to(device)
        self.adversary = self.adversary.to(device)

        if not warm_start:
            self.beta_learner.apply(lambda m: (
                m.reset_parameters() if hasattr(m, 'reset_parameters') else None))
            self.adversary.apply(lambda m: (
                m.reset_parameters() if hasattr(m, 'reset_parameters') else None))

        beta1 = 0.
        self.optimizer_learner = OAdam(
            add_multi_weight_decay([self.h456, self.beta_learner],
                                   beta_learner_weight_decay),
            lr=beta_learner_lr, betas=(beta1, .01))
        self.optimizer_adversary = OAdam(
            add_weight_decay(self.adversary, adversary_weight_decay,
                             skip_list=self.skip_list),
            lr=adversary_lr, betas=(beta1, .01))
        return G, X, P, Y, H

    def _learner_nets_eval(self):
        self.h456.eval(); self.beta_learner.eval(); self.adversary.eval()

    def _one_pass(self, device, train_learner_every, train_adversary_every):
        """One epoch over the loader: adversary step then learner step per batch."""
        for it, (Gb, Xb, Pb, Yb, Hb) in enumerate(self.train_dl):
            Gb, Xb, Pb, Yb, Hb = map(lambda x: x.to(device), (Gb, Xb, Pb, Yb, Hb))
            self._learner_nets_eval()

            if it % train_adversary_every == 0:
                self.adversary.train()
                test = self.adversary(Xb)
                W = self._psi_fun(Gb, Xb, Pb, Yb, Hb, test)
                loss_raw = W * test
                loss_learner = torch.mean(loss_raw)
                loss_adv = 0.25 * torch.mean(loss_raw**2) - loss_learner
                self.optimizer_adversary.zero_grad()
                loss_adv.backward()
                self.optimizer_adversary.step()
                self.adversary.eval()

            if it % train_learner_every == 0:
                self.h456.train(); self.beta_learner.train()
                test = self.adversary(Xb)
                W = self._psi_fun(Gb, Xb, Pb, Yb, Hb, test)
                loss_learner = torch.mean(W * test)
                self.optimizer_learner.zero_grad()
                loss_learner.backward()
                self.optimizer_learner.step()
                self._learner_nets_eval()

    def _warm_up(self, G, X, P, Y, H):
        """Warm up (h4,h5,h6,beta) by minimizing mean(W^2) with LBFGS."""
        optim_warm = optim.LBFGS(
            list(self.h456.parameters()) + list(self.beta_learner.parameters()),
            line_search_fn="strong_wolfe")

        def closure():
            optim_warm.zero_grad()
            W = self._psi_fun(G, X, P, Y, H)
            loss = torch.mean(W**2)
            loss.backward()
            return loss

        optim_warm.step(closure)

    def _earlystop_eval(self, device, X_val, n_epochs=100,
                        train_learner_every=1, train_adversary_every=1):
        """Train a throwaway copy of the game to collect adversary test functions
        f(X_val) for the early-stopping criterion (as in the one-stage version)."""
        f_of_x_val_collection = []
        for epoch in range(n_epochs):
            self._one_pass(device, train_learner_every, train_adversary_every)
            self._learner_nets_eval()
            with torch.no_grad():
                f_of_x_val_collection.append(self.adversary(X_val))
        return f_of_x_val_collection

    def fit(self, G, X, P, Y, H,
            G_val, X_val, P_val, Y_val, H_val, eval_freq=1,
            beta_learner_weight_decay=1e-3, adversary_weight_decay=1e-4,
            adversary_norm_reg=1e-3,
            beta_learner_lr=1e-3, adversary_lr=1e-3,
            n_epochs=100, batch_size=100,
            train_learner_every=1, train_adversary_every=1,
            warm_start=False, logger=None, model_dir='.',
            device=None, verbose=0):
        G, X, P, Y, H = self._pretrain(
            G, X, P, Y, H,
            beta_learner_weight_decay, adversary_weight_decay,
            beta_learner_lr, adversary_lr, n_epochs, batch_size,
            warm_start, model_dir, device, verbose)

        # early-stopping test-function collection (throwaway training run)
        self._warm_up(G[:10000], X[:10000], P[:10000], Y[:10000], H[:10000])
        f_of_x_val_collection = self._earlystop_eval(
            device, X_val, n_epochs=100,
            train_learner_every=train_learner_every,
            train_adversary_every=train_adversary_every)

        # reset and train for real
        self.h456.apply(reinit_weights)
        self.beta_learner.apply(reinit_weights)
        self.adversary.apply(reinit_weights)
        self._warm_up(G[:10000], X[:10000], P[:10000], Y[:10000], H[:10000])

        eval_history = []
        min_eval = float("inf")
        best_state = copy.deepcopy(self.beta_learner.state_dict())
        for epoch in range(n_epochs):
            if self.verbose > 0:
                print("Epoch #", epoch, sep="")
            self._one_pass(device, train_learner_every, train_adversary_every)
            torch.save(self.beta_learner,
                       os.path.join(self.model_dir, "epoch{}".format(epoch)))
            if epoch % eval_freq == 0:
                self._learner_nets_eval()
                curr_eval = self._approx_sup_moment_eval(
                    G_val, X_val, P_val, Y_val, H_val, f_of_x_val_collection)
                eval_history.append(curr_eval)
                if min_eval > curr_eval:
                    min_eval = curr_eval
                    best_state = copy.deepcopy(self.beta_learner.state_dict())

        self.beta_learner.load_state_dict(best_state)
        torch.save(self.beta_learner, os.path.join(self.model_dir, "earlystop"))
        return self

    def predict(self, X, model='earlystop'):
        """beta predictions from the early-stopped learner, in eval (deterministic)
        mode -- dropout OFF, unlike the legacy one-stage predict."""
        net = load_module(os.path.join(self.model_dir, "{}".format(model)))
        net.eval()
        with torch.no_grad():
            return net(X).cpu().data.numpy()
