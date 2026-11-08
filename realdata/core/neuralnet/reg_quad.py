import os
import random
import numpy as np
import tempfile
import torch


def load_module(path):
    return torch.load(path, weights_only=False)

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torch import optim
# from .oadam import OAdam


# def add_weight_decay(net, l2_value, skip_list=()):
#     decay, no_decay = [], []
#     for name, param in net.named_parameters():
#         if not param.requires_grad:
#             continue  # frozen weights
#         if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
#             no_decay.append(param)
#         else:
#             decay.append(param)
#     return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]
#
#
# def _kernel(x, y, basis_func, sigma):
#     return basis_func(torch.cdist(x, y) * torch.abs(sigma))

class _BaseReg:
    def _pretrain(self, G, X, P, Y,
                  learner_weight_decay,
                  learner_lr,
                  n_epochs, batch_size,
                  model_dir, device, verbose, add_sample_inds=False
                  ):
        """
        Prepares the variables required to begin training.
        """
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

        if add_sample_inds:
            sample_inds = torch.tensor(np.arange(Y.shape[0]))
            self.train_ds = TensorDataset(G, X, P, Y, sample_inds)
        else:
            self.train_ds = TensorDataset(G, X, P, Y)

        self.train_dl = DataLoader(self.train_ds, batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0, worker_init_fn=seed_worker,
                                    generator=g)

        self.beta    = self.beta.to(device)
        self.residual = self.residual.to(device)

        self.optimizer   = optim.Adam(
            list(self.beta.parameters()) + list(self.residual.parameters()),
            lr=learner_lr)

        # self.scheduler_adversary = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer_adversary, 'min',
        #     factor=0.1, patience=10, threshold=1e-3)

        return G, X, P, Y

    def predict(self, X, model='avg', burn_in=0, CI_alpha=None):
        """
        Parameters
        ----------
        X : observable covariate
        model : one of ('avg', 'final'), whether to use an average of models or the final
        burn_in : discard the first "burn_in" epochs when doing averaging
        CI_alpha : if not None but a float, then it also returns the a/2 and 1-a/2, percentile of
            the predictions across different epochs (proxy for a confidence interval)
        """
        if model == 'avg':
            preds = np.array([load_module(os.path.join(self.model_dir,
                                                      "epoch{}".format(i)))(X).cpu().data.numpy()
                              for i in np.arange(burn_in, self.n_epochs)])
            if CI_alpha is None:
                return np.mean(preds, axis=0)
            else:
                return np.mean(preds, axis=0),\
                    np.percentile(
                        preds, 100 * CI_alpha / 2, axis=0), np.percentile(preds, 100 * (1 - CI_alpha / 2), axis=0)
        if model == 'final':
            return load_module(os.path.join(self.model_dir,
                                           "epoch{}".format(self.n_epochs - 1)))(X).cpu().data.numpy()
        if isinstance(model, int):
            return load_module(os.path.join(self.model_dir,
                                           "epoch{}".format(model)))(X).cpu().data.numpy()


class _BaseRegQuad(_BaseReg):
    def fit(self, G, X, P, Y,
            learner_weight_decay=1e-3,
            learner_lr=1e-3,
            n_epochs=100, batch_size=100,
            model_dir='.', device=None, verbose=0
            ):
        """
        Parameters
        ----------
        G : instruments
        X : observed covariates
        P : price
        Y : revenue
        # G_val : validation::instruments
        # X_val : validation::observed covariates
        # P_val : validation::price
        # Y_val : validation::revenue
        model_dir : folder where to store the learned models after every epoch
        """

        # Load preprocessed data
        G,X,P,Y = self._pretrain(G=G, X=X, P=P, Y=Y,
                  learner_weight_decay=learner_weight_decay,
                  learner_lr=learner_lr,
                  n_epochs=n_epochs, batch_size=batch_size,
                  add_sample_inds=False,
                  model_dir=model_dir, device=device, verbose=verbose
                  )

        for epoch in range(n_epochs):
            if self.verbose > 0:
                print("Epoch #", epoch, sep="")

            for it, (Gb, Xb, Pb, Yb) in enumerate(self.train_dl):
                Gb, Xb, Pb, Yb = map(lambda x: x.to(device), (Gb, Xb, Pb, Yb))

                self.beta.train()
                self.residual.train()

                loss = torch.mean((Yb \
                    - self.beta(Xb)[:,0].view(-1,1) * Pb \
                        - self.beta(Xb)[:,1].view(-1,1) * (Pb**2) \
                            - self.residual(torch.cat((Xb,Gb), dim=1)))**2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.beta.eval()
                self.residual.eval()

            torch.save(self.beta, os.path.join(
                self.model_dir, "epoch{}".format(epoch)))

        return self

class VanillaRegQuad(_BaseRegQuad):
    def __init__(self, beta, residual) -> None:
        self.beta      = beta
        self.residual  = residual
        self.skip_list = []
