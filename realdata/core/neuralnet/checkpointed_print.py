"""Persistent single-run PRINT trainer with validation-moment early stopping.

No seed or hyperparameter selection is performed. Checkpoints permit an
interrupted run to resume with its optimizer and random-number-generator states.
"""
import os
import copy
import random

import numpy as np
import torch

from .genius_quad_twostage import TwoStageGeniusQuad, reinit_weights


def _rng_state():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _set_rng_state(s):
    torch.set_rng_state(s["torch"].cpu() if hasattr(s["torch"], "cpu") else s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(s["cuda"].cpu() if hasattr(s["cuda"], "cpu") else s["cuda"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


class CheckpointedGeniusQuad(TwoStageGeniusQuad):

    def fit(self, G, X, P, Y, H,
            G_val, X_val, P_val, Y_val, H_val, eval_freq=2,
            beta_learner_weight_decay=1e-3, adversary_weight_decay=1e-3,
            adversary_norm_reg=1e-3,
            beta_learner_lr=1e-3, adversary_lr=1e-2,
            n_epochs=700, batch_size=500,
            train_learner_every=1, train_adversary_every=1,
            warm_start=False, logger=None, model_dir='.',
            device=None, verbose=0,
            resume_dir=None, ckpt_every=25, probe_epochs=100,
            ):
        assert resume_dir is not None
        os.makedirs(resume_dir, exist_ok=True)
        probe_ckpt = os.path.join(resume_dir, "probe.pt")
        main_ckpt = os.path.join(resume_dir, "main.pt")
        self.resume_dir = resume_dir

        G, X, P, Y, H = self._pretrain(
            G, X, P, Y, H,
            beta_learner_weight_decay, adversary_weight_decay,
            beta_learner_lr, adversary_lr, n_epochs, batch_size,
            warm_start, model_dir, device, verbose)

        # ---- phase 1: throwaway probe run -> adversary test functions --------
        if os.path.exists(main_ckpt) or os.path.exists(probe_ckpt):
            ck = torch.load(probe_ckpt, map_location=device, weights_only=False)
            f_of_x_val_collection = [t.to(device) for t in ck["f_collection"]]
            _set_rng_state(ck["rng"])
            if hasattr(self.train_dl, "generator") and self.train_dl.generator is not None:
                self.train_dl.generator.set_state(ck["loader_g"])
            print("resume: probe phase loaded from %s" % probe_ckpt, flush=True)
        else:
            self._warm_up(G[:10000], X[:10000], P[:10000], Y[:10000], H[:10000])
            f_of_x_val_collection = self._earlystop_eval(
                device, X_val, n_epochs=probe_epochs,
                train_learner_every=train_learner_every,
                train_adversary_every=train_adversary_every)
            torch.save({"f_collection": [t.cpu() for t in f_of_x_val_collection],
                        "rng": _rng_state(),
                        "loader_g": self.train_dl.generator.get_state()},
                       probe_ckpt)
            print("probe phase done; checkpointed", flush=True)

        # ---- phase 2: main adversarial training with resume ------------------
        start_epoch = 0
        eval_history = []
        min_eval = float("inf")
        if os.path.exists(main_ckpt):
            ck = torch.load(main_ckpt, map_location=device, weights_only=False)
            self.h456.load_state_dict(ck["h456"])
            self.beta_learner.load_state_dict(ck["beta"])
            self.adversary.load_state_dict(ck["adv"])
            self.optimizer_learner.load_state_dict(ck["opt_learner"])
            self.optimizer_adversary.load_state_dict(ck["opt_adversary"])
            best_state = ck["best_state"]
            min_eval = ck["min_eval"]
            eval_history = ck["eval_history"]
            start_epoch = ck["epoch"] + 1
            _set_rng_state(ck["rng"])
            if hasattr(self.train_dl, "generator") and self.train_dl.generator is not None:
                self.train_dl.generator.set_state(ck["loader_g"])
            print("resume: main phase from epoch %d" % start_epoch, flush=True)
        else:
            self.h456.apply(reinit_weights)
            self.beta_learner.apply(reinit_weights)
            self.adversary.apply(reinit_weights)
            self._warm_up(G[:10000], X[:10000], P[:10000], Y[:10000], H[:10000])
            best_state = copy.deepcopy(self.beta_learner.state_dict())

        for epoch in range(start_epoch, n_epochs):
            self._one_pass(device, train_learner_every, train_adversary_every)
            if epoch % eval_freq == 0:
                self._learner_nets_eval()
                curr_eval = self._approx_sup_moment_eval(
                    G_val, X_val, P_val, Y_val, H_val, f_of_x_val_collection)
                eval_history.append(curr_eval)
                if min_eval > curr_eval:
                    min_eval = curr_eval
                    best_state = copy.deepcopy(self.beta_learner.state_dict())
            if (epoch + 1) % ckpt_every == 0 or epoch == n_epochs - 1:
                torch.save({"h456": self.h456.state_dict(),
                            "beta": self.beta_learner.state_dict(),
                            "adv": self.adversary.state_dict(),
                            "opt_learner": self.optimizer_learner.state_dict(),
                            "opt_adversary": self.optimizer_adversary.state_dict(),
                            "best_state": best_state,
                            "min_eval": min_eval,
                            "eval_history": eval_history,
                            "epoch": epoch,
                            "rng": _rng_state(),
                            "loader_g": self.train_dl.generator.get_state()},
                           main_ckpt)

        self.eval_history = eval_history
        # Persist the moment-criterion early-stopped model.
        self.beta_learner.load_state_dict(best_state)
        torch.save(self.beta_learner, os.path.join(resume_dir, "beta_earlystop.pt"))
        return self

    def predict(self, X, model='earlystop'):
        net = torch.load(os.path.join(self.resume_dir, "beta_%s.pt" % model), weights_only=False)
        net.eval()
        with torch.no_grad():
            return net(X).cpu().data.numpy()
