# buffer.py

import numpy as np
import torch


class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.norm_obs = []
        self.acts = []
        self.logps = []
        self.vals = []
        self.rews = []
        self.dones = []
        self.old_mus = []
        self.old_stds = []

    def store(self, obs, norm_obs, act, logp, val, rew, done, old_mu, old_std):
        self.obs.append(obs)
        self.norm_obs.append(norm_obs)
        self.acts.append(act)
        self.logps.append(logp)
        self.vals.append(val)
        self.rews.append(rew)
        self.dones.append(done)
        self.old_mus.append(old_mu)
        self.old_stds.append(old_std)

    def clear(self):
        self.__init__()

    def get_tensors(self, device):
        return {
            "obs": torch.tensor(
                np.array(self.norm_obs), dtype=torch.float32, device=device
            ),
            "acts": torch.tensor(
                np.array(self.acts), dtype=torch.float32, device=device
            ),
            "old_logp": torch.tensor(
                np.array(self.logps), dtype=torch.float32, device=device
            ),
            "old_vals": torch.tensor(
                np.array(self.vals), dtype=torch.float32, device=device
            ),
            "old_mu": torch.tensor(
                np.array(self.old_mus), dtype=torch.float32, device=device
            ),
            "old_std": torch.tensor(
                np.array(self.old_stds), dtype=torch.float32, device=device
            ),
        }

    def __len__(self):
        return len(self.rews)
