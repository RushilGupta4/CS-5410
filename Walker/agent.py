# agent.py

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import ActorCritic
from utils import compute_gae


class PPOAgent:
    def __init__(self, obs_dim, act_dim, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(obs_dim, act_dim).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=cfg.LR, eps=1e-5)
        self.sched = CosineAnnealingLR(self.opt, T_max=cfg.NUM_EPOCHS)

        self.clip_eps = cfg.CLIP_EPS
        self.vf_coef = cfg.VF_COEF

        self.ent_coef = cfg.ENT_COEF_START
        self.gamma, self.lam = cfg.GAMMA, cfg.LAM

    def update(self, buffer, epoch):
        # compute last value for GAE
        with torch.no_grad():
            norm_o = torch.tensor(
                buffer.norm_obs[-1], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            _, _, last_v = self.model(norm_o)
            last_v = last_v.cpu().item()

        advs, rets = compute_gae(
            np.array(buffer.rews),
            np.array(buffer.vals),
            np.array(buffer.dones),
            last_v,
            self.gamma,
            self.lam,
        )
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        batch = buffer.get_tensors(self.device)
        batch["rets"] = torch.tensor(rets, dtype=torch.float32, device=self.device)
        batch["advs"] = torch.tensor(advs, dtype=torch.float32, device=self.device)

        idxs = np.arange(len(advs))
        for _ in range(self.cfg.PPO_EPOCHS):
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), self.cfg.MINI_BATCH):
                mb = idxs[start : start + self.cfg.MINI_BATCH]

                # compute new log‐prob and value as before
                logp, ent, v = self.model.evaluate(batch["obs"][mb], batch["acts"][mb])
                ratio = (logp - batch["old_logp"][mb]).exp()

                # policy loss
                unclipped = ratio * batch["advs"][mb]
                clipped = (
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    * batch["advs"][mb]
                )
                actor_loss = -torch.min(unclipped, clipped).mean()

                # value loss
                old_v = batch["old_vals"][mb]
                v_clipped = old_v + torch.clamp(
                    v - old_v, -self.clip_eps, self.clip_eps
                )
                loss_un = (v - batch["rets"][mb]).pow(2)
                loss_cl = (v_clipped - batch["rets"][mb]).pow(2)
                value_loss = 0.5 * torch.max(loss_un, loss_cl).mean()

                loss = (
                    actor_loss + self.vf_coef * value_loss - self.ent_coef * ent.mean()
                )

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.GRAD_NORM
                )
                self.opt.step()

        # adjust entropy coefficient
        self.ent_coef = self.cfg.ENT_COEF_START + (
            self.cfg.ENT_COEF_END - self.cfg.ENT_COEF_START
        ) * (epoch / (self.cfg.NUM_EPOCHS - 1))

        self.sched.step()
        buffer.clear()
