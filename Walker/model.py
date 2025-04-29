import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(input_dim, output_dim, hidden_sizes, activation=nn.ReLU):
    layers = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128, num_layers=2):
        """
        A decoupled Actor-Critic with separate MLPs for policy (actor) and value (critic).
        """
        super().__init__()
        hidden_sized = [hidden] * num_layers
        self.actor_body = mlp(obs_dim, act_dim, hidden_sized, activation=nn.ReLU)

        self.log_scaler = mlp(obs_dim, act_dim, hidden_sized, activation=nn.ReLU)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic_body = mlp(obs_dim, 1, hidden_sized, activation=nn.ReLU)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # smaller gain for output heads
                gain = np.sqrt(2)
                nn.init.orthogonal_(m.weight, gain)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # actor forward
        mu = self.actor_body(x)

        # scale is around 1/2 to 2 (most of sigmoid output is around 0.5, so we get this)
        log_scale = nn.functional.sigmoid(self.log_scaler(x)) * 1.8 - 0.9
        log_std = (self.log_std * log_scale).squeeze(0)
        std = torch.exp(log_std).expand_as(mu).clamp(min=1e-6, max=10.0)

        # critic forward
        v = self.critic_body(x)

        return mu, std, v.squeeze(-1)

    def act(self, obs):
        mu, std, v = self(obs)
        dist = Normal(mu, std)
        z = dist.rsample()
        a = torch.tanh(z)
        # log prob with tanh correction
        logp = (dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1)
        return a, logp, v, mu, std

    def evaluate(self, obs, actions):
        mu, std, v = self(obs)
        dist = Normal(mu, std)
        # invert tanh
        eps = 1e-6
        pre_tanh = torch.atanh(torch.clamp(actions, -1 + eps, 1 - eps))
        logp = (dist.log_prob(pre_tanh) - torch.log(1 - actions.pow(2) + eps)).sum(-1)
        ent = dist.entropy().sum(-1)
        return logp, ent, v
