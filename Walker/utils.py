# utils.py

import numpy as np


class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x: np.ndarray):
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count

        self.mean = self.mean + delta * batch_count / tot_count
        self.var = np.maximum(M2 / tot_count, 1e-6)
        self.count = tot_count

    def normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        norm_x = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(norm_x, -10.0, 10.0)


def compute_gae(rewards, values, dones, next_value, gamma, lam):
    advs = np.zeros_like(rewards, dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        nxt_v = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * nxt_v * mask - values[t]
        last = delta + gamma * lam * mask * last
        advs[t] = last
    returns = advs + values
    return advs, returns
