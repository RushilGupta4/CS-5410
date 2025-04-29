# checkpoint.py

import os
import torch


def save_checkpoint(path, agent, obs_rms, epoch, total_steps):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

    ckpt = {
        "model_state": agent.model.state_dict(),
        "opt_state": agent.opt.state_dict(),
        "sched_state": agent.sched.state_dict(),
        "obs_rms": {
            "mean": obs_rms.mean,
            "var": obs_rms.var,
            "count": obs_rms.count,
        },
        "epoch": epoch,
        "total_steps": total_steps,
    }
    torch.save(ckpt, path)


def load_checkpoint(path, agent, obs_rms, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    agent.model.load_state_dict(ckpt["model_state"])
    agent.opt.load_state_dict(ckpt["opt_state"])
    agent.sched.load_state_dict(ckpt["sched_state"])

    o = ckpt["obs_rms"]
    obs_rms.mean, obs_rms.var, obs_rms.count = o["mean"], o["var"], o["count"]

    return ckpt["epoch"], ckpt["total_steps"]
