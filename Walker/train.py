# train.py

import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import Config
from env import make_env
from agent import PPOAgent
from buffer import RolloutBuffer
from checkpoint import save_checkpoint, load_checkpoint


def train(checkpoint_path=None):
    cfg = Config()
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    # environment & normalizers
    env, obs_rms = make_env(cfg.ENV_NAME, cfg.SEED, cfg.HARDCORE)

    # agent & buffer
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0], cfg)
    buffer = RolloutBuffer()

    writer = SummaryWriter(log_dir=cfg.LOG_DIR)
    best_eval = -np.inf
    total_steps = 0

    # optionally resume
    start_epoch = 0
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            start_epoch, total_steps = load_checkpoint(
                checkpoint_path, agent, obs_rms, device
            )
            print(f"=> Resumed from epoch {start_epoch}, total_steps {total_steps}")
        else:
            print(f"=> No checkpoint found at {checkpoint_path}, starting fresh.")

    obs, _ = env.reset(seed=cfg.SEED)
    avg_eval = -np.inf

    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        ep_rewards, ep_lengths = [], []
        eps_collected = 0

        # collect rollouts
        while eps_collected < cfg.EPS_PER_EPOCH:
            ep_r, ep_len = 0.0, 0
            done = False
            while not done and ep_len < cfg.MAX_LEN:
                norm_o = obs_rms.normalize(obs)
                with torch.no_grad():
                    a, logp, v, mu, std = agent.model.act(
                        torch.tensor(norm_o, device=device).unsqueeze(0)
                    )
                a_np = a.cpu().numpy()[0]
                next_obs, raw_r, term, trunc, _ = env.step(a_np)
                done = term or trunc

                buffer.store(
                    obs,
                    norm_o,
                    a_np,
                    logp.cpu().item(),
                    v.cpu().item(),
                    raw_r,
                    float(done),
                    mu.cpu().numpy()[0],
                    std.cpu().numpy()[0],
                )

                obs = next_obs
                ep_r += raw_r
                ep_len += 1
                total_steps += 1

                if done:
                    eps_collected += 1
                    ep_rewards.append(ep_r)
                    ep_lengths.append(ep_len)
                    writer.add_scalar("episode/reward", ep_r, total_steps)
                    writer.add_scalar("episode/length", ep_len, total_steps)
                    obs, _ = env.reset(
                        seed=cfg.SEED + epoch * cfg.EPS_PER_EPOCH + eps_collected
                    )

        # update observation normalizer
        obs_rms.update(np.array(buffer.obs))

        # perform PPO update
        agent.update(buffer, epoch)

        # checkpointing
        if (epoch + 1) % cfg.SAVE_EVERY == 0:
            # evaluation
            eval_rewards = []
            for i in range(cfg.VAL_EPS):
                o, _ = env.reset(seed=cfg.SEED + epoch * cfg.EPS_PER_EPOCH + i)
                done, r_sum = False, 0.0
                while not done:
                    with torch.no_grad():
                        a, _, _, _, _ = agent.model.act(
                            torch.tensor(obs_rms.normalize(o), device=device).unsqueeze(
                                0
                            )
                        )
                    o, r, term, trunc, _ = env.step(a.cpu().numpy()[0])
                    done = term or trunc
                    r_sum += r
                eval_rewards.append(r_sum)

            avg_eval = np.mean(eval_rewards)
            save_checkpoint(
                (
                    f"checkpoints/ppo_epoch{epoch+1}.pth"
                    if not cfg.HARDCORE
                    else f"checkpoints_hc/ppo_epoch{epoch+1}.pth"
                ),
                agent,
                obs_rms,
                epoch + 1,
                total_steps,
            )
        if avg_eval > best_eval:
            best_eval = avg_eval
            save_checkpoint(
                (
                    "checkpoints/best_ppo.pth"
                    if not cfg.HARDCORE
                    else "checkpoints_hc/best_ppo.pth"
                ),
                agent,
                obs_rms,
                epoch + 1,
                total_steps,
            )

        writer.add_scalar("training/mean_val_reward", avg_eval, epoch)
        writer.add_scalar("training/mean_reward", np.mean(ep_rewards), epoch)
        writer.add_scalar("training/mean_ep_length", np.mean(ep_lengths), epoch)
        writer.add_scalar(
            "training/learning_rate", agent.opt.param_groups[0]["lr"], epoch
        )
        writer.add_scalar("training/entropy_coef", agent.ent_coef, epoch)
        print(
            f"Time: {time.strftime('%H:%M:%S')} | "
            f"Epoch: {epoch+1}/{cfg.NUM_EPOCHS} | "
            f"Train Reward: {np.mean(ep_rewards):.2f} | "
            f"Val Reward: {avg_eval:.2f} | "
            f"Mean Ep Length: {np.mean(ep_lengths):.2f}"
        )

    writer.close()
    env.close()


if __name__ == "__main__":
    train()
