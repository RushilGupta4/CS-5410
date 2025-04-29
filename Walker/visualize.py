"""
visualize.py

Visualize a trained PPO agent interacting with a Gym environment.
"""

import os
import logging
import random

import gymnasium as gym
import numpy as np
import torch
import imageio  # Added for GIF saving

from config import Config
from env import make_env
from agent import PPOAgent
from checkpoint import load_checkpoint


def set_global_seed(seed: int):
    """Set seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def visualize(
    env_name: str,
    checkpoint_path: str,
    episodes: int,
    max_steps: int,
    seed: int,
    device: torch.device,
    hardcore: bool,
    save_gif: bool = False,
    gif_path: str = "visualization.gif",
):
    logging.info(f"Starting visualization: env={env_name}, seed={seed}")

    # Create environment and observation normalizer
    env, obs_rms = make_env(env_name, seed, hardcore)

    # Choose render mode based on save_gif parameter
    render_mode = "rgb_array" if save_gif else "human"
    render_env = gym.make(env_name, render_mode=render_mode, hardcore=hardcore)

    # Build agent and load checkpoint
    obs_dim = render_env.observation_space.shape[0]
    act_dim = render_env.action_space.shape[0]
    cfg = Config()
    agent = PPOAgent(obs_dim, act_dim, cfg)
    agent.model.to(device)
    agent.model.eval()

    if checkpoint_path:
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            load_checkpoint(checkpoint_path, agent, obs_rms, device)
            print(f"=> Loaded checkpoint from {checkpoint_path}")
        else:
            print("=> No checkpoint found at {checkpoint_path}, starting fresh.")

    # For GIF saving
    if save_gif:
        frames = []

    # Run episodes
    for ep in range(1, episodes + 1):
        obs, _ = render_env.reset(seed=seed + ep)
        total_reward = 0.0
        for step in range(1, max_steps + 1):
            norm_obs = obs_rms.normalize(obs)
            obs_tensor = torch.tensor(
                norm_obs, dtype=torch.float32, device=device
            ).unsqueeze(0)

            with torch.no_grad():
                a, _, _, _, _ = agent.model.act(obs_tensor)
                action = a.cpu().numpy()[0]

            obs, reward, terminated, truncated, _ = render_env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Capture frame for GIF if enabled
            if save_gif:
                frames.append(render_env.render())

            if done:
                break

        logging.info(
            f"Episode {ep}/{episodes}: reward={total_reward:.2f} in {step} steps"
        )

    # Save GIF if enabled
    if save_gif and frames:
        logging.info(f"Saving GIF to {gif_path}")
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"=> Saved visualization GIF to {gif_path}")

    render_env.close()


if __name__ == "__main__":
    # Load default configuration
    cfg = Config()
    ENV_NAME = cfg.ENV_NAME
    CHECKPOINT_PATH = (
        "checkpoints/best_ppo.pth"
        if not cfg.HARDCORE
        else "checkpoints_hc/best_ppo.pth"
    )

    file = sorted(
        [i for i in os.listdir("checkpoints") if "best" not in i],
        reverse=True,
        key=lambda x: int(x.split("_")[1].split(".")[0].replace("epoch", "")),
    )[0]
    CHECKPOINT_PATH = f"checkpoints/{file}"

    # CHECKPOINT_PATH = f"checkpoints_hc/ppo_epoch10000.pth"

    EPISODES = 1
    MAX_STEPS = 2048
    SEED = cfg.SEED + 200
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_GIF = True
    GIF_PATH = "walker_visualization.gif"  # Path to save the GIF file

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Seed all sources
    set_global_seed(SEED)

    # Run the visualization
    visualize(
        ENV_NAME,
        CHECKPOINT_PATH,
        EPISODES,
        MAX_STEPS,
        SEED,
        DEVICE,
        cfg.HARDCORE,
        SAVE_GIF,
        GIF_PATH,
    )
