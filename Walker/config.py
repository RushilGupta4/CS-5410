# config.py

from dataclasses import dataclass


@dataclass
class Config:
    # Environment
    ENV_NAME: str = "BipedalWalker-v3"
    SEED: int = 46

    # Training loop
    NUM_EPOCHS: int = 5000
    EPS_PER_EPOCH: int = 1
    VAL_EPS: int = 1
    MAX_LEN: int = 2048
    HARDCORE: bool = False

    # PPO update
    PPO_EPOCHS: int = 4
    MINI_BATCH: int = 64
    GAMMA: float = 0.99
    LAM: float = 0.95
    LR: float = 1e-4
    CLIP_EPS: float = 0.1

    # Loss coefficients
    ENT_COEF_START: float = 0.02
    ENT_COEF_END: float = 0.001

    VF_COEF: float = 0.5

    # Checkpointing & logging
    SAVE_EVERY: int = 10
    GRAD_NORM: float = 0.5
    LOG_DIR: str = "runs/ppo" if not HARDCORE else "runs/ppo_hc"

    # Device
    DEVICE: str = "cpu"
