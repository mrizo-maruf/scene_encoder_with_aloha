
import argparse

import carb
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from dataclasses import asdict, dataclass
from configs.main_config import MainConfig
import gymnasium

config = MainConfig()
log_dir = asdict(config).get('train_log_dir', None)
env = gymnasium.make("tasks:rlmodel-v1", config=config)
load_policy = asdict(config).get('load_policy', None)
model = SAC.load(load_policy,verbose=1,tensorboard_log=log_dir,)

for _ in range(100):
    obs, info = env.reset()
    done = False
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

env.close()