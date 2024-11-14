# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import argparse

import carb
import torch as th
import numpy as np

from stable_baselines3 import SAC, PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback

from dataclasses import asdict, dataclass
from configs.main_config import MainConfig
# from omni.isaac.gym.vec_env import VecEnvBase
# env = VecEnvBase(headless=False)
from omni.isaac.gym.vec_env import VecEnvBase
# my_env = VecEnvBase(headless=False)
# from tasks.reach import AlohaTask
# task = AlohaTask(name="Aloha", n_envs=2)
# my_env.set_task(task, backend="torch")
from stable_baselines3.common.utils import set_random_seed
import gymnasium
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from  stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# def make_env(env_id: str, rank: int, seed: int = 0):
#     """
#     Utility function for multiprocessed env.

#     :param env_id: the environment ID
#     :param num_env: the number of environments you wish to have in subprocesses
#     :param seed: the inital seed for RNG
#     :param rank: index of the subprocess
#     """
#     def _init():
#         env = gymnasium.make(env_id, render_mode="human")
#         env.reset(seed=seed + rank)
#         return env
#     set_random_seed(seed)
#     return _init
config = MainConfig()
log_dir = asdict(config).get('train_log_dir', None)
if __name__ == "__main__":
    env_id = "tasks:AlphaBaseEnv-v0"
    env_id_cartpole = "CartPole-v1"
    log_dir = asdict(config).get('train_log_dir', None)
    num_cpu = 2  # Number of processes to use
    # Create the vectorized environment
    #vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    # env = gymnasium.make(env_id)
    # env = DummyVecEnv([lambda: env])
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=log_dir, name_prefix="multy_move_task")
    env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv)
    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
    total_timesteps = 2500000
    model = SAC("MlpPolicy", env, train_freq=1, gradient_steps=2, verbose=1)
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    #model.save(log_dir + "/SAC_policy")
    env.close()

# tuning = False
# config = MainConfig()
# log_dir = asdict(config).get('train_log_dir', None)

# #env = SubprocVecEnv([lambda: gymnasium.make("tasks:AlphaBaseEnv", config=config)])

# #env = make_vec_env("tasks:AlphaBaseEnv", n_envs=1, seed=0, vec_env_cls=DummyVecEnv)#SubprocVecEnv with CartPole-v1 also not working
# # Wrap the VecEnv
# #env = VecExtractDictObs(env, key="observation")
# #env = gymnasium.make("tasks:AlphaBaseEnv", config=config)
# #env = gymnasium.vector.make("tasks:AlphaBaseEnv", num_envs=2)
# #env = VecEnvWrapper([lambda: gymnasium.make("tasks:AlphaBaseEnv", config=config)], num_envs=2)
# #env = gymnasium.make_vec("tasks:AlphaBaseEnv", num_envs=2, vectorization_mode="sync")
# #env = AlphaBaseEnv()
# checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=log_dir, name_prefix="gen1f5_move_task")

# total_timesteps = 2500000

# model = PPO("MlpPolicy", env, verbose=1,tensorboard_log=log_dir,)

# model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

# model.save(log_dir + "/SAC_policy")

# env.close()
