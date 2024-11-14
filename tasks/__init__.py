from tasks.env_yolo import AlphaBaseEnv
from gymnasium.envs.registration import register
print("register")

register(
     id="AlphaBaseEnv-v0",
     entry_point="tasks.env_yolo:AlphaBaseEnv",
     max_episode_steps=512,
)

register(
     id="AlphaBaseEnv-v3",
     entry_point="tasks.env_dyolo:AlphaBaseEnv",
     max_episode_steps=512,
)

register(
     id="AlphaBaseEnv-v1",
     entry_point="tasks.env_n_simple:AlphaBaseEnv",
     max_episode_steps=512,
)

register(
     id="AlphaBaseEnv-v2",
     entry_point="tasks.env_simple:AlphaBaseEnv",
     max_episode_steps=512,
)