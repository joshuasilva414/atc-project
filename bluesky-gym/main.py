import gymnasium as gym
import bluesky_gym
from stable_baselines3 import DDPG
bluesky_gym.register_envs()

# env = gym.make('MergeEnv-v0', render_mode='human')
env = gym.make('MergeEnv-v0')

model = DDPG("MultiInputPolicy", env)
model.learn(total_timesteps=2e3)
model.save(path="ddpg_test")
