import bluesky_gym
from stable_baselines3 import DDPG, PPO, SAC
from gymnasium.utils.save_video import save_video
import gymnasium as gym

# Register environments
bluesky_gym.register_envs()

env = gym.make('MergeEnv-v0', render_mode='human')

model = SAC("MultiInputPolicy", env)
model = model.load("./SAC_final.zip")

done = truncated = False
obs, info = env.reset()
while not (done or truncated):
    action = model(obs)
    obs_, reward_, done, truncated, info = env.step(action)
    save_video(
        frames=env.render()
        video_folder="vids",
        fps=30
        step_starting_index=0
        episode_index=1
    )
    obs = obs_