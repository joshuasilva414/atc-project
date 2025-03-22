import gymnasium as gym
import bluesky_gym

bluesky_gym.register_envs()

env = gym.make('MergeEnv-v0', render_mode='human')

obs, info = env.reset()
done = truncated = False
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(info)
    env.render()

env.close()