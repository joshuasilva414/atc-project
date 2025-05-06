import gymnasium as gym
import bluesky_gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os

# Create log directory
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

# Register environments
bluesky_gym.register_envs()

# Training parameters
total_timesteps = 75000
eval_freq = 2500
n_eval_episodes = 5

# Custom callback to capture observations
class ObservationCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10, verbose=0):
        super(ObservationCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.observations = []
        self.timesteps = []
        self.rewards = []
        # Additional metrics to track
        self.faf_reach_values = []
        self.avg_drift_values = []
        self.total_intrusions_values = []
        self.model_name = None
        
        # Custom monitor to capture observations and metrics
        class ObservationMonitor(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)
                self.current_obs = None
                self.current_info = {}
                self.faf_reach = 0
                self.average_drift = 0
                self.total_intrusions = 0
                
            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.current_obs = obs
                self.current_info = info
                
                # Extract metrics from info dictionary if available
                if 'faf_reach' in info:
                    self.faf_reach = info['faf_reach']
                if 'average_drift' in info:
                    self.average_drift = info['average_drift']
                if 'total_intrusions' in info:
                    self.total_intrusions = info['total_intrusions']
                
                # Try to extract metrics from env attributes if available
                try:
                    if hasattr(self.env.unwrapped, 'faf_reach'):
                        self.faf_reach = self.env.unwrapped.faf_reach
                    if hasattr(self.env.unwrapped, 'average_drift'):
                        self.average_drift = self.env.unwrapped.average_drift
                    if hasattr(self.env.unwrapped, 'total_intrusions'):
                        self.total_intrusions = self.env.unwrapped.total_intrusions
                except:
                    pass
                
                return obs, reward, terminated, truncated, info
                
            def reset(self, **kwargs):
                obs, info = self.env.reset(**kwargs)
                self.current_obs = obs
                self.current_info = info
                self.faf_reach = 0
                self.average_drift = 0
                self.total_intrusions = 0
                return obs, info
        
        # Apply observation monitor to evaluation environment
        self.eval_env = ObservationMonitor(eval_env)
        
    def _init_callback(self):
        # Get the model name from the model class
        if isinstance(self.model, DDPG):
            self.model_name = "DDPG"
        elif isinstance(self.model, PPO):
            self.model_name = "PPO"
        elif isinstance(self.model, SAC):
            self.model_name = "SAC"
        else:
            self.model_name = "Unknown"
            
        # Create directory for storing observations
        os.makedirs(f"{log_dir}{self.model_name}_observations", exist_ok=True)
    
    def _on_step(self):
        # Execute this every eval_freq steps
        if self.n_calls % self.eval_freq == 0:
            try:
                # Run one episode in eval environment to get a reward and observations
                obs, _ = self.eval_env.reset()
                done = False
                total_reward = 0
                episode_obs = []
                # Final metrics for this episode
                final_faf_reach = 0
                final_avg_drift = 0
                final_total_intrusions = 0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                    
                    # Capture the current observation
                    if hasattr(self.eval_env, 'current_obs'):
                        episode_obs.append(self.eval_env.current_obs)
                    
                    # Update metrics from the observation monitor
                    if done:  # Capture final values
                        final_faf_reach = self.eval_env.faf_reach
                        final_avg_drift = self.eval_env.average_drift
                        final_total_intrusions = self.eval_env.total_intrusions
                
                # Store the middle observation from the episode (if available)
                if episode_obs:
                    mid_idx = len(episode_obs) // 2
                    self.observations.append(episode_obs[mid_idx])
                    self.timesteps.append(self.n_calls)
                    self.rewards.append(total_reward)
                    
                    # Store the metrics
                    self.faf_reach_values.append(final_faf_reach)
                    self.avg_drift_values.append(final_avg_drift)
                    self.total_intrusions_values.append(final_total_intrusions)
                    
                    # Save current data
                    self.save_data()
                    print(f"Metrics - FAF reach: {final_faf_reach}, Avg drift: {final_avg_drift}, Total intrusions: {final_total_intrusions}")
                
                if self.verbose > 0:
                    print(f"Step: {self.n_calls}, Model: {self.model_name}, Reward: {total_reward}, Obs captured: {len(episode_obs)}")
            except Exception as e:
                print(f"Error in custom callback for {self.model_name}: {e}")
        
        return True
    
    def save_data(self):
        try:
            # Save the observations, timesteps, and rewards
            np.save(f"{log_dir}{self.model_name}_observations/obs.npy", np.array(self.observations, dtype=object))
            np.save(f"{log_dir}{self.model_name}_observations/timesteps.npy", np.array(self.timesteps))
            np.save(f"{log_dir}{self.model_name}_observations/rewards.npy", np.array(self.rewards))
            
            # Save additional metrics
            np.save(f"{log_dir}{self.model_name}_observations/faf_reach.npy", np.array(self.faf_reach_values))
            np.save(f"{log_dir}{self.model_name}_observations/average_drift.npy", np.array(self.avg_drift_values))
            np.save(f"{log_dir}{self.model_name}_observations/total_intrusions.npy", np.array(self.total_intrusions_values))
        except Exception as e:
            print(f"Error saving data for {self.model_name}: {e}")
    
    def on_training_end(self):
        # Close the observation environment
        self.eval_env.env.close()
        return super().on_training_end()

# Function to train a model and return evaluation results
def train_model(model_class, model_name):
    # Create and wrap the environment
    env = gym.make('MergeEnv-v0')
    env = Monitor(env, log_dir + model_name)
    
    # Create the evaluation environment
    eval_env = gym.make('MergeEnv-v0')
    eval_env = Monitor(eval_env, log_dir + model_name + "_eval")
    
    # Create callback
    custom_callback = ObservationCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        verbose=1
    )
    
    # Initialize the model
    model = model_class("MultiInputPolicy", env)
    
    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=custom_callback, progress_bar=True)
    
    # Save the model
    model.save(path=f"{model_name}_final")
    
    return model, custom_callback

# Train models
models = {
    "DDPG": DDPG,
    "PPO": PPO,
    "SAC": SAC
}

trained_models = {}
callbacks = {}
for name, model_class in models.items():
    print(f"Training {name}...")
    trained_models[name], callbacks[name] = train_model(model_class, name)
    print(f"{name} training completed!")

# Plot training reward
def plot_training_results():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for model_name in models.keys():
        x, y = ts2xy(load_results(log_dir + model_name), 'timesteps')
        ax.plot(x, y, label=model_name)
    
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Rewards')
    ax.set_title('Training Rewards over Time')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{log_dir}training_rewards.png")
    
    # Plot evaluation rewards from custom callbacks
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for model_name in models.keys():
        try:
            timesteps = np.load(f"{log_dir}{model_name}_observations/timesteps.npy")
            rewards = np.load(f"{log_dir}{model_name}_observations/rewards.npy")
            ax.plot(timesteps, rewards, label=model_name)
        except Exception as e:
            print(f"Error plotting data for {model_name}: {e}")
    
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Evaluation Rewards')
    ax.set_title('Evaluation Rewards from Observations')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{log_dir}eval_rewards_from_obs.png")
    
    # Plot additional metrics
    metrics = {
        'faf_reach': 'FAF Reach',
        'average_drift': 'Average Drift',
        'total_intrusions': 'Total Intrusions'
    }
    
    for metric_file, metric_name in metrics.items():
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for model_name in models.keys():
                try:
                    timesteps = np.load(f"{log_dir}{model_name}_observations/timesteps.npy")
                    metric_values = np.load(f"{log_dir}{model_name}_observations/{metric_file}.npy")
                    ax.plot(timesteps, metric_values, label=model_name)
                except Exception as e:
                    print(f"Error plotting {metric_name} for {model_name}: {e}")
            
            ax.set_xlabel('Timesteps')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} over Time')
            ax.legend()
            plt.tight_layout()
            plt.savefig(f"{log_dir}{metric_file}.png")
        except Exception as e:
            print(f"Error creating {metric_name} plot: {e}")
    
    # Plot first dimension of observations over time for comparison
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for model_name in models.keys():
            try:
                timesteps = np.load(f"{log_dir}{model_name}_observations/timesteps.npy")
                observations = np.load(f"{log_dir}{model_name}_observations/obs.npy")
                
                # Check if observations is not empty and has valid shape
                if len(observations) > 0:
                    # If observation is a dict (common with MultiInputPolicy)
                    if isinstance(observations[0], dict):
                        # Use the first key/feature
                        first_key = list(observations[0].keys())[0]
                        first_dim_values = [obs[first_key][0] for obs in observations]
                    # If observation is a numpy array
                    else:
                        first_dim_values = [obs[0] for obs in observations]
                        
                    ax.plot(timesteps, first_dim_values, label=model_name)
            except Exception as e:
                print(f"Error plotting observations for {model_name}: {e}")
        
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('First Observation Dimension')
        ax.set_title('First Observation Dimension Over Time')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{log_dir}first_observation_dim.png")
    except Exception as e:
        print(f"Error creating observation plot: {e}")

# Plot results
plot_training_results()
print(f"Training complete! Results saved to {log_dir}")
