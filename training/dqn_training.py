import os
from stable_baselines3 import DQN # type: ignore
from stable_baselines3.common.env_util import make_vec_env # type: ignore
from stable_baselines3.common.callbacks import CheckpointCallback
from environment.custom_env import CropMonitoringEnv
import numpy as np
import gymnasium as gym 

class CropMonitoringWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Modify observation space to be SB3-friendly
        self.observation_space = gym.spaces.Box(
            low=0, high=1, 
            shape=(2 + env.grid_size * env.grid_size,),  # position + health_map
            dtype=np.float32
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_obs(obs), reward, terminated, truncated, info
    
    def _process_obs(self, obs_dict):
        # Flatten and normalize observation
        position = obs_dict["position"] / self.env.grid_size
        health_map = obs_dict["health_map"].flatten()
        return np.concatenate([position, health_map])

def train_dqn():
    # Create environment
    env = CropMonitoringWrapper(CropMonitoringEnv(grid_size=10))
    
    # Create save directory
    os.makedirs("models/dqn", exist_ok=True)
    
    # Callback to save checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/dqn",
        name_prefix="dqn_model"
    )
    
    # Create DQN model
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,  
        buffer_size=200000,  
        learning_starts=20000,  
        batch_size=128,  
        tau=0.005,  
        gamma=0.99, 
        train_freq=8,  
        gradient_steps=4,  
        target_update_interval=8000,  
        exploration_fraction=0.2, 
        exploration_final_eps=0.01,  
        tensorboard_log="logs/dqn"
    )

    
    # Train the model
    model.learn(
        total_timesteps=1000000,
        callback=checkpoint_callback,
        tb_log_name="dqn"
    )
    
    # Save final model
    model.save("models/dqn/dqn_final")
    
    return model

if __name__ == "__main__":
    trained_model = train_dqn()