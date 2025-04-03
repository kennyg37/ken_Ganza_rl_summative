import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from environment.custom_env import CropMonitoringEnv

class CropMonitoringWrapper(gym.Wrapper):
    """
    Integrated wrapper for PPO training
    Converts dict observations to flattened arrays and normalizes values
    """
    def __init__(self, env):
        super().__init__(env)
        # Consistent observation space with DQN
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(2 + env.grid_size**2,),  # pos + flattened health_map
            dtype=np.float32
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_obs(obs), reward, terminated, truncated, info
    
    def _process_obs(self, obs_dict):
        """Convert dict observation to flattened array"""
        position = obs_dict["position"] / self.env.grid_size  # Normalized to [0,1]
        health_map = obs_dict["health_map"].flatten()
        return np.concatenate([position, health_map])

def create_env():
    """Environment creation function for make_vec_env"""
    env = CropMonitoringEnv(grid_size=10)
    return CropMonitoringWrapper(env)

def train_ppo():
    env = make_vec_env(
        create_env,
        n_envs=4,
        seed=42
    )
    
    # Setup model saving
    os.makedirs("models/ppo", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,  
        save_path="models/ppo",
        name_prefix="ppo_model"
    )
    
    # PPO Model configuration
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,  
        n_steps=4096,  
        batch_size=128,  
        n_epochs=20,  
        gamma=0.99,  
        gae_lambda=0.95,  
        clip_range=0.2, 
        ent_coef=0.01,  
        vf_coef=0.5,  
        max_grad_norm=0.5,  
        target_kl=0.01, 
        tensorboard_log="logs/ppo"
    )

    # Training
    print("Starting PPO training...")
    model.learn(
        total_timesteps=1000000, 
        callback=checkpoint_callback,
        tb_log_name="ppo_run"
    )
    
    # Save final model
    model.save("models/ppo/ppo_final")
    print("Training completed. Model saved.")
    return model

if __name__ == "__main__":
    trained_model = train_ppo()