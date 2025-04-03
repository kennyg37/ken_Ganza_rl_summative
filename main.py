import os
import time
import numpy as np
from stable_baselines3 import DQN, PPO
from environment.custom_env import CropMonitoringEnv
from gymnasium import spaces

class ObsWrapper:
    """Observation wrapper (works for both DQN and PPO)"""
    def __init__(self, env, grid_size=10):
        self.env = env
        self.grid_size = grid_size
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(2 + grid_size*grid_size,), 
            dtype=np.float32
        )
        
    def reset(self):
        obs, info = self.env.reset()
        return self._flatten_obs(obs), info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs(obs), reward, terminated, truncated, info
        
    def _flatten_obs(self, obs_dict):
        position = obs_dict["position"] / self.grid_size
        health_map = obs_dict["health_map"].flatten()
        return np.concatenate([position, health_map])

    def render(self):
        return self.env.render()

def run_agent(model_type="dqn", model_path=None, render=True, num_episodes=1):
    """Run either DQN or PPO agent"""
    env = CropMonitoringEnv(grid_size=10)
    wrapped_env = ObsWrapper(env)
    
    # Load appropriate model
    if model_type.lower() == "dqn":
        model = DQN.load(model_path or "models/dqn/dqn_model_500000_steps.zip")
    elif model_type.lower() == "ppo":
        model = PPO.load(model_path or "models\ppo\ppo_model_800000_steps.zip")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    for episode in range(num_episodes):
        obs, _ = wrapped_env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = wrapped_env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            if render:
                wrapped_env.render()
                time.sleep(0.1)
        
        print(f"Episode {episode + 1}/{num_episodes}, Total reward: {total_reward}")
    
    env.close()

def select_model_interactively():
    """Prompt user to select which model to run"""
    print("\n" + "="*40)
    print("Select Model to Run:")
    print("1. DQN (Value-Based)")
    print("2. PPO (Policy Gradient)")
    print("="*40)
    
    while True:
        choice = input("Enter your choice (1 or 2): ")
        if choice == "1":
            return "dqn"
        elif choice == "2":
            return "ppo"
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Crop Monitoring RL Agent")
    parser.add_argument("--model_type", choices=["dqn", "ppo"], help="Type of model to run")
    parser.add_argument("--model_path", help="Path to model file")
    parser.add_argument("--no-render", action="store_false", dest="render")
    parser.add_argument("--num_episodes", type=int, default=1)
    
    args = parser.parse_args()
    
    # If model type not specified, prompt user
    model_type = args.model_type or select_model_interactively()
    
    run_agent(
        model_type=model_type,
        model_path=args.model_path,
        render=args.render,
        num_episodes=args.num_episodes
    )