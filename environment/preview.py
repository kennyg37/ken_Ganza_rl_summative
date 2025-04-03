import time
import numpy as np
from custom_env import CropMonitoringEnv

def main():
    env = CropMonitoringEnv(grid_size=10, max_steps=200)
    env.render()  # This will initialize the window
    
    try:
        obs, _ = env.reset()
        for step in range(300):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()
            time.sleep(0.1)
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()

if __name__ == "__main__":
    main()