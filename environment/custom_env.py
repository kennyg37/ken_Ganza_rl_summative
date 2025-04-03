import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

class CropMonitoringEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, grid_size=10, max_steps=100):
        super(CropMonitoringEnv, self).__init__()
        
        # Environment parameters
        self.grid_size = grid_size  
        self.max_steps = max_steps
        self.current_step = 0
        
        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Dict({
            "position": spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32),
            "health_map": spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32)
        })
        
        # Initialize the health map (crop health status)
        self.drone_pos = np.array([0, 0], dtype=np.int32)
        self.health_map = np.random.rand(grid_size, grid_size) * 0.5 + 0.6   
        
        # Generate weak points (unhealthy crops)
        self._generate_weak_points()
        
        # Visualization parameters
        self.viewer = None
        
    def _generate_weak_points(self):
        """Create areas with weak crops (low health values)."""
        num_weak_points = np.random.randint(3, 7)
        for _ in range(num_weak_points):
            x, y = np.random.randint(0, self.grid_size, size=2)
            radius = np.random.randint(1, 3)
            
            # Create circular areas of unhealthy crops
            for i in range(max(0, x-radius), min(self.grid_size, x+radius+1)):
                for j in range(max(0, y-radius), min(self.grid_size, y+radius+1)):
                    if math.sqrt((i-x)**2 + (j-y)**2) <= radius:
                        self.health_map[i, j] = np.random.rand() * 0.3  # Unhealthy (0-0.3)
    
    def reset(self, seed=None):
        """Reset the environment to its initial state."""
        super().reset(seed=seed)
        
        self.drone_pos = np.array([0, 0], dtype=np.int32)
        self.health_map = np.random.rand(self.grid_size, self.grid_size) * 0.5 + 0.5
        self._generate_weak_points()
        self.current_step = 0

        return {
            "position": self.drone_pos.copy(),
            "health_map": self.health_map.copy()
        }, {}

    def step(self, action):
        """Apply the given action and return the new state, reward, and done flags."""
        terminated = False
        truncated = False
        reward = 0

        prev_pos = self.drone_pos.copy()

        # Define movement actions: 0 = up, 1 = right, 2 = down, 3 = left, 4 = scan
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        
        if isinstance(action, np.ndarray):
            action = int(action.item())

        if action in moves:
            new_pos = self.drone_pos + np.array(moves[action])

            # Check if within bounds
            if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
                self.drone_pos = new_pos
                reward -= 0.1  # Small movement penalty
            else:
                reward -= 0.2  # Invalid move penalty

        elif action == 4:  # Scan action
            health_value = self.health_map[self.drone_pos[0], self.drone_pos[1]]
            
            # Only give reward if the crop is unhealthy (health < 0.7)
            if health_value < 0.7:
                # Reward based on how unhealthy the crop is
                reward += 5 * (0.7 - health_value)
                
                # Improve crop health slightly
                self.health_map[self.drone_pos[0], self.drone_pos[1]] = min(
                    1.0, self.health_map[self.drone_pos[0], self.drone_pos[1]] + 0.05
                )
            else:
                # Small penalty for scanning healthy crops
                reward -= 0.5

        # Encourage movement toward weak crops
        weakest_spot = np.unravel_index(np.argmin(self.health_map), self.health_map.shape)
        distance_before = np.linalg.norm(prev_pos - np.array(weakest_spot))
        distance_after = np.linalg.norm(self.drone_pos - np.array(weakest_spot))

        if distance_after < distance_before:
            reward += 0.2  # Small reward for moving toward weak crops

        # Check termination conditions
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
            
            # Add final reward based on average crop health
            avg_health = np.mean(self.health_map)
            reward += 10 * avg_health  # Reward for maintaining good crop health

        observation = {
            "position": self.drone_pos.copy(),
            "health_map": self.health_map.copy()
        }

        return observation, reward, terminated, truncated, {}
    
    def render(self, mode='human'):
        if mode == 'human':
            if not hasattr(self, '_renderer'):
                from environment.rendering import Renderer
                self._renderer = Renderer(self)
            self._renderer.render() 

    def close(self):
        """Close the environment."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None