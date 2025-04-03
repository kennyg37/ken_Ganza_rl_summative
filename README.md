# Crop Monitoring Drone RL Project

This project implements reinforcement learning algorithms for training autonomous drones to monitor crops efficiently. The project compares two different RL approaches: Deep Q-Network (DQN) and Proximal Policy Optimization (PPO).

## Project Structure

```
project_root/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment implementation
│   ├── rendering.py             # Visualization components using PyOpenGL
├── training/
│   ├── dqn_training.py          # Training script for DQN using SB3
│   ├── pg_training.py           # Training script for PPO using SB3
├── models/
│   ├── dqn/                     # Saved DQN models
│   └── pg/                      # Saved policy gradient models
├── main.py                      # Entry point for running experiments
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Environment Description

The environment simulates a crop field where a drone must:
- Navigate through a 3D space
- Monitor different crop sections
- Avoid obstacles (trees, buildings)
- Optimize battery usage
- Collect data from various monitoring points

### State Space
- Drone position (x, y, z)
- Drone orientation (roll, pitch, yaw)
- Battery level
- Distance to nearest monitoring point
- Obstacle detection data

### Action Space
- Discrete actions:
  - Move forward/backward
  - Move left/right
  - Move up/down
  - Rotate left/right
  - Hover
  - Start/stop monitoring

### Rewards
- Positive rewards:
  - Successfully monitoring a crop section
  - Efficient path planning
  - Battery conservation
- Negative rewards:
  - Collisions
  - Missing monitoring points
  - Excessive battery usage

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train DQN model:
```bash
python training/dqn_training.py
```

2. Train PPO model:
```bash
python training/pg_training.py
```

3. Run visualization:
```bash
python main.py --visualize
```

## Requirements

- Python 3.8+
- Gymnasium
- Stable-Baselines3
- PyOpenGL
- NumPy
- PyTorch

## Results

The project compares the performance of DQN and PPO algorithms in terms of:
- Training convergence
- Average rewards
- Path optimization
- Battery efficiency
- Monitoring coverage

Detailed results and comparisons are available in the project report.