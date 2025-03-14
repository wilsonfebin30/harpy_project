# Harpy Robot Training Project

This repository contains code for training a bipedal robot called Harpy using reinforcement learning in Isaac Sim. The project implements Proximal Policy Optimization (PPO) to teach the robot to walk with a stable gait.

## Problem Description

I've been encountering RTX warnings and CUDA errors during training that appear as:
```
[Warning] [rtx.scenedb.plugin] SceneDbContext : TLAS limit buffer size 8448000128
[Warning] [rtx.scenedb.plugin] SceneDbContext : TLAS limit : valid false, within: false
[Warning] [rtx.scenedb.plugin] SceneDbContext : TLAS limit : decrement: 167690, decrement size: 8363520384
```

These errors are related to conflicts between the NVIDIA RTX rendering pipeline and PhysX GPU physics simulation. I've created optimized versions of the training scripts that resolve these issues by properly configuring the GPU memory allocation and disabling direct GPU API access.

## Setup Requirements

- Isaac Sim 4.5.0
- Python 3.8+
- PyTorch
- NVIDIA GPU with CUDA support

## Repository Structure

### Core Files

- **`harpy_ppo_training.py`**: Main implementation with all the core classes (HarpyEnv, PPOTrainer, RobustArticulationView)
- **`optimized_run_training.py`**: Optimized training script that fixes RTX/CUDA issues (USE THIS ONE!)

### Configuration Files

- **`training_config.yaml`**: PPO algorithm and environment settings
- **`rlgames_config.yaml`**: Configuration for the RL-Games framework

### Helper Scripts

- **`fix_physics_settings.py`**: Utility to fix PhysX GPU settings
- **`harpy_reward_system.py`**: The reward calculation system

### Testing & Evaluation Scripts

- **`harpy_policy_test.py`**: For testing trained models
- **`harpy_play.py`**: For visualizing trained agent behavior

### Debugging Tools

- **`basic_debug.py`**: To help identify where issues occur
- **`minimal_test.py`**: Minimal test script for isolating problems

### Fixed Training Scripts

- **`fixed_harpy_gpu.py`**: The most comprehensive fix for GPU issues
- **`harpy_minimal.py`**: Simplified training that avoids common issues

## How to Run Training

Use the optimized training script:

```bash
./python.sh ~/harpy_project/scripts/optimized_run_training.py --num_envs 1 --iterations 10 --device cuda:0 --headless
```

Parameters:
- `--num_envs`: Number of parallel environments (start with 1)
- `--iterations`: Number of training iterations
- `--device`: Computing device (cuda:0 for GPU, cpu for CPU)
- `--headless`: Run without visualization (optional)

## Fixing Common Issues

If you encounter CUDA errors:

1. Use the `fix_physics_settings.py` script to apply optimized PhysX settings:
   ```python
   from fix_physics_settings import fix_physics_settings, optimize_renderer_settings
   
   # Apply fixes after creating SimulationApp
   fix_physics_settings()
   optimize_renderer_settings()
   ```

2. Try the most aggressive fix with `fixed_harpy_gpu.py` which has comprehensive error handling:
   ```bash
   ./python.sh ~/harpy_project/scripts/fixed_harpy_gpu.py --num_envs 1 --iterations 10 --headless
   ```

3. If all else fails, you can use CPU physics as a fallback:
   ```bash
   ./python.sh ~/harpy_project/scripts/optimized_run_training.py --device cpu --num_envs 1 --iterations 10 --headless
   ```

## Testing a Trained Model

After training, you can test the trained model with:

```bash
./python.sh ~/harpy_project/scripts/harpy_policy_test.py --model ~/harpy_project/models/harpy_ppo_model_X.pt
```

Or run a trained policy with:

```bash
./python.sh ~/harpy_project/scripts/harpy_play.py --checkpoint ~/harpy_project/models/harpy_ppo_model_X.pt
```

## Project Structure and Dependencies

The code follows this workflow:
1. Configuration is defined in the YAML files
2. `harpy_ppo_training.py` contains the core implementation classes
3. Training is executed through `optimized_run_training.py` or other scripts
4. Evaluation is done with the test/play scripts

The most important dependency is the `RobustArticulationView` class which properly handles GPU memory for the robot articulations.

## Contributing

If you find additional fixes or improvements, please submit a pull request!
