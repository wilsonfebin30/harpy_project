#!/usr/bin/env python3
# Copyright (c) 2022-2025
# Adapted from Isaac Lab's play script for Harpy robot

"""Script to play/test a trained checkpoint of a Harpy RL agent."""

import argparse
import os
import sys
import yaml
import time
import torch
import math

# Set CUDA debugging environment variables for GPU stability
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PHYSX_GPU_TEMP_BUFFER_SIZE"] = "67108864"  # Increase buffer size (64MB)

# Parse arguments
parser = argparse.ArgumentParser(description="Play/test a trained Harpy RL agent checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--use_last_checkpoint", action="store_true", help="Use the last checkpoint when not specified.")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
parser.add_argument("--config", type=str, default=None, help="Path to training configuration file")
parser.add_argument("--device", type=str, default=None, help="Device to use (cpu or cuda:0)")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--duration", type=float, default=60.0, help="Duration to run in seconds")
args = parser.parse_args()

# Import SimulationApp - this needs to be after argument parsing
from isaacsim.simulation_app import SimulationApp

# Configure simulation app with optimized PhysX settings
sim_app_settings = {
    "headless": args.headless,
    "physics": {
        "gpu_found_lost_pairs_capacity": 16777216,  
        "gpu_total_aggregate_pairs_capacity": 16777216,
        "gpu_max_rigid_contact_count": 16777216,
        "gpu_max_rigid_patch_count": 4194304,       
        "enable_gpu_dynamics": True,                
        "enable_direct_gpu_api": False,            
        "gpu_temp_buffer_capacity": 67108864,      
        "gpu_max_num_partitions": 4,               
        "gpu_dynamics_debug_verification": False,   
        "gpu_dynamics_deterministic": False        
    }
}
app = SimulationApp(sim_app_settings)

# Import required modules
import torch
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.articulations import ArticulationView
from isaacsim.core.api.utils.torch.rotations import *
from isaacsim.core.api import SimulationContext

# Project paths
HOME_DIR = os.path.expanduser("~")
PROJECT_DIR = os.path.join(HOME_DIR, "harpy_project")
HARPY_USD_PATH = os.path.join(PROJECT_DIR, "assets", "harpy", "harpy.usd")
CONFIG_PATH = args.config if args.config else os.path.join(PROJECT_DIR, "config", "rlgames_config.yaml")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
LOG_DIR = os.path.join(PROJECT_DIR, "logs", "harpy_rlgames")

# Add your scripts to path
scripts_dir = os.path.join(PROJECT_DIR, "scripts")
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

# Import your custom Harpy environment
from harpy_ppo_training import HarpyEnv, RslRlVecEnvWrapper, load_config

# Import RL-Games components
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

print("Starting Harpy RL-Games play/test script...")


def get_checkpoint_path(checkpoint_path=None, use_last=False):
    """
    Find the checkpoint to load
    """
    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path):
            return checkpoint_path
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
    
    # Find the latest or the best checkpoint
    log_dirs = sorted([d for d in os.listdir(LOG_DIR) if os.path.isdir(os.path.join(LOG_DIR, d))])
    
    if not log_dirs:
        print(f"No checkpoints found in {LOG_DIR}")
        return None
    
    # Use the most recent log directory
    latest_log_dir = os.path.join(LOG_DIR, log_dirs[-1])
    nn_dir = os.path.join(latest_log_dir, "nn")
    
    if not os.path.exists(nn_dir):
        print(f"No 'nn' directory found in {latest_log_dir}")
        return None
    
    checkpoint_files = sorted([f for f in os.listdir(nn_dir) if f.endswith(".pth")])
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {nn_dir}")
        return None
    
    if use_last:
        # Use the latest checkpoint
        checkpoint_file = checkpoint_files[-1]
    else:
        # Use the final model or best model
        best_files = [f for f in checkpoint_files if "best" in f]
        if best_files:
            checkpoint_file = best_files[-1]
        else:
            checkpoint_file = checkpoint_files[-1]
    
    checkpoint_path = os.path.join(nn_dir, checkpoint_file)
    print(f"Using checkpoint: {checkpoint_path}")
    return checkpoint_path


def create_rlgames_wrapper(env, rl_device, clip_obs=10.0, clip_actions=1.0):
    """Create a wrapper for the Harpy environment to work with RL-Games."""
    class RlGamesGpuEnv:
        def __init__(self, config_name, num_actors, **kwargs):
            self.env = env
            self.num_actors = self.env.num_envs
            
        def step(self, actions):
            obs, rewards, dones, info = self.env.step(actions)
            return obs, rewards, dones, info
            
        def reset(self):
            return self.env.reset()
            
        def get_number_of_agents(self):
            return 1
            
        def get_env_info(self):
            return {
                'observation_space': self.env._num_observations,
                'action_space': self.env._num_actions,
                'agents': 1,
                'value_size': 1
            }
            
    class RlGamesVecEnvWrapper:
        def __init__(self, env, rl_device, clip_obs=10.0, clip_actions=1.0):
            self.env = env
            self.num_envs = self.env.num_envs
            self.device = rl_device
            self.clip_obs = clip_obs
            self.clip_actions = clip_actions
            
        def step(self, actions):
            if self.clip_actions > 0.0:
                actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
            
            obs, rewards, dones, info = self.env.step(actions)
            
            if self.clip_obs > 0.0:
                obs = torch.clamp(obs, -self.clip_obs, self.clip_obs)
                
            return obs, rewards, dones, info
            
        def reset(self):
            obs, info = self.env.reset()
            
            if self.clip_obs > 0.0:
                obs = torch.clamp(obs, -self.clip_obs, self.clip_obs)
                
            return obs, info
            
        def get_number_of_agents(self):
            return 1
            
        def get_env_info(self):
            return {
                'observation_space': self.env._num_observations,
                'action_space': self.env._num_actions,
                'agents': 1,
                'value_size': 1
            }
            
    return RlGamesGpuEnv, RlGamesVecEnvWrapper
    
    
def main():
    """Play with RL-Games agent."""
    # Set device
    sim_device = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(sim_device)
    print(f"Using device: {device}")
    
    # Load configuration
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            agent_cfg = yaml.safe_load(f)
        print(f"Loaded RL-Games configuration from {CONFIG_PATH}")
    else:
        print(f"Configuration file not found: {CONFIG_PATH}")
        agent_cfg = {"params": {
            "config": {
                "name": "Harpy",
                "env_name": "rlgpu",
                "device": sim_device,
                "device_name": sim_device.split(':')[0]
            }
        }}
    
    # Find checkpoint path
    checkpoint_path = get_checkpoint_path(args.checkpoint, args.use_last_checkpoint)
    if checkpoint_path is None:
        print("No checkpoint found to play. Please specify a checkpoint.")
        app.close()
        return
        
    # Update config with checkpoint
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoint_path
    print(f"Using checkpoint: {agent_cfg['params']['load_path']}")
    
    # Load environment configuration
    env_config = load_config(args.config)
    
    # Create environment
    print(f"Creating environment with {args.num_envs} environments")
    env = HarpyEnv(
        num_envs=args.num_envs,
        sim_device=sim_device,
        headless=args.headless,
        config=env_config
    )
    
    # Wrap for video recording if enabled
    if args.video:
        print("Recording videos during testing")
        video_folder = os.path.join(LOG_DIR, "videos", "play")
        os.makedirs(video_folder, exist_ok=True)
        
        import gymnasium as gym
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,  # Record from the start
            "video_length": args.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Configure for RL-Games
    clip_obs = agent_cfg["params"].get("env", {}).get("clip_observations", 10.0)
    clip_actions = agent_cfg["params"].get("env", {}).get("clip_actions", 1.0)
    
    # Create wrapper for RL-Games
    RlGamesGpuEnv, RlGamesVecEnvWrapper = create_rlgames_wrapper(
        env, sim_device, clip_obs, clip_actions)
    
    # Wrap environment
    env_wrapper = RlGamesVecEnvWrapper(env, sim_device, clip_obs, clip_actions)
    
    # Register the environment to RL-Games registry
    vecenv.register(
        "IsaacRlgWrapper", 
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env_wrapper})
    
    # Set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.num_envs
    
    # Create runner from RL-Games
    runner = Runner()
    runner.load(agent_cfg)
    
    # Create player
    player = runner.create_player()
    player.restore(checkpoint_path)
    player.reset()
    
    # Get physics timestep
    dt = 1.0/60.0  # Default physics timestep
    
    # Reset environment
    obs, _ = env.reset()
    
    # Initialize RNN states if used
    if player.is_rnn:
        player.init_rnn()
    
    # Record metrics
    episode_rewards = []
    episode_lengths = []
    current_reward = 0
    step_count = 0
    max_velocity = 0
    avg_height = 0
    
    print(f"Running evaluation for {args.duration} seconds or until completion...")
    timestep = 0
    total_time = 0.0
    
    # Run evaluation loop
    while app.is_running() and total_time < args.duration:
        start_time = time.time()
        
        # Run in inference mode
        with torch.inference_mode():
            # Convert obs to agent format
            obs_tensor = player.obs_to_torch(obs)
            
            # Get action from policy
            actions = player.get_action(obs_tensor, is_deterministic=True)
            
            # Step environment
            obs, rewards, dones, info = env.step(actions)
            
            # Update metrics
            current_reward += rewards.sum().item()
            step_count += 1
            
            # Track position and velocity
            root_pos, _ = env._harpies.get_world_poses()
            root_vel = env._harpies.get_velocities()
            current_height = root_pos[:, 2].mean().item()
            current_velocity = root_vel[:, 0].mean().item()
            
            avg_height += current_height
            max_velocity = max(max_velocity, current_velocity)
            
            # Handle episode termination
            if any(dones):
                # For terminated environments
                for i, done in enumerate(dones):
                    if done:
                        print(f"Episode {len(episode_rewards)+1} finished with reward: {current_reward}")
                        episode_rewards.append(current_reward)
                        episode_lengths.append(step_count)
                        
                        # Reset metrics
                        current_reward = 0
                        step_count = 0
                
                # Reset RNN state if needed
                if player.is_rnn and player.states is not None:
                    for s in player.states:
                        s[:, dones, :] = 0.0
        
        # Increment timestep
        timestep += 1
        
        # Print progress every second
        if timestep % 60 == 0:
            print(f"Time: {total_time:.1f}s | Height: {current_height:.2f}m | Velocity: {current_velocity:.2f} m/s")
        
        # Exit loop after recording one video if video enabled
        if args.video and timestep >= args.video_length:
            print("Video recording complete")
            break
        
        # Time delay for real-time rendering
        sleep_time = dt - (time.time() - start_time)
        if args.real_time and sleep_time > 0:
            time.sleep(sleep_time)
            
        total_time += dt
    
    # Calculate final metrics
    avg_reward = sum(episode_rewards) / max(1, len(episode_rewards))
    avg_length = sum(episode_lengths) / max(1, len(episode_lengths))
    avg_height = avg_height / max(1, timestep)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Total simulation time: {total_time:.2f}s")
    print(f"Episodes completed: {len(episode_rewards)}")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Average episode length: {avg_length:.1f}")
    print(f"Average robot height: {avg_height:.2f}m")
    print(f"Maximum velocity: {max_velocity:.2f}m/s")
    
    # Close environment
    env.close()
    print("Evaluation complete!")


if __name__ == "__main__":
    try:
        # Run main function
        main()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close the simulator app
        app.close()
        print("Application closed")
