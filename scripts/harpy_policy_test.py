#!/usr/bin/env python3

"""
Test script for evaluating and deploying a trained PPO policy for the Harpy robot
Compatible with the actual harpy project file structure
"""

import os
import torch
import numpy as np
import omni
from isaacsim.core.api import SimulationContext
from isaacsim.core.api.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.api.articulations import ArticulationView
from isaacsim.core.api.utils.torch.rotations import *
import argparse

# Get script dir and add to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import ActorCritic model from training script
from harpy_ppo_training import ActorCritic

# Project paths (based on provided screenshots)
HOME_DIR = os.path.expanduser("~")
PROJECT_DIR = os.path.join(HOME_DIR, "harpy_project")
HARPY_USD_PATH = os.path.join(PROJECT_DIR, "assets", "harpy", "harpy.usd")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

class HarpyPolicyTest:
    def __init__(self, model_path, headless=False):
        self.model_path = model_path
        self.headless = headless
        
        # Set up simulation
        self.sim_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.sim_device)
        self.sim = SimulationContext(physics_dt=1.0/60.0, rendering_dt=1.0/60.0, device=self.sim_device)
        
        # Policy parameters
        self.num_observations = 47
        self.num_actions = 14
        
        # Initialize simulation
        self._setup_scene()
        
        # Load policy
        self._load_policy()
    
    def _setup_scene(self):
        """Setup test harpy in existing scene"""
        # Using existing physics scene and ground plane
        
        # Add test harpy robot - use a different path to avoid conflicting with existing robot
        test_harpy_path = "/World/test_harpy"
        
        # Add reference if doesn't exist
        stage = get_current_stage()
        if not stage.GetPrimAtPath(test_harpy_path):
            add_reference_to_stage(HARPY_USD_PATH, test_harpy_path)
        
        # Create ground plane if it doesn't exist
        ground_path = "/World/groundPlane"
        if not stage.GetPrimAtPath(ground_path):
            from harpy_ppo_training import create_ground_plane
            create_ground_plane(stage, ground_path, size=100, color=(0.3, 0.3, 0.3))
        
        # Position robot in a clear area
        transform = omni.isaac.core.utils.stage.get_prim_at_path(test_harpy_path).GetAttribute("xformOp:translate")
        transform.Set((5.0, 5.0, 0.5))  # 0.5m above ground, offset from origin
        
        # Create view for robot
        self.harpy = ArticulationView(
            prim_paths_expr=test_harpy_path,
            name="test_harpy_view",
        )
        
        # Initialize physics
        self.sim.initialize_physics()
        self.harpy.initialize()
        
        print("Test harpy initialized in scene")
        print(f"Using harpy USD from: {HARPY_USD_PATH}")
    
    def _load_policy(self):
        """Load trained policy from checkpoint"""
        # Initialize policy network
        self.policy = ActorCritic(self.num_observations, self.num_actions).to(self.device)
        
        # Load trained weights
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # New format
                self.policy.load_state_dict(checkpoint['model_state_dict'])
                iteration = checkpoint.get('iteration', 'unknown')
                print(f"Loaded policy from {self.model_path} (iteration: {iteration})")
            else:
                # Old format (direct state dict)
                self.policy.load_state_dict(checkpoint)
                print(f"Loaded policy from {self.model_path}")
            
            # Set to evaluation mode
            self.policy.eval()
        else:
            print(f"Error: Model file {self.model_path} not found!")
            raise FileNotFoundError(f"Model file {self.model_path} not found")
    
    def _get_observations(self):
        """Get observations from the test robot"""
        # Get robot state
        root_pos, root_rot = self.harpy.get_world_poses()
        root_velocities = self.harpy.get_velocities()
        root_lin_vel = root_velocities[:, 0:3]
        root_ang_vel = root_velocities[:, 3:6]
        
        # Get joint positions and velocities
        dof_pos = self.harpy.get_joint_positions()
        dof_vel = self.harpy.get_joint_velocities()
        
        # Project gravity vector into robot frame
        down_dir = get_basis_vector(root_rot, torch.tensor(2, device=root_rot.device))
        
        # Construct observation vector
        obs = torch.cat(
            (
                root_pos,                      # 3
                root_rot,                      # 4 (quaternion)
                root_lin_vel,                  # 3
                root_ang_vel,                  # 3
                dof_pos,                       # num_dof
                dof_vel,                       # num_dof
                down_dir,                      # 3
            ),
            dim=-1,
        )
        
        return obs
    
    def reset_robot(self):
        """Reset the test robot to initial position"""
        # Reset robot pose with stable starting position
        dof_pos = torch.zeros((1, self.harpy.num_dof), device=self.device)
        
        # Apply small knee bend for stability
        knee_indices = [2, 9]  # Adjust these based on your joint order
        bend_angle = 0.3  # Small bend angle in radians
        for idx in knee_indices:
            dof_pos[:, idx] = bend_angle
        
        # Apply slight outward rotation to hips for better balance
        hip_indices = [0, 7]  # Adjust these based on your joint order
        hip_angle = 0.1  # Small outward angle
        for idx in hip_indices:
            dof_pos[:, idx] = hip_angle
        
        # Set joint velocities
        dof_vel = torch.zeros((1, self.harpy.num_dof), device=self.device)
        
        # Set initial state
        self.harpy.set_joint_positions(dof_pos)
        self.harpy.set_joint_velocities(dof_vel)
        
        # Step simulation several times to let the robot settle
        for _ in range(10):
            self.sim.step()
        
        print("Robot reset to initial position")
    
    def run(self, duration=60.0):
        """Run the policy for the specified duration in seconds"""
        print(f"Running policy for {duration} seconds")
        
        # Reset robot
        self.reset_robot()
        
        # Run for specified duration
        sim_time = 0.0
        dt = 1.0/60.0  # Simulation time step
        
        # Tracking variables
        total_reward = 0.0
        step_count = 0
        fall_count = 0
        
        # Statistics tracking
        avg_height = 0.0
        avg_velocity = 0.0
        max_velocity = 0.0
        
        while sim_time < duration:
            # Get observations
            obs = self._get_observations()
            
            # Get action from policy (deterministic during evaluation)
            with torch.no_grad():
                action = self.policy.get_action(obs, deterministic=True)[0]
            
            # Apply action
            self.harpy.set_joint_position_targets(action)
            
            # Step simulation
            self.sim.step()
            
            # Calculate reward (same as in training)
            root_pos, root_rot = self.harpy.get_world_poses()
            root_velocities = self.harpy.get_velocities()
            
            # Track statistics
            avg_height += root_pos[:, 2].item()
            curr_velocity = root_velocities[:, 0].item()
            avg_velocity += curr_velocity
            max_velocity = max(max_velocity, curr_velocity)
            
            # Calculate reward components similar to training
            # Forward velocity
            target_velocity = 1.0  # Fixed target velocity for evaluation
            vel_error = abs(curr_velocity - target_velocity)
            forward_reward = 1.0 - np.tanh(vel_error)
            
            # Upright orientation reward
            up_vec = get_basis_vector(root_rot, 2)
            target_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(1, 1)
            up_reward = torch.sum(up_vec * target_up, dim=1).item() * 2.0
            
            # Height stability reward
            target_height = 0.5
            height_error = abs(root_pos[:, 2].item() - target_height)
            height_reward = 1.0 - np.tanh(height_error * 5.0)
            
            # Energy penalty
            joint_velocities = self.harpy.get_joint_velocities()
            energy_penalty = torch.sum(joint_velocities ** 2).item() * 0.001
            
            # Y-axis velocity and angular velocity penalties
            y_vel_penalty = abs(root_velocities[:, 1].item()) * 0.1
            ang_vel_penalty = torch.sum(torch.abs(root_velocities[:, 3:6])).item() * 0.05
            
            # Combine rewards
            reward = forward_reward + up_reward + height_reward - energy_penalty - y_vel_penalty - ang_vel_penalty
            total_reward += reward
            
            # Check if fallen
            if root_pos[:, 2].item() < 0.2:
                fall_count += 1
                print(f"Robot fell at time {sim_time:.2f}s - resetting")
                self.reset_robot()
            
            # Update time
            sim_time += dt
            step_count += 1
            
            # Print progress
            if int(sim_time) % 5 == 0 and int(sim_time / dt) % 300 == 0:
                avg_reward = total_reward / max(1, step_count)
                curr_height = root_pos[:, 2].item()
                print(f"Time: {sim_time:.1f}s | Reward: {avg_reward:.4f} | Height: {curr_height:.2f}m | Velocity: {curr_velocity:.2f}m/s")
        
        # Final statistics
        avg_reward = total_reward / max(1, step_count)
        avg_height /= max(1, step_count)
        avg_velocity /= max(1, step_count)
        
        print(f"\nTest completed:")
        print(f"Total simulation time: {sim_time:.2f}s")
        print(f"Average reward: {avg_reward:.4f}")
        print(f"Average height: {avg_height:.2f}m")
        print(f"Average velocity: {avg_velocity:.2f}m/s")
        print(f"Maximum velocity: {max_velocity:.2f}m/s")
        print(f"Total falls: {fall_count}")
        
        return {
            "avg_reward": avg_reward,
            "avg_height": avg_height,
            "avg_velocity": avg_velocity,
            "max_velocity": max_velocity,
            "falls": fall_count
        }


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test trained Harpy policy")
    parser.add_argument("--model", type=str, help="Path to model checkpoint file")
    parser.add_argument("--duration", type=float, default=60.0, help="Test duration in seconds")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    args = parser.parse_args()
    
    # Determine which model to use
    if args.model:
        model_path = args.model
    else:
        # Find the latest model
        model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith("harpy_ppo_model_") and f.endswith(".pt")]
        if not model_files:
            print(f"No model files found in {MODELS_DIR}")
            return
        
        latest_model = sorted(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        model_path = os.path.join(MODELS_DIR, latest_model)
    
    print(f"Testing model from: {model_path}")
    
    # Create and run policy test
    policy_test = HarpyPolicyTest(model_path, headless=args.headless)
    results = policy_test.run(duration=args.duration)
    
    print("\nTesting complete!")


if __name__ == "__main__":
    import sys
    main()
