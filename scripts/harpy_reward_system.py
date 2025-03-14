#!/usr/bin/env python3
# Copyright (c) 2022-2025
# Adapted from Isaac Lab's reward system for bipedal robots

"""
This module implements an improved reward system for the Harpy bipedal robot,
inspired by the Cassie robot implementation from Isaac Lab.
"""

import torch
import math

class HarpyRewardSystem:
    """
    Reward system for Harpy bipedal robot, designed for flat terrain walking.
    Similar to the Cassie implementation but customized for Harpy's specific dynamics.
    """
    
    def __init__(self, device="cuda:0"):
        """Initialize reward system with default parameters."""
        self.device = device
        
        # Default reward weights
        self.forward_weight = 1.0           # Reward for moving forward at target velocity
        self.upright_weight = 2.0           # Reward for maintaining upright orientation
        self.height_weight = 1.0            # Reward for maintaining proper height
        self.energy_penalty_weight = 0.005  # Penalty for excessive joint movement/power
        self.joint_limit_penalty = 0.01     # Penalty for approaching joint limits
        self.y_vel_penalty_weight = 0.1     # Penalty for sideways movement
        self.z_vel_penalty_weight = 0.05    # Penalty for vertical movement
        self.ang_vel_penalty_weight = 0.05  # Penalty for excessive angular velocity
        self.base_stability_weight = 0.3    # Reward for stability of the base
        self.foot_contact_weight = 0.2      # Reward for proper foot contact during gait
        
        # Target parameters
        self.target_velocity = 0.5          # Target forward velocity (m/s)
        self.target_height = 0.5            # Target robot height (m)
        
        # Progress tracking
        self.reset_progress_trackers()
        
    def update_target_velocity(self, velocity):
        """Update target velocity for curriculum learning."""
        self.target_velocity = velocity
    
    def reset_progress_trackers(self):
        """Reset all progress tracking variables."""
        self.cumulative_reward = 0.0
        self.step_count = 0
        self.forward_progress = 0.0
        self.last_position = None
        
    def compute_rewards(self, root_pos, root_rot, root_velocities, joint_positions, joint_velocities, 
                        foot_contact=None, prev_root_pos=None, num_envs=1):
        """
        Compute rewards for the Harpy robot.
        
        Args:
            root_pos (torch.Tensor): Base positions [num_envs, 3]
            root_rot (torch.Tensor): Base rotations as quaternions [num_envs, 4]
            root_velocities (torch.Tensor): Base velocities [num_envs, 6] (lin_vel, ang_vel)
            joint_positions (torch.Tensor): Joint positions [num_envs, num_dofs]
            joint_velocities (torch.Tensor): Joint velocities [num_envs, num_dofs]
            foot_contact (torch.Tensor, optional): Contact state for feet [num_envs, 2]
            prev_root_pos (torch.Tensor, optional): Previous base positions for progress calculation
            num_envs (int): Number of environments
            
        Returns:
            torch.Tensor: Rewards for each environment [num_envs]
        """
        # Extract components
        root_lin_vel = root_velocities[:, 0:3]      # Linear velocity [num_envs, 3]
        root_ang_vel = root_velocities[:, 3:6]      # Angular velocity [num_envs, 3]
        
        # Device for tensor creation
        device = root_pos.device
        
        # Initialize reward tensor
        rewards = torch.zeros(num_envs, device=device)
        
        # 1. Forward velocity reward - primary objective
        forward_vel = root_lin_vel[:, 0]            # x-component of linear velocity
        vel_error = torch.abs(forward_vel - self.target_velocity)
        forward_reward = torch.exp(-vel_error * 3.0) * self.forward_weight
        rewards += forward_reward
        
        # 2. Upright orientation reward - maintain proper posture
        # Get the robot's up direction (z-axis in local coordinates)
        up_vec = self._get_up_vector(root_rot)
        target_up = torch.tensor([0.0, 0.0, 1.0], device=device).repeat(num_envs, 1)
        up_reward = (torch.sum(up_vec * target_up, dim=1)**2) * self.upright_weight
        rewards += up_reward
        
        # 3. Height stability reward - encourage consistent height
        height = root_pos[:, 2]  # z-coordinate
        height_error = torch.abs(height - self.target_height)
        height_reward = torch.exp(-height_error * 5.0) * self.height_weight
        rewards += height_reward
        
        # 4. Energy efficiency penalty
        energy_penalty = torch.sum(joint_velocities**2, dim=1) * self.energy_penalty_weight
        rewards -= energy_penalty
        
        # 5. Joint limit penalty - penalize being close to joint limits
        # This would need to be customized with actual joint limits for Harpy
        joint_limit_penalty = self._compute_joint_limit_penalty(joint_positions) * self.joint_limit_penalty
        rewards -= joint_limit_penalty
        
        # 6. Sideways movement penalty
        y_vel_penalty = torch.abs(root_lin_vel[:, 1]) * self.y_vel_penalty_weight
        rewards -= y_vel_penalty
        
        # 7. Vertical movement penalty - discourage bouncing
        z_vel_penalty = torch.abs(root_lin_vel[:, 2]) * self.z_vel_penalty_weight
        rewards -= z_vel_penalty
        
        # 8. Angular velocity penalty - discourage excessive rotation
        ang_vel_penalty = torch.sum(torch.abs(root_ang_vel), dim=1) * self.ang_vel_penalty_weight
        rewards -= ang_vel_penalty
        
        # 9. Base stability reward - encourage smooth base movement
        # Calculated based on angular velocity of the base
        base_stability = torch.exp(-torch.norm(root_ang_vel, dim=1) * 2.0) * self.base_stability_weight
        rewards += base_stability
        
        # 10. Foot contact reward - if foot contact information is available
        if foot_contact is not None:
            foot_contact_reward = self._compute_foot_contact_reward(foot_contact, forward_vel) * self.foot_contact_weight
            rewards += foot_contact_reward
        
        # Fall penalty - strongly discourage falling
        fall_penalty = torch.where(height < 0.2, torch.ones_like(rewards) * 2.0, torch.zeros_like(rewards))
        rewards -= fall_penalty
        
        # Track forward progress if previous positions are provided
        if prev_root_pos is not None:
            progress = root_pos[:, 0] - prev_root_pos[:, 0]  # x-direction progress
            self.forward_progress += progress.mean().item()
            self.last_position = root_pos.clone()
        elif self.last_position is None:
            self.last_position = root_pos.clone()
            
        # Update tracking metrics
        self.cumulative_reward += rewards.mean().item()
        self.step_count += 1
        
        return rewards
    
    def _get_up_vector(self, root_rot):
        """
        Get the up vector (z-axis) in world coordinates from the rotation quaternion.
        
        Args:
            root_rot (torch.Tensor): Base rotation quaternions [num_envs, 4]
            
        Returns:
            torch.Tensor: Up vectors [num_envs, 3]
        """
        # This assumes that the 3rd basis vector (index 2) corresponds to the up direction
        # Use specified rotation functions if they're available in your environment
        try:
            from isaacsim.core.api.utils.torch.rotations import get_basis_vector
            return get_basis_vector(root_rot, torch.tensor([2], device=root_rot.device, dtype=torch.long))
        except ImportError:
            # Fallback implementation using quaternion to rotation matrix conversion
            qw, qx, qy, qz = root_rot[:, 0], root_rot[:, 1], root_rot[:, 2], root_rot[:, 3]
            
            # Third column of rotation matrix gives the z-axis
            x = 2 * (qx * qz + qw * qy)
            y = 2 * (qy * qz - qw * qx)
            z = 1 - 2 * (qx * qx + qy * qy)
            
            return torch.stack([x, y, z], dim=1)
    
    def _compute_joint_limit_penalty(self, joint_positions):
        """
        Compute penalty for joints approaching their limits.
        This is a simplified implementation - you should customize this with Harpy's actual joint limits.
        
        Args:
            joint_positions (torch.Tensor): Joint positions [num_envs, num_dofs]
            
        Returns:
            torch.Tensor: Joint limit penalties [num_envs]
        """
        # Simplified joint limits - you should replace these with actual limits
        joint_ranges = torch.tensor([
            [-1.0, 1.0],  # Joint 0 range
            [-1.0, 1.0],  # Joint 1 range
            # ... add ranges for all joints
        ], device=joint_positions.device)
        
        # If ranges not defined for all joints, use a default range
        if joint_ranges.shape[0] < joint_positions.shape[1]:
            default_range = [-1.0, 1.0]
            num_missing = joint_positions.shape[1] - joint_ranges.shape[0]
            default_ranges = torch.tensor([default_range] * num_missing, device=joint_positions.device)
            joint_ranges = torch.cat([joint_ranges, default_ranges], dim=0)
        
        # Calculate how close each joint is to its limit
        lower_limits = joint_ranges[:, 0].unsqueeze(0)  # [1, num_dofs]
        upper_limits = joint_ranges[:, 1].unsqueeze(0)  # [1, num_dofs]
        
        # Normalized position in range [0, 1]
        normalized_pos = (joint_positions - lower_limits) / (upper_limits - lower_limits)
        
        # Penalty increases as joints get closer to limits (0 or 1)
        margin = 0.1
        lower_penalty = torch.maximum(torch.zeros_like(normalized_pos), 
                                     margin - normalized_pos) / margin
        upper_penalty = torch.maximum(torch.zeros_like(normalized_pos), 
                                     normalized_pos + margin - 1.0) / margin
        
        # Combine penalties
        joint_penalties = lower_penalty + upper_penalty
        
        # Sum across all joints
        return torch.sum(joint_penalties, dim=1)
    
    def _compute_foot_contact_reward(self, foot_contact, forward_vel):
        """
        Compute reward for proper foot contact patterns during walking.
        
        Args:
            foot_contact (torch.Tensor): Contact state for feet [num_envs, 2]
            forward_vel (torch.Tensor): Forward velocity [num_envs]
            
        Returns:
            torch.Tensor: Foot contact reward [num_envs]
        """
        # Encourage alternating foot contacts when moving
        num_envs = foot_contact.shape[0]
        device = foot_contact.device
        
        # Calculate if robot is moving
        is_moving = torch.abs(forward_vel) > 0.1
        
        # Basic reward: at least one foot should be in contact
        any_contact = torch.sum(foot_contact, dim=1) > 0
        
        # Advanced reward: feet should alternate contact when moving
        # If robot is moving, we want an alternating gait pattern
        left_foot = foot_contact[:, 0]
        right_foot = foot_contact[:, 1]
        
        # Ideal pattern: exactly one foot in contact at a time when moving
        one_foot_contact = ((left_foot + right_foot) == 1).float()
        
        # Combine rewards
        contact_reward = torch.where(
            is_moving,
            one_foot_contact,  # When moving: reward alternating contacts
            any_contact.float()  # When standing: reward any contact
        )
        
        return contact_reward
        
    def get_metrics(self):
        """Get training metrics for logging."""
        if self.step_count == 0:
            avg_reward = 0.0
        else:
            avg_reward = self.cumulative_reward / self.step_count
            
        metrics = {
            'avg_reward': avg_reward,
            'forward_progress': self.forward_progress,
            'step_count': self.step_count
        }
        
        return metrics