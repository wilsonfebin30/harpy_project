#!/usr/bin/env python3

"""
PPO Training Script for Harpy Robot 
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import yaml

import carb
import omni
from pxr import UsdGeom, Gf, UsdPhysics, Sdf
from isaacsim.core.api import World, SimulationContext
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.prims import Articulation
# Updated import for 4.5.0
from isaacsim.core.utils.torch.rotations import *

# Project paths
HOME_DIR = os.path.expanduser("~")
PROJECT_DIR = os.path.join(HOME_DIR, "harpy_project")
HARPY_USD_PATH = os.path.join(PROJECT_DIR, "assets", "harpy", "harpy.usd")
SAVE_DIR = os.path.join(PROJECT_DIR, "models")
LOG_DIR = os.path.join(PROJECT_DIR, "logs", "harpy_ppo")

# Create directories if they don't exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Default configuration values (will be overridden by config file if available)
default_config = {
    'ppo': {
        'clip_param': 0.2,
        'entropy_coef': 0.01,
        'value_loss_coef': 0.5,
        'max_grad_norm': 0.5,
        'num_mini_batches': 4,
        'num_epochs': 5,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'learning_rate': 3.0e-4
    },
    'env': {
        'num_envs': 64,
        'episode_length': 500,
        'env_spacing': 2.0
    },
    'training': {
        'iterations': 5000,
        'save_interval': 100,
        'curriculum': {
            'start_velocity': 0.3,
            'end_velocity': 1.5,
            'curriculum_length': 0.7
        }
    },
    'rewards': {
        'forward_weight': 1.0,
        'upright_weight': 2.0,
        'height_weight': 1.0,
        'energy_penalty': 0.001,
        'y_vel_penalty': 0.1,
        'ang_vel_penalty': 0.05
    }
}

# Load configuration if available, otherwise use defaults
def load_config(config_path=None, default=default_config):
    """Load configuration from file or use defaults"""
    if config_path is None:
        config_path = os.path.join(PROJECT_DIR, "config", "training_config.yaml")
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            print(f"Loaded training configuration from {config_path}")
            return config
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        print("Using default configuration.")
        return default

# Global config variable
training_config = None

class RolloutStorage:
    def __init__(self, num_envs, num_steps, obs_shape, action_shape, device):
        """Initialize rollout storage buffers"""
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device
        
        # Initialize buffers
        self.observations = torch.zeros((num_steps, num_envs, obs_shape), device=device)
        self.actions = torch.zeros((num_steps, num_envs, action_shape), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)
        self.advantages = torch.zeros((num_steps, num_envs), device=device)
        self.returns = torch.zeros((num_steps, num_envs), device=device)
        
        self.step = 0
        
    def insert(self, obs, actions, rewards, dones, values, log_probs):
        """Insert data for a single step"""
        self.observations[self.step].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.dones[self.step].copy_(dones)
        self.values[self.step].copy_(values)
        self.log_probs[self.step].copy_(log_probs)
        
        self.step = (self.step + 1) % self.num_steps
        
    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        """Compute returns and advantages using GAE"""
        last_gae_lam = 0
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = self.values[step + 1]
            
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            self.advantages[step] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        self.returns = self.advantages + self.values
        
    def get_generator(self, mini_batch_size):
        """Generate mini-batches of data for training"""
        batch_size = self.num_steps * self.num_envs
        sampler = torch.randperm(batch_size)
        
        for i in range(0, batch_size, mini_batch_size):
            indices = sampler[i:i + mini_batch_size]
            
            obs_batch = self.observations.reshape(batch_size, -1)[indices]
            actions_batch = self.actions.reshape(batch_size, -1)[indices]
            values_batch = self.values.reshape(batch_size)[indices]
            returns_batch = self.returns.reshape(batch_size)[indices]
            log_probs_batch = self.log_probs.reshape(batch_size)[indices]
            advantages_batch = self.advantages.reshape(batch_size)[indices]
            
            yield obs_batch, actions_batch, values_batch, returns_batch, log_probs_batch, advantages_batch

def create_ground_plane(stage, path, size=100, color=(0.3, 0.3, 0.3)):
    """Create a ground plane with better physics setup to avoid warnings"""
    from pxr import UsdGeom, Gf, Sdf, UsdPhysics
    
    if not stage.GetPrimAtPath(path):
        # Create the ground plane as a single prim with physics
        ground_prim = UsdGeom.Xform.Define(stage, path)
        
        # Create mesh for ground
        mesh = UsdGeom.Mesh.Define(stage, f"{path}/mesh")
        
        # Set properties
        points = [
            (-size, -size, 0),
            (size, -size, 0),
            (size, size, 0),
            (-size, size, 0)
        ]
        
        indices = [0, 1, 2, 3]
        
        # Set points and indices
        mesh.GetPointsAttr().Set(points)
        mesh.GetFaceVertexIndicesAttr().Set(indices)
        mesh.GetFaceVertexCountsAttr().Set([4])
        
        # Set display color
        display_color = UsdGeom.Primvar(mesh.GetPrim().CreateAttribute("primvars:displayColor", 
                                                                     Sdf.ValueTypeNames.Float3Array))
        display_color.Set([Gf.Vec3f(*color)])
        display_color.SetInterpolation("constant")
        
        # IMPORTANT: Apply physics only to the main ground plane, not the mesh
        # This avoids the hierarchy warning
        rb = UsdPhysics.RigidBodyAPI.Apply(ground_prim.GetPrim())
        
        # Add collision to the mesh
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        
        # Make the plane static
        mass_api = UsdPhysics.MassAPI.Apply(ground_prim.GetPrim())
        mass_api.CreateMassAttr().Set(0.0)  # Static, infinite mass
        
        print(f"Created ground plane at {path}")

# Define a wrapper for RSL RL in Isaac Sim 4.5.0
class RslRlVecEnvWrapper:
    """
    A minimal wrapper to simulate the RSL-RL library interface 
    """
    def __init__(self, base_env):
        self.base_env = base_env
        self.num_envs = base_env.num_envs
        self.device = base_env.device
        self.unwrapped = base_env
        self.headless = base_env.headless
        
    def reset(self):
        return self.base_env.reset()
    
    def step(self, actions):
        return self.base_env.step(actions)
    
    def close(self):
        return self.base_env.close()

# Define ArticulationView 
class ArticulationView:
    """
    A revised wrapper for Articulation class to work with multiple robots
    """
    def __init__(self, prim_paths_expr, name="articulation_view"):
        print(f"Creating ArticulationView with pattern: {prim_paths_expr}")
        self.prim_paths_expr = prim_paths_expr
        self.name = name
        self.world = World.instance()
        import re
        self.prim_paths = []
        
        # Use stage traversal instead of get_all_prim_paths()
        from pxr import Usd, Sdf
        from isaacsim.core.utils.stage import get_current_stage
        
        self.stage = get_current_stage()
        pattern = re.compile(prim_paths_expr.replace("*", ".*"))
        
        # Traverse stage to find matching paths
        def traverse_stage(prim):
            if prim:
                path = prim.GetPath().pathString
                if re.match(pattern, path):
                    self.prim_paths.append(path)
                
                # Traverse children
                for child in prim.GetChildren():
                    traverse_stage(child)
        
        # Start traversal from root
        traverse_stage(self.stage.GetPseudoRoot())
        
        print(f"Found {len(self.prim_paths)} matching paths: {self.prim_paths}")
        
        # Get physics simulation, but don't check for initialization
        if not hasattr(self, "sim"):
            self.sim = SimulationContext.instance()
            # Simply initialize physics directly without checking
            try:
                print("Initializing physics before creating articulations...")
                self.sim.initialize_physics()
            except Exception as e:
                print(f"Note: Physics already initialized or error: {e}")
        
        # Create articulations with proper error handling
        self.articulations = []
        for path in self.prim_paths:
            print(f"Creating articulation for path: {path}")
            try:
                # Import here to ensure it's available at runtime
                from isaacsim.core.prims import Articulation
                
                # For CUDA memory management
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear CUDA cache before each articulation
                
                # Add a small delay to let the GPU catch up
                import time
                time.sleep(0.1)
                
                # Create the articulation with careful error handling
                try:
                    # Use the safer constructor approach
                    art = Articulation(path)
                    self.articulations.append(art)
                    print(f"Successfully created articulation for {path}")
                except RuntimeError as re:
                    if "CUDA" in str(re):
                        print(f"CUDA error while creating articulation for {path}. Trying again with CPU fallback...")
                        # Try a CPU fallback approach
                        import os
                        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Force synchronous CUDA
                        art = Articulation(path)
                        self.articulations.append(art)
                        print(f"Successfully created articulation for {path} with CPU fallback")
            except Exception as e:
                print(f"ERROR creating articulation for {path}: {e}")
                import traceback
                traceback.print_exc()
        
        # Make sure we have at least one articulation
        if not self.articulations:
            raise RuntimeError(f"No articulations were created from pattern: {prim_paths_expr}")
            
        # Setup joint info based on the first articulation
        self.num_envs = len(self.articulations)
        self.num_dof = self.articulations[0].num_dof
        self.device = self.world.get_physics_context().device
        
        print(f"Created ArticulationView with {self.num_envs} robots, each with {self.num_dof} DOFs")
    
    def initialize(self):
        """Initialize all articulations with safer approach"""
        print(f"Initializing {len(self.articulations)} articulations")
        for i, art in enumerate(self.articulations):
            try:
                # Add a small delay between initializations
                import time
                time.sleep(0.05)
                
                art.initialize()
                if i % 10 == 0 or i == len(self.articulations) - 1:  # Print progress for every 10th robot
                    print(f"  Initialized articulation {i+1}/{len(self.articulations)}")
            except Exception as e:
                print(f"ERROR initializing articulation {i}: {e}")
                # Continue with the rest even if one fails
        print("Articulation initialization complete")
    
    # Fixed method to get world poses using USD transforms
    def get_world_poses(self):
        """Get world poses (position, rotation) for all articulations using USD transforms"""
        positions = []
        rotations = []
        
        from pxr import UsdGeom
        import torch
        
        for i, art in enumerate(self.articulations):
            try:
                # Get the path for this articulation
                path = self.prim_paths[i]
                
                # Use USD transforms to get position and rotation
                xform = UsdGeom.Xformable(self.stage.GetPrimAtPath(path))
                transform = xform.ComputeLocalToWorldTransform(0)
                
                # Extract position and rotation from the matrix
                pos = transform.ExtractTranslation()
                rot = transform.ExtractRotationQuat()
                
                # Convert to tensors
                pos_tensor = torch.tensor([pos[0], pos[1], pos[2]], device=self.device)
                rot_tensor = torch.tensor([rot.GetReal(), rot.GetImaginary()[0], 
                                          rot.GetImaginary()[1], rot.GetImaginary()[2]], 
                                         device=self.device)
                
                positions.append(pos_tensor)
                rotations.append(rot_tensor)
            except Exception as e:
                print(f"Error getting pose for articulation {i}: {e}")
                # Provide default values as fallback
                positions.append(torch.zeros(3, device=self.device))
                rotations.append(torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device))
        
        positions_tensor = torch.stack(positions)
        rotations_tensor = torch.stack(rotations)
        
        return positions_tensor, rotations_tensor
    
    def get_velocities(self):
        """Get velocities for all articulations"""
        vels = []
        
        for art in self.articulations:
            try:
                vel = art.get_world_velocity()
                vels.append(vel)
            except Exception as e:
                print(f"Error getting velocity: {e}")
                # Use zeros as fallback
                import torch
                vels.append(torch.zeros(6, device=self.device))
        
        import torch
        return torch.stack(vels)
    
    def get_joint_positions(self):
        """Get joint positions for all articulations"""
        positions = []
        
        for art in self.articulations:
            try:
                positions.append(art.get_joint_positions())
            except Exception as e:
                print(f"Error getting joint positions: {e}")
                # Use zeros as fallback
                import torch
                positions.append(torch.zeros((1, self.num_dof), device=self.device))
        
        import torch
        return torch.stack(positions)
    
    def get_joint_velocities(self):
        """Get joint velocities for all articulations"""
        velocities = []
        
        for art in self.articulations:
            try:
                velocities.append(art.get_joint_velocities())
            except Exception as e:
                print(f"Error getting joint velocities: {e}")
                # Use zeros as fallback
                import torch
                velocities.append(torch.zeros((1, self.num_dof), device=self.device))
        
        import torch
        return torch.stack(velocities)
    
    def set_joint_positions(self, positions):
        """Set joint positions for all articulations"""
        for i, art in enumerate(self.articulations):
            try:
                art.set_joint_positions(positions[i])
            except Exception as e:
                print(f"Error setting joint positions for articulation {i}: {e}")
    
    def set_joint_velocities(self, velocities):
        """Set joint velocities for all articulations"""
        for i, art in enumerate(self.articulations):
            try:
                art.set_joint_velocities(velocities[i])
            except Exception as e:
                print(f"Error setting joint velocities for articulation {i}: {e}")
    
    def set_joint_position_targets(self, targets):
        """Set joint position targets for all articulations"""
        for i, art in enumerate(self.articulations):
            try:
                art.set_joint_position_targets(targets[i])
            except Exception as e:
                print(f"Error setting joint position targets for articulation {i}: {e}")

# This is the key fix - define RobustArticulationView BEFORE HarpyEnv class
'''
This class fixes the main source of CUDA errors in the Harpy training by:
    1. Carefully managing GPU memory allocations
    2. Disabling direct GPU API access
    3. Adding synchronization points at critical operations
    4. Creating and initializing articulations one by one with delays
    5. Providing robust error handling and fallbacks
'''
class RobustArticulationView:
    """Truly GPU-optimized articulation view"""
    def __init__(self, prim_paths, device="cuda:0"):
        import torch
        
        # Force explicit GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self.prim_paths = prim_paths
        self.articulations = []
        self.device = device
        
        # Get stage for world transforms
        from isaacsim.core.utils.stage import get_current_stage
        self.stage = get_current_stage()
        
        # Create articulations with careful memory management
        self._create_articulations()

    # Add these lines before articulation creation:
    from isaacsim.core.api import SimulationContext
    sim = SimulationContext.instance()
    # Explicitly disable direct GPU API if using GPU
    if self.device.startswith("cuda"):
        from isaacsim.core.simulation_manager.sim_options import BackendOptions
        physics_options = BackendOptions()
        physics_options.enable_gpu_dynamics = True
        physics_options.enable_direct_gpu_api = False
        sim.set_backend_options(physics_options)
        
    def _create_articulations(self):
        """GPU-specific creation logic with better memory management"""
        import torch
        import time
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache and synchronize explicitly
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # Get initial memory stats
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated()/1024**2
            print(f"GPU memory before articulation creation: {mem_before:.2f} MB")
            
        # Import here to ensure it's available at runtime
        from isaacsim.core.prims import Articulation
        
        print(f"Creating articulations for {len(self.prim_paths)} robots on {self.device}")
        
        # IMPORTANT: Creating only ONE articulation at first to test
        # This helps isolate any issues with the articulation creation
        if len(self.prim_paths) > 0:
            try:
                # Force explicit synchronization 
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Use environment variable to debug CUDA operations
                import os
                os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
                
                # Create one articulation - KEY CHANGE: Remove device parameter
                path = self.prim_paths[0]
                print(f"Creating test articulation at {path}")
                
                # FIXED: Create articulation without device parameter
                art = Articulation(path)
                
                # Wait for GPU operations to complete
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                    
                self.articulations.append(art)
                print(f"Successfully created test articulation")
                
                # If we got here, we can safely add more articulations
                for i in range(1, len(self.prim_paths)):
                    path = self.prim_paths[i]
                    print(f"Creating articulation {i+1}/{len(self.prim_paths)} at {path}")
                    
                    # Clear cache between articulations
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Create articulation without device parameter
                    art = Articulation(path)
                    self.articulations.append(art)
                    
                    # Check memory after each articulation
                    if torch.cuda.is_available():
                        mem_current = torch.cuda.memory_allocated()/1024**2
                        print(f"GPU memory: {mem_current:.2f} MB")
                    
            except Exception as e:
                print(f"Error creating articulations: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Could not create articulations: {e}")
        
        # Validate articulations
        if not self.articulations:
            raise RuntimeError(f"No articulations could be created for paths: {self.prim_paths}")
        
        self.num_envs = len(self.articulations)
        self.num_dof = self.articulations[0].num_dof
        
        print(f"Successfully created {self.num_envs} articulations with {self.num_dof} DOFs")

    def initialize(self):
        """Initialize all articulations with extreme safety measures"""
        import torch
        import time
        print(f"Initializing {len(self.articulations)} articulations")
        
        # Force memory cleanup before initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        for i, art in enumerate(self.articulations):
            try:
                # Add very significant delay between initializations (0.5s)
                time.sleep(0.5)
                
                # Initialize one by one
                art.initialize()
                
                # Synchronize after each initialization
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                if i % 10 == 0 or i == len(self.articulations) - 1:
                    print(f"  Initialized articulation {i+1}/{len(self.articulations)}")
            except Exception as e:
                print(f"Error initializing articulation {i}: {e}")
                # Continue with the rest even if one fails
                time.sleep(1.0)  # Extra delay on error
                
        print("Articulation initialization complete")
        
        # Give GPU extra time to stabilize after initialization
        if self.device.startswith("cuda"):
            time.sleep(1.0)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_world_poses(self):
        """Get world poses (position, rotation) for all articulations"""
        import torch
        positions = []
        rotations = []
        
        for i, art in enumerate(self.articulations):
            try:
                pos, rot = art.get_world_pose()
                positions.append(pos)
                rotations.append(rot)
            except Exception as e:
                # Fallback to USD transform method if direct access fails
                from pxr import UsdGeom, Gf
                path = self.prim_paths[i]
                xform = UsdGeom.Xformable(self.stage.GetPrimAtPath(path))
                transform = xform.ComputeLocalToWorldTransform(0)
                
                # Extract position and rotation from the matrix
                pos = transform.ExtractTranslation()
                rot = transform.ExtractRotationQuat()
                
                # Convert to tensors
                pos_tensor = torch.tensor([pos[0], pos[1], pos[2]], device=self.device)
                rot_tensor = torch.tensor([rot.GetReal(), rot.GetImaginary()[0], 
                                          rot.GetImaginary()[1], rot.GetImaginary()[2]], 
                                         device=self.device)
                
                positions.append(pos_tensor)
                rotations.append(rot_tensor)
        
        positions_tensor = torch.stack(positions)
        rotations_tensor = torch.stack(rotations)
        
        return positions_tensor, rotations_tensor
    
    def get_velocities(self):
        """Get velocities for all articulations"""
        import torch
        velocities = []
        
        for art in self.articulations:
            try:
                vel = art.get_world_velocity()
                velocities.append(vel)
            except Exception as e:
                # Use zeros as fallback
                velocities.append(torch.zeros(6, device=self.device))
        
        return torch.stack(velocities)
    
    def get_joint_positions(self):
        """Get joint positions for all articulations"""
        import torch
        positions = []
        
        for art in self.articulations:
            try:
                positions.append(art.get_joint_positions())
            except Exception as e:
                # Use zeros as fallback
                positions.append(torch.zeros(self.num_dof, device=self.device))
        
        return torch.stack(positions)
    
    def get_joint_velocities(self):
        """Get joint velocities for all articulations"""
        import torch
        velocities = []
        
        for art in self.articulations:
            try:
                velocities.append(art.get_joint_velocities())
            except Exception as e:
                # Use zeros as fallback
                velocities.append(torch.zeros(self.num_dof, device=self.device))
        
        return torch.stack(velocities)
    
    def set_joint_positions(self, positions):
        """Set joint positions for all articulations"""
        for i, art in enumerate(self.articulations):
            try:
                art.set_joint_positions(positions[i])
            except Exception as e:
                print(f"Error setting joint positions for articulation {i}: {e}")
    
    def set_joint_velocities(self, velocities):
        """Set joint velocities for all articulations"""
        for i, art in enumerate(self.articulations):
            try:
                art.set_joint_velocities(velocities[i])
            except Exception as e:
                print(f"Error setting joint velocities for articulation {i}: {e}")
    
    def set_joint_position_targets(self, targets):
        """Set joint position targets for all articulations"""
        for i, art in enumerate(self.articulations):
            try:
                art.set_joint_position_targets(targets[i])
            except Exception as e:
                print(f"Error setting joint position targets for articulation {i}: {e}")

class HarpyEnv(RslRlVecEnvWrapper):
    def __init__(self, num_envs, sim_device, headless=False, config=None):
        """Initialize Harpy environment"""
        print("Initializing HarpyEnv...")
        print(f"Parameters: num_envs={num_envs}, sim_device={sim_device}, headless={headless}")
        
        # Store config
        self.config = config if config is not None else training_config
        
        # Create base environment that interfaces with the RSL RL wrapper
        print("Creating base environment...")
        base_env = self._create_base_env(num_envs, sim_device, headless)
        super().__init__(base_env)
        
        print(f"Creating SimulationContext on device {sim_device}")
        self.sim = SimulationContext(physics_dt=1.0/60.0, rendering_dt=1.0/60.0, device=sim_device)
        self.device = sim_device
        self.num_envs = num_envs
        
        # Define observation and action spaces
        self._num_observations = 47  # 3 (pos) + 4 (rot) + 3 (lin_vel) + 3 (ang_vel) + 14*2 (joints) + 3 (gravity)
        self._num_actions = 14       # Number of joints
        print(f"Observation space: {self._num_observations}, Action space: {self._num_actions}")
        
        # Get environment settings from config if available
        if self.config is not None:
            self._env_spacing = self.config['env'].get('env_spacing', 2.0)
            self._max_episode_length = self.config['env'].get('episode_length', 500)
        else:
            self._env_spacing = 2.0
            self._max_episode_length = 500
            
        print(f"Environment spacing: {self._env_spacing}, Max episode length: {self._max_episode_length}")
        self._episode_length = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Curriculum learning parameter
        self.target_velocity = 0.3  # Initial target velocity (will be increased during training)
        
        # Setup scene
        print("Setting up simulation scene...")
        self._setup_scene()
        
        # Initialize metrics for training
        self._total_reward = torch.zeros(self.num_envs, device=self.device)
        self._episode_reward = torch.zeros(self.num_envs, device=self.device)
        
        print("HarpyEnv initialized successfully.")

    def _optimize_gpu_settings(self):
        """Apply GPU-specific optimizations"""
        if not self.device.startswith("cuda"):
            return
            
        import torch
        import os
        
        # Set environment variables for better CUDA debugging
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Synchronous CUDA for better error messages
        
        # Optimize CUDA operations
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster matrix operations
        torch.backends.cudnn.benchmark = True  # Auto-tuner for best CUDA algorithms
        
        # Explicitly set memory allocation strategy
        if hasattr(torch.cuda, 'memory_stats'):  # Check if available
            torch.cuda.empty_cache()
            
            # Check available GPU memory
            mem_free, mem_total = torch.cuda.mem_get_info()
            mem_free_gb = mem_free / (1024**3)
            mem_total_gb = mem_total / (1024**3)
            
            print(f"GPU memory: {mem_free_gb:.2f} GB free out of {mem_total_gb:.2f} GB total")
            
            # Adjust caching allocator if memory is limited
            if mem_free_gb < 2.0:  # Less than 2GB free
                # Use a more conservative memory strategy
                print("Low GPU memory detected, using conservative allocation strategy")
                torch.cuda.set_per_process_memory_fraction(0.7)  # Limit to 70% of GPU memory

    def _create_base_env(self, num_envs, sim_device, headless):
        """Create a base environment that follows ManagerBasedRLEnv"""
        class BaseHarpyEnv:
            def __init__(self):
                super().__init__()
                self.num_envs = num_envs
                self.device = sim_device
                self.headless = headless
                self.unwrapped = self
                print(f"BaseHarpyEnv created: num_envs={num_envs}, device={sim_device}, headless={headless}")

            def reset(self):
                # This will be overridden by the parent class
                obs = torch.zeros((self.num_envs, 47), device=self.device)
                return obs, {}

            def step(self, actions):
                # This will be overridden by the parent class
                obs = torch.zeros((self.num_envs, 47), device=self.device)
                rewards = torch.zeros(self.num_envs, device=self.device)
                dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                return obs, rewards, dones, {}
            
        return BaseHarpyEnv()

    def _setup_scene(self):
        """Set up the environment with multiple instances of the Harpy robot on GPU"""
        # Add CUDA memory management at the start
        import torch
        import time
        import gc

        # Force garbage collection and CUDA synchronization
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"GPU Memory before scene setup: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

        print(f"Setting up scene with Harpy USD from: {HARPY_USD_PATH}")

        if not hasattr(self, 'sim') or self.sim is None:
            from isaacsim.core.api import SimulationContext
            # Configure for PhysX GPU Direct API compatibility
            # NOTE: We need to use the appropriate flags to enable Direct GPU API
            physics_config = {
                "use_gpu_pipeline": True,
                "enable_stabilization": True,
                "gpu_found_lost_pairs_capacity": 1024*1024,  # Increase capacity
                "gpu_max_rigid_contact_count": 1024*1024,
                "gpu_max_rigid_patch_count": 1024*1024, 
                "gpu_dynamic_allocation_scale": 4.0,  # Scale up memory allocation
                "gpu_heap_capacity": 256*1024*1024,  # 256 MB of GPU heap
                "enable_gpu_dynamics": True,
                "enable_direct_gpu_api": False  # CRITICAL: This fixes illegal memory access errors
            }
            self.sim = SimulationContext(physics_dt=1.0/60.0, rendering_dt=1.0/60.0, 
                                        device=self.device, physics_config=physics_config)
            
        # Verify that the USD file exists
        if not os.path.exists(HARPY_USD_PATH):
            raise FileNotFoundError(f"Could not find Harpy USD file at: {HARPY_USD_PATH}")
        
        # Get stage and create harpy paths array
        from isaacsim.core.utils.stage import get_current_stage
        stage = get_current_stage()
        self._harpy_paths = []
        
        # Start with just 1 robot for testing, gradually increase later
        test_num_envs = min(1, self.num_envs)  # Start with just 1 robot
        grid_size = 1  # Simple grid
        
        print(f"Creating {test_num_envs} robots for testing (scaled from {self.num_envs})")

        # Create a ground plane if it doesn't exist
        ground_path = "/World/groundPlane"
        if not stage.GetPrimAtPath(ground_path):
            print("Creating ground plane")
            create_ground_plane(stage, ground_path, size=100, color=(0.3, 0.3, 0.3))

        # Initialize physics before adding robots
        print("Pre-initializing physics...")
        self.sim.initialize_physics()
        
        # CRITICAL: Extra steps to stabilize the simulation
        for _ in range(5):
            self.sim.step()
            time.sleep(0.1)  # Give more time between steps

        # Create robots with much more spacing to avoid physics conflicts
        for i in range(test_num_envs):
            row = i // grid_size
            col = i % grid_size
        
            # Create unique path for each robot with proper spacing
            harpy_path = f"/World/harpy_{i}"
        
            # Add reference to USD asset with error handling
            print(f"Creating robot {i+1}/{test_num_envs} at path {harpy_path}")
            
            try:
                from isaacsim.core.utils.stage import add_reference_to_stage
                add_reference_to_stage(HARPY_USD_PATH, harpy_path)
            
                # Position robot in grid with much more spacing
                from pxr import UsdGeom, Gf
                position = (10.0, 10.0, 0.5)  # Fixed position far from origin

                # Get the robot's transform and set position
                xform = UsdGeom.Xformable(stage.GetPrimAtPath(harpy_path))
                translate_op = None
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        break

                if translate_op:
                    translate_op.Set(Gf.Vec3d(position))
                else:
                    xform.AddTranslateOp().Set(Gf.Vec3d(position))

                print(f"Positioned robot {i+1} at {position}")
                self._harpy_paths.append(harpy_path)
            
                # Add longer delay between robot creation to let GPU catch up
                time.sleep(1.0)

                # Step simulation after each robot creation
                for _ in range(3):
                    self.sim.step()
                    time.sleep(0.1)
                
            except Exception as e:
                print(f"Error creating robot {i}: {e}")

        # If no robots were created, raise an error
        if not self._harpy_paths:
            raise RuntimeError("Failed to create any robots")

        # Step simulation to let stage settle - longer duration
        print("Running settling steps...")
        for _ in range(10):  # More settling steps
            self.sim.step()
            time.sleep(0.2)  # Longer delay between steps

        # Create new articulation view
        print(f"Creating ArticulationView for {len(self._harpy_paths)} robots")
        
        # Make sure CUDA is synchronized before proceeding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # CRITICAL: Add a pause before creating articulations
        time.sleep(1.0)
        
        try:
            # Create robust articulation view - use pure GPU version
            self._harpies = RobustArticulationView(self._harpy_paths, self.device)
            
            # Initialize articulations
            self._harpies.initialize()
            
            print(f"Created robust articulation view with {len(self._harpy_paths)} robots")
        except Exception as e:
            print(f"ERROR setting up articulations: {e}")
            import traceback
            traceback.print_exc()
            
            raise RuntimeError("Failed to create articulation view")  # Stop execution if this fails

        print(f"Scene setup complete with {len(self._harpy_paths)} robots")
        
        # Final CUDA sync and memory check
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as sync_error:
                print(f"Warning: CUDA synchronization error at end of setup: {sync_error}")
                
            print(f"GPU Memory after scene setup: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    def reset(self):
        """Reset the environment and return initial observations with proper error handling"""
        print("Resetting environment...")
    
        try:
            self.sim.reset()
        except Exception as e:
            print(f"Warning: Error during sim.reset(): {e}")
        
        # Ensure _harpies exists
        if not hasattr(self, '_harpies') or self._harpies is None:
            print("WARNING: _harpies not set up properly, attempting to set up scene again")
            self._setup_scene()
        
        # Only proceed if _harpies exists now
        if hasattr(self, '_harpies') and self._harpies is not None:
            try:
                # Reset articulations to initial pose
                self._harpies.initialize()
            except Exception as e:
                print(f"Error initializing articulations during reset: {e}")
        else:
            print("ERROR: Could not set up _harpies, reset will be incomplete")
    
        # Reset robot states - set to a stable standing position
        try:
            zero_joints = torch.zeros((self.num_envs, self._harpies.num_dof), device=self.device)
        
            # Apply small knee bend to increase stability (assuming indices 2 and 9 are knees)
            # You'll need to adjust these indices based on your robot's joint configuration
            knee_indices = [2, 9]  # Adjust these based on your joint order (KneeLeft, KneeRight)
            bend_angle = 0.3  # Small bend angle in radians
        
            for idx in knee_indices:
                zero_joints[:, idx] = bend_angle
        
            # Apply slight outward rotation to hips for better balance
            hip_indices = [0, 7]  # Adjust these based on your joint order (FrontalHipLeft, FrontalHipRight)
            hip_angle = 0.1  # Small outward angle
        
            for idx in hip_indices:
                zero_joints[:, idx] = hip_angle
        
            print("Setting initial joint positions and velocities")
            try:
                self._harpies.set_joint_positions(zero_joints)
            except Exception as e:
                print(f"Error setting joint positions: {e}")
            
            try:
                self._harpies.set_joint_velocities(torch.zeros_like(zero_joints))
            except Exception as e:
                print(f"Error setting joint velocities: {e}")
        except Exception as e:
            print(f"Error setting up initial robot state: {e}")
    
        # Add a small settling period to let the robot stabilize
        print("Running settling steps...")
        for _ in range(5):
            try:
                self.sim.step()
            except Exception as e:
                print(f"Error during settling steps: {e}")
    
        # Reset episode counters
        self._episode_length = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._episode_reward = torch.zeros(self.num_envs, device=self.device)
    
        # Get initial observation with robust error handling
        try:
            print("Getting initial observations")
            obs = self._get_observations()
            print(f"Initial observation shape: {obs.shape}")
        except Exception as e:
            print(f"Error getting initial observations: {e}")
            import traceback
            traceback.print_exc()
            # Create dummy observations as a last resort
            obs = torch.zeros((self.num_envs, self._num_observations), device=self.device)
    
        print("Environment reset complete")
        return obs, {}

    def _get_observations(self):
        """Get observations from all robots with robust error handling"""
        try:
            # Get robot state - use try/except for robustness
            try:
                root_pos, root_rot = self._harpies.get_world_poses()
            except Exception as e:
                print(f"Error getting world poses: {e}")
                import traceback
                traceback.print_exc()
                # Provide default values
                root_pos = torch.zeros((self.num_envs, 3), device=self.device)
                root_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * self.num_envs, device=self.device)
            
            try:
                root_velocities = self._harpies.get_velocities()
                root_lin_vel = root_velocities[:, 0:3]
                root_ang_vel = root_velocities[:, 3:6]
            except Exception as e:
                print(f"Error getting velocities: {e}")
                # Provide default values
                root_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
                root_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
        
            # Get joint positions and velocities
            try:
                dof_pos = self._harpies.get_joint_positions()
            except Exception as e:
                print(f"Error getting joint positions: {e}")
                # Provide default values
                dof_pos = torch.zeros((self.num_envs, self._harpies.num_dof), device=self.device)
            
            try:
                dof_vel = self._harpies.get_joint_velocities()
            except Exception as e:
                print(f"Error getting joint velocities: {e}")
                # Provide default values
                dof_vel = torch.zeros((self.num_envs, self._harpies.num_dof), device=self.device)
        
            # Project gravity vector into robot frame
            try:
                # Ensure the input is a tensor with proper shape and type
                root_rot = root_rot.to(torch.float32)
                v_axis = torch.tensor([2], device=self.device, dtype=torch.long)  # Use long dtype for indices
                down_dir = get_basis_vector(root_rot, v_axis)
            except Exception as e:
                print(f"Error computing down direction: {e}")
                # Provide default values - assume down is in global -z direction
                down_dir = torch.tensor([[0.0, 0.0, -1.0]] * self.num_envs, device=self.device)
        
            # Construct observation vector
            obs = torch.cat(
                (
                    root_pos,                      # 3
                    root_rot,                      # 4 (quaternion)
                    root_lin_vel,                  # 3
                    root_ang_vel,                  # 3
                    dof_pos,                       # 14 (joints)
                    dof_vel,                       # 14 (joints)
                    down_dir,                      # 3
                ),
                dim=-1,
            )   
        
            # Ensure correct observation dimension
            if obs.shape[1] != 47:
                print(f"Warning: Observation shape is {obs.shape[1]}, padding to 47")
                padding_needed = 47 - obs.shape[1]
                padding = torch.zeros((self.num_envs, padding_needed), device=self.device)
                obs = torch.cat([obs, padding], dim=1)

            return obs
        except Exception as e:
            print(f"Error in _get_observations: {e}")
            import traceback
            traceback.print_exc()
            # Return zero observations as fallback
            return torch.zeros((self.num_envs, self._num_observations), device=self.device)

    def step(self, actions):
        """Take an action in the environment"""
        # Apply actions to robots (position targets for joints)
        self._harpies.set_joint_position_targets(actions)
        
        # Step the simulation
        self.sim.step()
        
        # Get new observations
        obs = self._get_observations()
        
        # Calculate rewards
        rewards = self._compute_rewards()
        
        # Check for termination conditions
        dones = self._check_termination()
        
        # Increment episode length
        self._episode_length += 1
        
        # Check for max episode length
        timeout = self._episode_length >= self._max_episode_length
        dones = torch.logical_or(dones, timeout)
        
        # Update episode rewards
        self._episode_reward += rewards
        
        # Prepare info for logging
        info = {}
        for i in range(self.num_envs):
            if dones[i]:
                info[f"episode_{i}/reward"] = self._episode_reward[i].item()
                info[f"episode_{i}/length"] = self._episode_length[i].item()
                
                # Reset episode stats for this env
                self._episode_length[i] = 0
                self._episode_reward[i] = 0
        
        return obs, rewards, dones, info

    def _compute_rewards(self):
        """Compute rewards for walking task with robust error handling"""
        try:
            # Get robot state
            try:
                root_pos, root_rot = self._harpies.get_world_poses()
            except Exception as e:
                print(f"Error getting poses in rewards: {e}")
                # Provide default values
                root_pos = torch.zeros((self.num_envs, 3), device=self.device)
                root_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * self.num_envs, device=self.device)
            
            try:
                root_velocities = self._harpies.get_velocities()
            except Exception as e:
                print(f"Error getting velocities in rewards: {e}")
                # Provide default values
                root_velocities = torch.zeros((self.num_envs, 6), device=self.device)
        
            # Get reward weights from config if available
            if self.config is not None:
                forward_weight = self.config['rewards'].get('forward_weight', 1.0)
                upright_weight = self.config['rewards'].get('upright_weight', 2.0)
                height_weight = self.config['rewards'].get('height_weight', 1.0)
                energy_penalty_weight = self.config['rewards'].get('energy_penalty', 0.001)
                y_vel_penalty_weight = self.config['rewards'].get('y_vel_penalty', 0.1)
                ang_vel_penalty_weight = self.config['rewards'].get('ang_vel_penalty', 0.05)
            else:
                forward_weight = 1.0
                upright_weight = 2.0
                height_weight = 1.0
                energy_penalty_weight = 0.001
                y_vel_penalty_weight = 0.1
                ang_vel_penalty_weight = 0.05
        
            # Forward velocity (x-axis) - primary reward
            forward_vel = root_velocities[:, 0]  # x-component of linear velocity
        
            # Limit maximum reward for forward velocity (prevents too aggressive movement)
            vel_error = torch.abs(forward_vel - self.target_velocity)
            forward_reward = (1.0 - torch.tanh(vel_error)) * forward_weight
        
            # Upright orientation reward - with error handling
            try:
                # Create a properly shaped tensor for the axis index with correct dtype
                root_rot = root_rot.to(torch.float32)
                v_axis = torch.tensor([2], device=self.device, dtype=torch.long)  # Use long dtype for indices
                up_vec = get_basis_vector(root_rot, v_axis)  # z-axis in robot frame
            except Exception as e:
                print(f"Error computing up vector: {e}")
                # Default to global up vector
                up_vec = torch.zeros((self.num_envs, 3), device=self.device)
                up_vec[:, 2] = 1.0  # z-axis up
            
            target_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
            up_reward = torch.sum(up_vec * target_up, dim=1) * upright_weight
        
            # Height stability reward - encourage consistent height
            target_height = 0.5  # Expected standing height
            height = root_pos[:, 2]  # z-coordinate
            height_error = torch.abs(height - target_height)
            height_reward = (1.0 - torch.tanh(height_error * 5.0)) * height_weight
        
            # Penalize falling
            fall_penalty = torch.zeros_like(forward_reward)
            fall_penalty = torch.where(height < 0.25, torch.ones_like(fall_penalty) * 2.0, fall_penalty)
        
            # Energy penalty to encourage efficient movement
            try:
                joint_velocities = self._harpies.get_joint_velocities()
                energy_penalty = torch.sum(joint_velocities ** 2, dim=1) * energy_penalty_weight
            except Exception as e:
                print(f"Error computing energy penalty: {e}")
                # Default to zero penalty
                energy_penalty = torch.zeros_like(forward_reward)
        
            # Y-axis velocity penalty - discourage sideways movement
            y_vel_penalty = torch.abs(root_velocities[:, 1]) * y_vel_penalty_weight
        
            # Angular velocity penalty - discourage spinning but allow some rotation
            ang_vel_penalty = torch.sum(torch.abs(root_velocities[:, 3:6]), dim=1) * ang_vel_penalty_weight
        
            # Combine all rewards
            reward = forward_reward + up_reward + height_reward - fall_penalty - energy_penalty - y_vel_penalty - ang_vel_penalty
        
            return reward
        except Exception as e:
            print(f"Error in _compute_rewards: {e}")
            import traceback
            traceback.print_exc()
            # Return small positive reward as fallback to avoid breaking training
            return torch.ones(self.num_envs, device=self.device) * 0.01

    def _check_termination(self):
        """Check termination conditions (e.g., robot has fallen)"""
        # Get robot state
        root_pos, _ = self._harpies.get_world_poses()
        
        # Termination if robot is too low (fallen)
        too_low = root_pos[:, 2] < 0.2
        
        return too_low

    def close(self):
        """Clean up environment resources"""
        print("Closing HarpyEnv...")
        if hasattr(self, 'sim') and self.sim is not None:
            try:
                # Stop simulation
                print("Stopping simulation...")
                self.sim.stop()
                print("Simulation stopped")
            except Exception as e:
                print(f"Error stopping simulation: {e}")
    
        # Release any other resources if needed
        print("HarpyEnv closed successfully.")

class ActorCritic(nn.Module):
    def __init__(self, num_obs, num_actions):
        super(ActorCritic, self).__init__()
        
        # Shared base network with more capacity
        self.base = nn.Sequential(
            nn.Linear(num_obs, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU()
        )
        
        # Policy head (actor)
        self.policy_mean = nn.Linear(128, num_actions)
        
        # Initialize action std as learnable parameter
        # Starting with conservative (smaller) values
        initial_std = torch.ones(1, num_actions) * 0.1
        self.policy_logstd = nn.Parameter(torch.log(initial_std))
        
        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    module.bias.data.zero_()
        
        # Special initialization for policy mean
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        self.policy_mean.bias.data.zero_()

    def forward(self, x):
        """Forward pass through network"""
        base_output = self.base(x)
        
        # Policy (mean and standard deviation)
        mean = self.policy_mean(base_output)
        logstd = self.policy_logstd.expand_as(mean)
        std = torch.exp(logstd)
        
        # Value
        value = self.value(base_output)
        
        return mean, std, value

    def get_action(self, obs, deterministic=False):
        """Sample action from current policy"""
        mean, std, value = self.forward(obs)
        
        # If deterministic, return mean
        if deterministic:
            return mean, None, value
        
        # Sample from normal distribution
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value

    def get_value(self, obs):
        """Get state value estimate"""
        base_output = self.base(obs)
        value = self.value(base_output)
        return value

class PPOTrainer:
    def __init__(self, env, device, config=None):
        print(f"Initializing PPOTrainer on device: {device}")
        self.env = env
        self.device = device
        self.config = config if config is not None else training_config
        
        # Get PPO parameters from config if available
        if self.config is not None:
            self.ppo_clip_param = self.config['ppo'].get('clip_param', 0.2)
            self.ppo_entropy_coef = self.config['ppo'].get('entropy_coef', 0.01)
            self.ppo_value_loss_coef = self.config['ppo'].get('value_loss_coef', 0.5)
            self.ppo_max_grad_norm = self.config['ppo'].get('max_grad_norm', 0.5)
            self.ppo_num_mini_batches = self.config['ppo'].get('num_mini_batches', 4)
            self.ppo_num_epochs = self.config['ppo'].get('num_epochs', 5)
            self.ppo_learning_rate = self.config['ppo'].get('learning_rate', 3e-4)
            self.ppo_gamma = self.config['ppo'].get('gamma', 0.99)
            self.ppo_gae_lambda = self.config['ppo'].get('gae_lambda', 0.95)
            
            self.episode_length = self.config['env'].get('episode_length', 500)
            self.save_interval = self.config['training'].get('save_interval', 100)
        else:
            # Use default values
            self.ppo_clip_param = 0.2
            self.ppo_entropy_coef = 0.01
            self.ppo_value_loss_coef = 0.5
            self.ppo_max_grad_norm = 0.5
            self.ppo_num_mini_batches = 4
            self.ppo_num_epochs = 5
            self.ppo_learning_rate = 3e-4
            self.ppo_gamma = 0.99
            self.ppo_gae_lambda = 0.95
            
            self.episode_length = 500
            self.save_interval = 100
            
        print(f"PPO parameters: clip={self.ppo_clip_param}, entropy={self.ppo_entropy_coef}, value_loss={self.ppo_value_loss_coef}")
        print(f"Learning rate: {self.ppo_learning_rate}, Episodes: {self.episode_length}, Save interval: {self.save_interval}")
        
        # Initialize actor-critic network
        print("Creating actor-critic network...")
        self.actor_critic = ActorCritic(env._num_observations, env._num_actions).to(device)
        
        # Initialize optimizer
        print("Creating optimizer...")
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.ppo_learning_rate)
        
        # Initialize logging
        print(f"Setting up logging to {LOG_DIR}")
        self.writer = SummaryWriter(log_dir=LOG_DIR)
        
        # Initialize rollout storage
        print(f"Creating rollout storage: envs={env.num_envs}, steps={self.episode_length}, obs={env._num_observations}, actions={env._num_actions}")
        self.rollout = RolloutStorage(
            num_envs=env.num_envs,
            num_steps=self.episode_length,
            obs_shape=env._num_observations,
            action_shape=env._num_actions,
            device=device
        )
        
        # Get initial observations
        print("Getting initial observations...")
        try:
            self.obs, _ = env.reset()
            print(f"Initial observation shape: {self.obs.shape}")
            self.rollout.observations[0].copy_(self.obs)
        except Exception as e:
            print(f"ERROR during initial observation: {e}")
            import traceback
            traceback.print_exc()
        
        # Tracking variables
        self.iteration = 0
        
        # Apply GPU optimizations if available
        if device.startswith("cuda"):
            self.optimize_for_gpu()
            
        print(f"PPOTrainer initialized on device: {device}")
        print(f"Observation space: {env._num_observations}, Action space: {env._num_actions}")
        
    def optimize_for_gpu(self):
        """Apply GPU-specific optimizations before training"""
        import torch
    
        if not self.device.startswith("cuda"):
            return  # Only apply these optimizations for GPU
    
        print("Applying GPU-specific optimizations...")
    
        # Use mixed precision where possible
        self.use_mixed_precision = True
        self.scaler = torch.cuda.amp.GradScaler()
    
        # Optimize GPU memory allocation
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
        # Use a more efficient batch size based on GPU availability
        self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
        print(f"GPU Memory: {self.gpu_memory:.2f} GB")
    
        # Adjust batch size based on available memory
        if self.gpu_memory < 8:  # Less than 8GB
            self.ppo_num_mini_batches = max(2, self.ppo_num_mini_batches // 2)
            print(f"Reduced mini-batch count to {self.ppo_num_mini_batches} for lower memory usage")
        elif self.gpu_memory > 16:  # More than 16GB
            self.ppo_num_mini_batches = min(8, self.ppo_num_mini_batches * 2)
            print(f"Increased mini-batch count to {self.ppo_num_mini_batches} for better parallelism")
    
        print("GPU optimization complete")
        
    def train(self, num_iterations):
        """Run training for specified number of iterations"""
        print(f"Starting training for {num_iterations} iterations")
        
        # Curriculum learning parameters from config
        if self.config is not None:
            start_vel = self.config['training']['curriculum'].get('start_velocity', 0.3)
            end_vel = self.config['training']['curriculum'].get('end_velocity', 1.5)
            curriculum_length = self.config['training']['curriculum'].get('curriculum_length', 0.7)
        else:
            start_vel = 0.3
            end_vel = 1.5
            curriculum_length = 0.7
            
        print(f"Curriculum learning: start_vel={start_vel}, end_vel={end_vel}, length={curriculum_length}")
        
        # Track time and add periodic save
        import time
        last_save_time = time.time()
        overall_start_time = time.time()
        
        for iteration in range(num_iterations):
            iter_start_time = time.time()
            self.iteration = iteration
            
        # Progressive curriculum (increase target velocity)
            progress = min(1.0, iteration / (num_iterations * curriculum_length))
            target_velocity = start_vel + (end_vel - start_vel) * progress
            self.env.target_velocity = target_velocity  # Pass this to environment
            
            print(f"Iteration {iteration+1}/{num_iterations} | Target velocity: {target_velocity:.2f} m/s")
            
            # Collect experience
            print(f"  Collecting rollout data for iteration {iteration+1}...")
            rollout_start = time.time()
            self.collect_rollout()
            rollout_time = time.time() - rollout_start
            print(f"  Rollout collection completed in {rollout_time:.2f}s")
            
            # Update policy
            policy_start = time.time()
            print(f"  Updating policy for iteration {iteration+1}...")
            self.update_policy()
            policy_time = time.time() - policy_start
            print(f"  Policy update completed in {policy_time:.2f}s")
            
            # Save checkpoint at regular intervals
            if (iteration + 1) % self.save_interval == 0:
                print(f"  Saving checkpoint at iteration {iteration+1}...")
                self.save_checkpoint(iteration + 1)
                
            # Add periodic save regardless of interval (every 5 minutes)
            current_time = time.time()
            if current_time - last_save_time > 300:  # 5 minutes
                print(f"  Saving emergency checkpoint at iteration {iteration+1}...")
                self.save_checkpoint(iteration + 1)
                last_save_time = current_time
                
            # Log iteration timing
            iter_time = time.time() - iter_start_time
            print(f"  Iteration {iteration+1} completed in {iter_time:.2f}s (Rollout: {rollout_time:.2f}s, Policy: {policy_time:.2f}s)")
            self.writer.add_scalar("timing/iteration_time", iter_time, iteration)
            self.writer.add_scalar("timing/rollout_time", rollout_time, iteration)
            self.writer.add_scalar("timing/policy_time", policy_time, iteration)
            
            # Log memory usage for debugging
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated(0)/1024**2
                mem_reserved = torch.cuda.memory_reserved(0)/1024**2
                print(f"  GPU Memory: {mem_allocated:.2f}MB allocated, {mem_reserved:.2f}MB reserved")
                self.writer.add_scalar("memory/gpu_allocated_mb", mem_allocated, iteration)
                self.writer.add_scalar("memory/gpu_reserved_mb", mem_reserved, iteration)
                
        total_time = time.time() - overall_start_time
        print(f"Training complete! Total time: {total_time:.2f}s for {num_iterations} iterations")
        print(f"Average time per iteration: {total_time/num_iterations:.2f}s")

    def collect_rollout(self):
        """Collect rollout data by running policy in environment"""
        print("  Beginning rollout collection...")
        
        # Debug robot state
        with torch.no_grad():
            # Check initial robot positions
            root_pos, _ = self.env._harpies.get_world_poses()
            heights = root_pos[:, 2]  # z-coordinate
            min_height = heights.min().item()
            max_height = heights.max().item()
            mean_height = heights.mean().item()
            print(f"  Robot heights - Min: {min_height:.3f}, Max: {max_height:.3f}, Mean: {mean_height:.3f}")
        
        for step in range(self.episode_length):
            if step % 100 == 0:
                print(f"    Rollout step {step}/{self.episode_length}")
                
            # Get action from policy
            with torch.no_grad():
                action_mean, action_std, value = self.actor_critic(self.obs)
                dist = Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Take action in environment
            next_obs, reward, done, info = self.env.step(action)
            
            # Store transition
            self.rollout.insert(
                self.obs,
                action,
                reward,
                done,
                value.squeeze(-1),
                log_prob
            )
            
            # Update observation
            self.obs = next_obs
            
            # Log episode stats
            for env_idx in range(self.env.num_envs):
                if f"episode_{env_idx}/reward" in info:
                    episode_reward = info[f"episode_{env_idx}/reward"]
                    episode_length = info[f"episode_{env_idx}/length"]
                    self.writer.add_scalar(f"train/episode_reward_{env_idx}", episode_reward, self.iteration)
                    self.writer.add_scalar(f"train/episode_length_{env_idx}", episode_length, self.iteration)
                    print(f"    Env {env_idx} - Episode reward: {episode_reward:.2f}, length: {episode_length}")
        
        # Compute returns and advantages
        with torch.no_grad():
            next_value = self.actor_critic(self.obs)[2].squeeze(-1)
            
        print("  Computing returns and advantages...")
        self.rollout.compute_returns_and_advantages(next_value, self.ppo_gamma, self.ppo_gae_lambda)
        print("  Rollout collection complete")

    def update_policy(self):
        """Update policy using PPO algorithm with GPU optimizations"""
        print("  Updating policy...")
        
        # Calculate batch size
        mini_batch_size = int(self.episode_length * self.env.num_envs / self.ppo_num_mini_batches)
        print(f"  Mini-batch size: {mini_batch_size}")
        
        # Track losses for logging
        policy_loss_epoch = 0
        value_loss_epoch = 0
        entropy_epoch = 0
        
        # PPO update loop with GPU optimizations
        for epoch in range(self.ppo_num_epochs):
            print(f"    PPO epoch {epoch+1}/{self.ppo_num_epochs}")
            data_generator = self.rollout.get_generator(mini_batch_size)
            
            batch_count = 0
            for sample in data_generator:
                batch_count += 1
                obs_batch, actions_batch, values_batch, returns_batch, log_probs_batch, advantages_batch = sample
                
                # Normalize advantages
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                
                # Clear gradients
                self.optimizer.zero_grad()
                
                # Use mixed precision if on GPU and enabled
                if hasattr(self, 'use_mixed_precision') and self.use_mixed_precision and self.device.startswith("cuda"):
                    with torch.cuda.amp.autocast():
                        # Forward pass
                        action_mean, action_std, values = self.actor_critic(obs_batch)
                        
                        # Get new action distribution
                        dist = torch.distributions.Normal(action_mean, action_std)
                        new_log_probs = dist.log_prob(actions_batch).sum(dim=-1)
                        entropy = dist.entropy().sum(dim=-1).mean()
                        
                        # Compute ratio for PPO
                        ratio = torch.exp(new_log_probs - log_probs_batch)
                        
                        # Compute surrogate losses
                        surr1 = ratio * advantages_batch
                        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_param, 1.0 + self.ppo_clip_param) * advantages_batch
                        
                        # Policy loss
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Value loss
                        value_loss = 0.5 * ((values.squeeze(-1) - returns_batch) ** 2).mean()
                        
                        # Total loss
                        loss = policy_loss + self.ppo_value_loss_coef * value_loss - self.ppo_entropy_coef * entropy
                    
                    # Scaled backward pass for mixed precision
                    self.scaler.scale(loss).backward()
                    
                    # Clip gradients with scaler
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.ppo_max_grad_norm)
                    
                    # Update with scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard precision training
                    # Forward pass
                    action_mean, action_std, values = self.actor_critic(obs_batch)
                    
                    # Get new action distribution
                    dist = torch.distributions.Normal(action_mean, action_std)
                    new_log_probs = dist.log_prob(actions_batch).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                    
                    # Compute ratio for PPO
                    ratio = torch.exp(new_log_probs - log_probs_batch)
                    
                    # Compute surrogate losses
                    surr1 = ratio * advantages_batch
                    surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_param, 1.0 + self.ppo_clip_param) * advantages_batch
                    
                    # Policy loss
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    value_loss = 0.5 * ((values.squeeze(-1) - returns_batch) ** 2).mean()
                    
                    # Total loss
                    loss = policy_loss + self.ppo_value_loss_coef * value_loss - self.ppo_entropy_coef * entropy
                    
                    # Standard backward pass
                    loss.backward()
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.ppo_max_grad_norm)
                    
                    # Update
                    self.optimizer.step()
                
                policy_loss_epoch += policy_loss.item()
                value_loss_epoch += value_loss.item()
                entropy_epoch += entropy.item()
                
                # Force synchronization every few batches to prevent memory buildup
                if self.device.startswith("cuda") and batch_count % 4 == 0:
                    torch.cuda.synchronize()
            
            print(f"    Processed {batch_count} mini-batches in epoch {epoch+1}")
            
            # Explicitly free memory after each epoch
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        
        # Average losses over epochs and mini-batches
        num_updates = self.ppo_num_epochs * self.ppo_num_mini_batches
        policy_loss_epoch /= num_updates
        value_loss_epoch /= num_updates
        entropy_epoch /= num_updates
        
        # Log metrics
        self.writer.add_scalar("train/policy_loss", policy_loss_epoch, self.iteration)
        self.writer.add_scalar("train/value_loss", value_loss_epoch, self.iteration)
        self.writer.add_scalar("train/entropy", entropy_epoch, self.iteration)
        
        # Log GPU memory stats
        if self.device.startswith("cuda"):
            mem_allocated = torch.cuda.memory_allocated(0)/1024**2
            mem_reserved = torch.cuda.memory_reserved(0)/1024**2
            self.writer.add_scalar("gpu/memory_allocated_mb", mem_allocated, self.iteration)
            self.writer.add_scalar("gpu/memory_reserved_mb", mem_reserved, self.iteration)
        
        print(f"  Policy update complete - Losses: Policy={policy_loss_epoch:.4f}, Value={value_loss_epoch:.4f}, Entropy={entropy_epoch:.4f}")

    def save_checkpoint(self, iteration):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(SAVE_DIR, f"harpy_ppo_model_{iteration}.pt")
        
        checkpoint = {
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iteration': iteration,
            'config': self.config  # Save config with model for reproducibility
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"  Model checkpoint saved to {checkpoint_path}")

def main(config_path=None):
    """Main function to run training"""
    global training_config
    
    # Load configuration
    training_config = load_config(config_path)
    
    # Set device
    sim_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(sim_device)
    print(f"Using device: {device}")
    
    # Get parameters from config
    num_envs = training_config['env']['num_envs']
    num_iterations = training_config['training']['iterations']
    
    try:
        # Create environment
        print("\nCreating HarpyEnv...")
        env = HarpyEnv(num_envs=num_envs, sim_device=sim_device, headless=False, config=training_config)
        
        # Create trainer
        print("\nCreating PPOTrainer...")
        trainer = PPOTrainer(env=env, device=device, config=training_config)
        
        # Run training
        print("\nStarting training...")
        trainer.train(num_iterations=num_iterations)
        
        # Close environment
        env.close()
        
        print(f"Training complete! Models saved to {SAVE_DIR}")
    except Exception as e:
        print(f"ERROR in main function: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting harpy_ppo_training.py")
    try:
        import argparse
        
        parser = argparse.ArgumentParser(description="Train Harpy robot with PPO")
        parser.add_argument("--config", type=str, default=None, help="Path to training config file")
        args = parser.parse_args()
        
        main(args.config)
    except Exception as e:
        print(f"ERROR in main script: {e}")
        import traceback
        traceback.print_exc()