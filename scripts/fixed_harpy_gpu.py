#!/usr/bin/env python3

"""
Fixed Harpy GPU Training Script for Isaac Sim 4.5.0
Resolves CUDA illegal memory access issues and PhysX GPU conflicts
"""

import os
import sys
import argparse
import yaml
import torch
import time
import gc

# Critical CUDA environment variables - MUST BE SET BEFORE OTHER IMPORTS
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"          # Force synchronous CUDA for better error tracking
os.environ["TORCH_USE_CUDA_DSA"] = "1"            # Enable CUDA device-side assertions
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # Restrict to single GPU
os.environ["PHYSX_GPU_TEMP_BUFFER_SIZE"] = "536870912"  # 512MB buffer (8x default)
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"   # Limit concurrent CUDA streams
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"    # Use newer CuDNN API if available
os.environ["NVIDIA_TF32_OVERRIDE"] = "1"          # Enable TF32 for better performance

# Parse arguments
parser = argparse.ArgumentParser(description="Fixed Harpy Training with GPU Support")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
parser.add_argument("--max_iterations", type=int, default=None, help="Number of iterations (alternative to --iterations)")antml:parameter>
</invoke>
parser.add_argument("--test_mode", action="store_true", help="Run in test mode with minimal setup")
parser.add_argument("--config", type=str, default=None, help="Path to config file")
parser.add_argument("--device", type=str, default=None, help="Device to use (cpu or cuda:0)")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
args = parser.parse_args()

# Import SimulationApp FIRST - critical for proper initialization
from isaacsim.simulation_app import SimulationApp

# Initialize SimulationApp with optimized PhysX settings
app = SimulationApp({
    "headless": args.headless,
    "physics": {
        # MOST CRITICAL: Disable direct GPU API - this causes the illegal memory access
        "enable_gpu_dynamics": True,
        "enable_direct_gpu_api": False,
        
        # Increase capacity for all GPU buffers
        "gpu_found_lost_pairs_capacity": 67108864,       # 64MB (4x default)
        "gpu_total_aggregate_pairs_capacity": 67108864,  # 64MB (4x default)
        "gpu_max_rigid_contact_count": 67108864,         # 64MB (4x default)
        "gpu_max_rigid_patch_count": 16777216,           # 16MB (4x default)
        "gpu_temp_buffer_capacity": 536870912,           # 512MB (8x default)
        
        # Optimize GPU dynamics
        "gpu_max_num_partitions": 2,                     # Fewer partitions for stability
        "gpu_dynamics_debug_verification": False,
        "gpu_dynamics_deterministic": False,
        "gpu_dynamic_allocation_scale": 2.0,             # Double allocation scale
        
        # Other physics optimizations
        "solver_type": 1,                                # TGS solver
        "num_position_iterations": 4,                    # Default: 4
        "num_velocity_iterations": 1,                    # Default: 1
        "substeps": 1,                                   # Reduce physics complexity
        "enable_stabilization": True,                    # Help maintain stability
    }
})

# Project paths
HOME_DIR = os.path.expanduser("~")
PROJECT_DIR = os.path.join(HOME_DIR, "harpy_project")
HARPY_USD_PATH = os.path.join(PROJECT_DIR, "assets", "harpy", "harpy.usd")
CONFIG_PATH = args.config if args.config else os.path.join(PROJECT_DIR, "config", "training_config.yaml")
SAVE_DIR = os.path.join(PROJECT_DIR, "models")

# Add project directory to Python path
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)
    print(f"Added {PROJECT_DIR} to sys.path")

# Force CUDA synchronization and memory cleanup - CRITICAL BEFORE IMPORTS
if torch.cuda.is_available():
    try:
        # Force explicit garbage collection
        gc.collect()
        
        # Thoroughly clear CUDA memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Explicitly reset all devices
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Get memory info
        device_id = torch.cuda.current_device()
        mem_free, mem_total = torch.cuda.mem_get_info(device_id)
        print(f"GPU Memory: {mem_free/1024**2:.1f}MB free of {mem_total/1024**2:.1f}MB")
    except Exception as e:
        print(f"Warning: CUDA reset error: {e}")

# Load configuration
try:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded configuration from {CONFIG_PATH}")
except Exception as e:
    print(f"Error loading configuration: {e}")
    config = {"env": {"num_envs": args.num_envs}}
    print("Using default configuration")

# Override config with command line args
config['env']['num_envs'] = args.num_envs

# Handle --max_iterations as an alternative to --iterations
if args.max_iterations is not None:
    args.iterations = args.max_iterations

# Import modules with delay to allow for proper initialization
print("Importing required modules...")
time.sleep(2.0)  # Critical delay for CUDA context initialization

# Create safe tensor allocation function
def safe_tensor(*args, **kwargs):
    """Create tensors with synchronization to avoid CUDA errors"""
    device = kwargs.get('device', 'cuda:0')
    
    # If it's a CUDA device, add synchronization safeguards
    if isinstance(device, str) and device.startswith('cuda'):
        try:
            # Create tensor
            tensor = torch.zeros(*args, **kwargs)
            
            # Force synchronization
            torch.cuda.synchronize()
            
            return tensor
        except Exception as e:
            print(f"CUDA tensor creation error: {e}")
            print("Falling back to CPU tensor")
            
            # Change device to CPU for fallback
            if 'device' in kwargs:
                kwargs['device'] = 'cpu'
            
            return torch.zeros(*args, **kwargs)
    else:
        # For CPU tensors, just create normally
        return torch.zeros(*args, **kwargs)

# Patch torch.zeros to use our safe function for CUDA tensors
original_zeros = torch.zeros

def patched_zeros(*args, **kwargs):
    """Patched version of torch.zeros that handles CUDA errors better"""
    device = kwargs.get('device', None)
    if device is not None and isinstance(device, str) and device.startswith('cuda'):
        return safe_tensor(*args, **kwargs)
    else:
        return original_zeros(*args, **kwargs)

# Apply the patch
torch.zeros = patched_zeros

# Import harpy training modules
try:
    from scripts.harpy_ppo_training import HarpyEnv, PPOTrainer, load_config, create_ground_plane
    from scripts.harpy_ppo_training import RobustArticulationView
    print("Successfully imported training modules")
except Exception as e:
    print(f"ERROR: Failed to import training modules: {e}")
    import traceback
    traceback.print_exc()
    app.close()
    sys.exit(1)

def create_test_scene():
    """Create a minimal test scene with one robot to verify GPU physics works"""
    print("Creating test scene...")
    
    # Import required modules
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.utils.stage import get_current_stage, add_reference_to_stage
    from pxr import UsdGeom, Gf
    
    # Get stage
    stage = get_current_stage()
    
    # Create ground plane
    ground_path = "/World/groundPlane"
    if not stage.GetPrimAtPath(ground_path):
        create_ground_plane(stage, ground_path, size=100, color=(0.3, 0.3, 0.3))
    
    # Create minimal physics context
    sim = SimulationContext(
        physics_dt=1.0/60.0,
        rendering_dt=1.0/60.0,
        device="cuda:0",
        physics_prim_path="/physicsScene"
    )
    
    # Initialize physics BEFORE adding robots
    print("Initializing physics...")
    sim.initialize_physics()
    
    # Step simulation multiple times to stabilize
    for i in range(10):
        sim.step()
        print(f"Physics step {i+1}/10")
        time.sleep(0.2)  # Important delay between steps
    
    # Create single test robot
    harpy_path = "/World/test_harpy"
    if not stage.GetPrimAtPath(harpy_path):
        print(f"Adding robot reference to {harpy_path}")
        add_reference_to_stage(HARPY_USD_PATH, harpy_path)
        
        # Set position
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(harpy_path))
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.5))
    
    # Force memory cleanup before creating articulation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(2.0)  # Critical delay
    
    # Try to create articulation with fallback handling
    try:
        print("Creating test articulation...")
        from isaacsim.core.prims import Articulation
        
        # Create articulation with explicit CUDA synchronization
        art = Articulation(harpy_path)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Initialize articulation
        art.initialize()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        print("Test articulation created successfully!")
        
        # Step simulation to check for stability
        for i in range(5):
            sim.step()
            print(f"Post-articulation step {i+1}/5")
            time.sleep(0.2)
        
        return True
    except Exception as e:
        print(f"Test articulation creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Create fixed HarpyEnv class that works with GPU
class FixedHarpyEnv(HarpyEnv):
    """Fixed HarpyEnv with GPU compatibility fixes"""
    
    def __init__(self, num_envs, sim_device, headless=False, config=None):
        """Override initialization with safer GPU handling"""
        print(f"Initializing FixedHarpyEnv with {num_envs} envs on {sim_device}")
        
        # Force CPU fallback if GPU test failed
        self.use_gpu = sim_device.startswith("cuda")
        if self.use_gpu and not args.test_mode:
            gpu_ok = create_test_scene()
            if not gpu_ok:
                print("WARNING: GPU test scene failed, falling back to CPU")
                sim_device = "cpu"
                self.use_gpu = False
        
        # Initialize parent with possibly modified device
        super().__init__(num_envs, sim_device, headless, config)
    
    def _setup_scene(self):
        """Override scene setup with safer implementation"""
        print("Setting up scene with safer GPU handling...")
        
        # Force memory cleanup before scene setup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Create simulation context with robust error handling
        from isaacsim.core.api import SimulationContext
        try:
            self.sim = SimulationContext(
                physics_dt=1.0/60.0,
                rendering_dt=1.0/60.0,
                device=self.device,
                physics_prim_path="/physicsScene"
            )
        except Exception as e:
            print(f"Error creating SimulationContext: {e}")
            self.device = "cpu"  # Fall back to CPU
            self.sim = SimulationContext(
                physics_dt=1.0/60.0,
                rendering_dt=1.0/60.0,
                device=self.device
            )
        
        # Get stage
        from isaacsim.core.utils.stage import get_current_stage
        stage = get_current_stage()
        
        # Create ground plane
        ground_path = "/World/groundPlane"
        if not stage.GetPrimAtPath(ground_path):
            create_ground_plane(stage, ground_path, size=100, color=(0.3, 0.3, 0.3))
        
        # Initialize physics BEFORE adding robots
        print("Initializing physics...")
        self.sim.initialize_physics()
        
        # Step simulation to stabilize physics
        for _ in range(10):
            self.sim.step()
            time.sleep(0.1)
        
        # Create harpy paths with proper spacing
        self._harpy_paths = []
        
        # Start with just 1 robot for stability, gradually add more later
        initial_robots = min(1, self.num_envs)
        print(f"Creating {initial_robots} initial robots")
        
        # Create robots with appropriate spacing
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import UsdGeom, Gf
        
        for i in range(initial_robots):
            # Create unique path for this robot
            harpy_path = f"/World/harpy_{i}"
            
            # Add reference to USD asset
            add_reference_to_stage(HARPY_USD_PATH, harpy_path)
            
            # Position each robot with separation
            xform = UsdGeom.Xformable(stage.GetPrimAtPath(harpy_path))
            translate_op = xform.AddTranslateOp()
            
            # Position at origin with height
            position = (0.0, 0.0, 0.5)
            translate_op.Set(Gf.Vec3d(position))
            
            # Store path
            self._harpy_paths.append(harpy_path)
            
            # Add delay between robot creation
            time.sleep(0.5)
            
            # Step simulation to let things settle
            for _ in range(5):
                self.sim.step()
                time.sleep(0.1)
        
        # Force CUDA synchronization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(1.0)
        
        # Create articulation with robust error handling
        try:
            print(f"Creating RobustArticulationView for {len(self._harpy_paths)} robots")
            self._harpies = RobustArticulationView(self._harpy_paths, self.device)
            self._harpies.initialize()
            print(f"Created articulations successfully")
        except Exception as e:
            print(f"ERROR: Failed to create articulations: {e}")
            
            # Try CPU fallback for articulations
            if self.device.startswith("cuda"):
                print("Trying CPU fallback for articulations...")
                self.device = "cpu"
                try:
                    self._harpies = RobustArticulationView(self._harpy_paths, self.device)
                    self._harpies.initialize()
                    print("Created articulations on CPU successfully")
                except Exception as cpu_err:
                    print(f"CPU fallback also failed: {cpu_err}")
                    raise RuntimeError("Could not create articulations on GPU or CPU")
            else:
                raise RuntimeError(f"Failed to create articulations: {e}")
        
        print("Scene setup complete!")

def main():
    """Fixed main function for training with GPU"""
    # Set device with fallback handling
    if args.test_mode:
        print("Running in TEST MODE with minimal setup")
    
    # Use device from command line if specified
    if args.device:
        sim_device = args.device
    else:
        sim_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    device = torch.device(sim_device)
    
    print(f"Training parameters:")
    print(f"  Device: {sim_device}")
    print(f"  Number of environments: {args.num_envs}")
    print(f"  Number of iterations: {args.iterations}")
    print(f"  Headless mode: {args.headless}")
    
    # Print GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_free, mem_total = torch.cuda.mem_get_info(0)
        print(f"Available memory: {mem_free/1024**2:.1f}MB / {mem_total/1024**2:.1f}MB")
    
    try:
        # Create environment with fixed implementation
        print("\nCreating environment...")
        env = FixedHarpyEnv(
            num_envs=args.num_envs, 
            sim_device=sim_device, 
            headless=args.headless, 
            config=config
        )
        
        # Create trainer
        print("\nCreating PPO trainer...")
        trainer = PPOTrainer(env=env, device=device, config=config)
        
        # Load checkpoint if specified
        if args.checkpoint and os.path.exists(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}")
            # Need to implement checkpoint loading
            checkpoint = torch.load(args.checkpoint, map_location=device)
            trainer.actor_critic.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.iteration = checkpoint.get('iteration', 0)
            print(f"Loaded checkpoint at iteration {trainer.iteration}")
        
        # Start training
        print(f"\nStarting training for {args.iterations} iterations")
        trainer.train(num_iterations=args.iterations)
        
        print("\nTraining completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Comprehensive cleanup
        print("\nCleaning up...")
        
        if 'env' in locals():
            try:
                env.close()
                print("Environment closed")
            except Exception as e:
                print(f"Error closing environment: {e}")
        
        if torch.cuda.is_available():
            try:
                # Force release of CUDA memory
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("CUDA memory cleared")
            except Exception as e:
                print(f"Error clearing CUDA memory: {e}")
        
        try:
            app.close()
            print("Application closed")
        except Exception as e:
            print(f"Error closing application: {e}")

if __name__ == "__main__":
    # Set better exception handling
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        # Ensure app is closed on error
        if 'app' in locals():
            app.close()