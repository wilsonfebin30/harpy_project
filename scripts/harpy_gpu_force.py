#!/usr/bin/env python3

"""GPU-only Harpy training with CUDA context management fixes"""

import os
import sys
import argparse
import yaml
import torch
import time

# Critical NVIDIA environment variables for CUDA stability
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Force synchronous CUDA
os.environ["TORCH_USE_CUDA_DSA"] = "1"    # Device side assertions
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force single GPU
os.environ["PHYSX_GPU_TEMP_BUFFER_SIZE"] = "268435456"  # 256MB (4x larger)
os.environ["CUDA_CACHE_DISABLE"] = "0"    # Enable cache
os.environ["CUDA_AUTO_BOOST"] = "1"       # Enable auto boost
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "32"  # Max connections

# Parse arguments
parser = argparse.ArgumentParser(description="Train Harpy with PPO on GPU")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
parser.add_argument("--config", type=str, default=None, help="Path to config file")
args = parser.parse_args()

# Import SimulationApp first
from isaacsim.simulation_app import SimulationApp

# Initialize SimulationApp with aggressive CUDA settings
app = SimulationApp({
    "headless": args.headless,
    "physics": {
        "gpu_found_lost_pairs_capacity": 33554432,      # 32MB
        "gpu_total_aggregate_pairs_capacity": 33554432, # 32MB
        "gpu_max_rigid_contact_count": 33554432,        # 32MB
        "gpu_max_rigid_patch_count": 8388608,           # 8MB
        "enable_gpu_dynamics": True,                    # Force GPU physics
        "enable_direct_gpu_api": False,                 # CRITICAL: Disable direct GPU API
        "gpu_temp_buffer_capacity": 268435456,          # 256MB
        "gpu_max_num_partitions": 1,                    # Reduced partitions
        "gpu_dynamics_debug_verification": False,
        "gpu_dynamics_deterministic": False,
        "gpu_dynamic_allocation_scale": 2.0,
        "solver_type": 1,                               # TGS solver
        "up_axis": "z",
        "substeps": 1,                                  # Reduced complexity
        "enable_stabilization": True
    }
})

# Project paths
HOME_DIR = os.path.expanduser("~")
PROJECT_DIR = os.path.join(HOME_DIR, "harpy_project")

# Add project directory to Python path
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

# Force reset CUDA before proceeding
if torch.cuda.is_available():
    try:
        print("Resetting CUDA device...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        current_device = torch.cuda.current_device()
        torch.cuda.device(current_device).empty_cache()
        # Reset all devices
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
        # Return to main device
        torch.cuda.set_device(current_device)
        print(f"CUDA device {current_device} reset completed")
    except Exception as e:
        print(f"Warning during CUDA reset: {e}")

# Define custom tensor initialization to avoid illegal memory access
import torch.cuda
old_empty = torch.cuda.FloatTensor().new_empty

def safer_cuda_empty(*args, **kwargs):
    """Safer CUDA tensor creation with sync points"""
    try:
        tensor = old_empty(*args, **kwargs)
        torch.cuda.synchronize()
        return tensor
    except Exception as e:
        print(f"Tensor creation failed: {e}")
        # Fall back to CPU tensor if CUDA fails
        return torch.zeros(*args, device="cpu", **kwargs)

# Patch torch.cuda.FloatTensor creation
torch.cuda.FloatTensor().new_empty = safer_cuda_empty

# Function to monitor CUDA memory during execution
def monitor_cuda():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"CUDA Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")

# Load configuration
config_path = args.config if args.config else os.path.join(PROJECT_DIR, "config", "training_config.yaml")
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded configuration from {config_path}")
except Exception as e:
    print(f"Error loading configuration: {e}")
    app.close()
    sys.exit(1)

# Create a minimal, simplified articulation to test GPU physics
def test_gpu_physics():
    """Create a minimal test to verify GPU physics is working"""
    print("Testing GPU physics...")
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import UsdGeom, Gf, Sdf
    
    # Get stage
    stage = get_current_stage()
    
    # Create single box instead of complex robot
    box_path = "/World/TestBox"
    if not stage.GetPrimAtPath(box_path):
        box = UsdGeom.Cube.Define(stage, box_path)
        box.GetSizeAttr().Set(1.0)
        
        # Set initial position
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(box_path))
        if not xform.GetXformOpOrderAttr().Get():
            translate_op = xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(0.0, 0.0, 5.0))
    
    # Setup physics with minimal settings
    sim_context = SimulationContext(
        physics_dt=1.0/60.0,
        rendering_dt=1.0/60.0,
        device="cuda:0"  # Try with GPU
    )
    
    try:
        # Initialize physics context
        sim_context.initialize_physics()
        
        # Run 100 steps to see if physics works
        for i in range(100):
            sim_context.step()
            if i % 10 == 0:
                print(f"Physics test step {i}")
                monitor_cuda()
        
        print("GPU physics test successful!")
        return True
    except Exception as e:
        print(f"GPU physics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Import required modules
try:
    time.sleep(2.0)  # Delay to let GPU stabilize
    from scripts.harpy_ppo_training import HarpyEnv, PPOTrainer
    print("Successfully imported training modules")
except Exception as e:
    print(f"ERROR: Failed to import modules: {e}")
    import traceback
    traceback.print_exc()
    app.close()
    sys.exit(1)

def main():
    """Main function for training with GPU physics"""
    # Test GPU physics first
    if not test_gpu_physics():
        print("WARNING: GPU physics test failed, but we'll try to continue anyway")
    
    # Get configuration parameters
    num_envs = args.num_envs
    iterations = args.iterations
    
    print(f"Training parameters:")
    print(f"  Number of environments: {num_envs}")
    print(f"  Number of iterations: {iterations}")
    print(f"  Headless mode: {args.headless}")
    
    # Set device
    sim_device = "cuda:0"
    
    print(f"Using device: {sim_device}")
    monitor_cuda()
    
    try:
        print("\nCreating environment...")
        # Ensure config has required structure
        if 'env' not in config:
            config['env'] = {'num_envs': num_envs}
        else:
            config['env']['num_envs'] = num_envs
            
        # Force GPU physics in config
        if 'physics' not in config:
            config['physics'] = {}
        config['physics']['enable_gpu_dynamics'] = True
        config['physics']['enable_direct_gpu_api'] = False
        
        # Force more memory for GPU
        config['physics']['gpu_temp_buffer_capacity'] = 268435456
        
        # Create simplified environment for test
        # IMPORTANT MODIFICATION: Add more delay for GPU stability
        time.sleep(5.0)
        print("Creating test environment with reduced complexity...")
        
        # Modify environment creation to be more GPU-friendly
        class SimpleHarpyEnv(HarpyEnv):
            """Simplified Harpy environment with better GPU handling"""
            def _setup_scene(self):
                """Overridden scene setup with minimal complexity"""
                print("Setting up simplified scene...")
                import torch
                from isaacsim.core.utils.stage import get_current_stage
                from isaacsim.core.utils.stage import add_reference_to_stage
                from isaacsim.core.api import SimulationContext
                from pxr import UsdGeom, Gf
                
                # Force GPU cleanup before scene setup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Get stage
                stage = get_current_stage()
                
                # Create ground plane
                ground_path = "/World/groundPlane"
                if not stage.GetPrimAtPath(ground_path):
                    from scripts.harpy_ppo_training import create_ground_plane
                    create_ground_plane(stage, ground_path, size=100, color=(0.3, 0.3, 0.3))
                
                # Create simplified robot for test
                harpy_path = "/World/harpy_0"
                
                # Add reference to Harpy USD
                harpy_usd_path = os.path.join(PROJECT_DIR, "assets", "harpy", "harpy.usd")
                add_reference_to_stage(harpy_usd_path, harpy_path)
                
                # Position robot with existing transform
                xform = UsdGeom.Xformable(stage.GetPrimAtPath(harpy_path))
                
                # Check if transform already exists
                xform_ops = xform.GetOrderedXformOps()
                translate_op = None
                
                # Find existing transform
                for op in xform_ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        break
                
                # Set transform 
                if translate_op:
                    translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.5))
                else:
                    translate_op = xform.AddTranslateOp()
                    translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.5))
                
                # Store path for articulation creation
                self._harpy_paths = [harpy_path]
                
                # Initialize physics with delays
                print("Initializing physics...")
                if not hasattr(self, 'sim') or self.sim is None:
                    self.sim = SimulationContext(physics_dt=1.0/60.0, rendering_dt=1.0/60.0, device=self.device)
                
                self.sim.initialize_physics()
                
                # Step simulation with delays between steps
                print("Running settling steps...")
                for _ in range(10):
                    self.sim.step()
                    time.sleep(0.2)
                
                # Force CUDA sync before articulation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    time.sleep(1.0)
                
                # Create articulation view with super aggressive error handling
                print("Creating articulation view...")
                try:
                    from scripts.harpy_ppo_training import RobustArticulationView
                    self._harpies = RobustArticulationView(self._harpy_paths, self.device)
                    print("Articulation view created successfully")
                except Exception as e:
                    print(f"Failed to create articulation view: {e}")
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError("Failed to create articulation view")
        
        # Create simplified environment
        env = SimpleHarpyEnv(
            num_envs=1,  # Force single environment for now
            sim_device=sim_device,
            headless=args.headless,
            config=config
        )
        
        print("Environment created successfully!")
        monitor_cuda()
        
        # Create trainer
        print("\nCreating PPO trainer...")
        trainer = PPOTrainer(env=env, device=sim_device, config=config)
        
        # Run training with very minimal iterations first
        print("\nRunning minimal training test...")
        trainer.train(num_iterations=1)
        print("Minimal test completed successfully!")
        
        # Run full training if initial test was successful
        print(f"\nRunning full training for {iterations} iterations...")
        trainer.train(num_iterations=iterations)
        
        print(f"\nTraining completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        if 'env' in locals():
            try:
                env.close()
                print("Environment closed")
            except Exception as e:
                print(f"Error closing environment: {e}")
        
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("CUDA memory cleared")
            except Exception as e:
                print(f"Warning during CUDA cleanup: {e}")
        
        try:
            app.close()
            print("Application closed")
        except Exception as e:
            print(f"Error closing application: {e}")

if __name__ == "__main__":
    main()