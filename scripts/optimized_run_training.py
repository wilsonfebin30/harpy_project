#!/usr/bin/env python3

"""
Fixed training script for Harpy robot with robust error handling for Isaac Sim 4.5.0
"""

import os
import sys
import argparse
import yaml
import torch
import time

# Set CUDA debugging environment variables - CRITICAL for GPU stability
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force single GPU usage
os.environ["PHYSX_GPU_TEMP_BUFFER_SIZE"] = "268435456"  # 256MB buffer (quadrupled)
os.environ["RTX_PREFER_CUDA_INTEROP"] = "0"  # Disable RTX/CUDA interop
os.environ["RTX_SCENE_DB_SIZE_LIMIT"] = "256"  # Limit RTX scene DB size (MB)

# Import SimulationApp first - this must be done before other imports
from isaacsim.simulation_app import SimulationApp

# Parse arguments
parser = argparse.ArgumentParser(description="Train Harpy with PPO")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations")
parser.add_argument("--config", type=str, default=None, help="Path to training configuration file")
parser.add_argument("--device", type=str, default=None, help="Device to use (cpu or cuda:0)")
args = parser.parse_args()

app = SimulationApp({
    "headless": args.headless,
    "physics": {
        "gpu_found_lost_pairs_capacity": 33554432,    # 32MB (doubled)
        "gpu_total_aggregate_pairs_capacity": 33554432, # 32MB (doubled)
        "gpu_max_rigid_contact_count": 33554432,      # 32MB (doubled)
        "gpu_max_rigid_patch_count": 8388608,         # 8MB (doubled)
        "enable_gpu_dynamics": True,                  # USE GPU for physics
        "enable_direct_gpu_api": False,               # CRITICAL FIX: Disable direct GPU API
        "gpu_temp_buffer_capacity": 268435456,        # 256MB buffer (quadrupled)
        "gpu_max_num_partitions": 2,                  # Reduce partitions for stability
        "gpu_dynamic_allocation_scale": 8.0,          # Scale memory allocation
        "gpu_heap_capacity": 536870912                # 512MB heap (doubled)
    },
    "renderer": {
        "enable_hardware_accelerated_shadows": False,  # Disable shadows for better performance
        "enable_raytraced_shadows": False,            # Disable ray-traced shadows
        "enable_fsr": False,                          # Disable FSR for memory stability
        "enable_raytracing": False                    # Disable raytracing to preserve GPU memory
    }
})

# Project paths
HOME_DIR = os.path.expanduser("~")
PROJECT_DIR = os.path.join(HOME_DIR, "harpy_project")

# Add project directory to Python path
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)
    print(f"Added {PROJECT_DIR} to sys.path")

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

# Import required modules
try:
    print("Importing required modules...")
    
    # Check if Harpy USD file exists
    harpy_usd_path = os.path.join(PROJECT_DIR, "assets", "harpy", "harpy.usd")
    if not os.path.exists(harpy_usd_path):
        print(f"ERROR: Harpy USD file not found at {harpy_usd_path}")
        app.close()
        sys.exit(1)
    
    # Import training modules with proper error handling
    try:
        from harpy_ppo_training import HarpyEnv, PPOTrainer
        print("Successfully imported HarpyEnv and PPOTrainer")
    except ImportError as e:
        try:
            from scripts.harpy_ppo_training import HarpyEnv, PPOTrainer
            print("Successfully imported HarpyEnv and PPOTrainer from scripts directory")
        except ImportError:
            print(f"ERROR: Failed to import training modules: {e}")
            print("Make sure the harpy_ppo_training.py file exists in the project directory or Python path")
            app.close()
            sys.exit(1)
except Exception as e:
    print(f"ERROR during imports: {e}")
    import traceback
    traceback.print_exc()
    app.close()
    sys.exit(1)

def force_gpu_sync():
    """Force GPU synchronization to prevent memory issues"""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB allocated")
        except Exception as e:
            print(f"Warning: GPU sync error: {e}")

def setup_test_environment():
    """Setup and test a single environment to verify things work"""
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from pxr import UsdGeom, Gf
    
    print("Setting up test environment...")
    stage = get_current_stage()
    
    # Create ground plane if needed
    ground_path = "/World/groundPlane"
    if not stage.GetPrimAtPath(ground_path):
        try:
            from harpy_ppo_training import create_ground_plane
            create_ground_plane(stage, ground_path, size=100)
            print("Created ground plane")
        except Exception as e:
            print(f"Note: Could not create ground plane: {e}")
    
    # Create a test robot
    test_path = "/World/test_harpy"
    if not stage.GetPrimAtPath(test_path):
        try:
            add_reference_to_stage(harpy_usd_path, test_path)
            
            # Position it
            xform = UsdGeom.Xformable(stage.GetPrimAtPath(test_path))
            xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5))
            print(f"Created test harpy at {test_path}")
        except Exception as e:
            print(f"Warning: Could not create test robot: {e}")
            
    # Initialize physics
    sim = SimulationContext.instance()
    sim.initialize_physics()
    
    # Step a few times to verify stability
    for _ in range(5):
        sim.step()
        time.sleep(0.1)
    
    print("Test environment validation complete")

def main():
    # Add memory cleanup at the start of main
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    # Disable TensorFloat32 to avoid precision issues
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # Configure PyTorch for better stability
    if hasattr(torch, 'set_default_tensor_type'):
        torch.set_default_tensor_type(torch.FloatTensor)
        
    # Validate environment setup
    setup_test_environment()
    force_gpu_sync()

    # Get configuration parameters
    num_envs = args.num_envs
    iterations = args.iterations
    
    # Make sure config has proper structure
    if 'env' not in config:
        config['env'] = {}
    config['env']['num_envs'] = num_envs
    
    print(f"Training parameters:")
    print(f"  Number of environments: {num_envs}")
    print(f"  Number of iterations: {iterations}")
    print(f"  Headless mode: {args.headless}")
    
    # Set device with robust selection
    if args.device:
        sim_device = args.device
    else:
        # Prefer CUDA if available, with fallback to CPU
        sim_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    device = torch.device(sim_device)
    print(f"Using device: {device}")
    
    # Print GPU info if available
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Initial memory: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB allocated")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
    
    # Create environment with robust error handling
    try:
        print("\nCreating environment...")
        start_time = time.time()
        
        # Add extra memory management before environment creation
        force_gpu_sync()
        
        # Try with a single environment first
        test_config = config.copy()
        test_config['env']['num_envs'] = 1
        
        if num_envs > 1:
            print("Creating test environment first...")
            test_env = HarpyEnv(
                num_envs=1, 
                sim_device=sim_device, 
                headless=args.headless, 
                config=test_config
            )
            print("Test environment created successfully!")
            test_env.close()
            force_gpu_sync()
            
        # Create the actual environment with requested num_envs
        env = HarpyEnv(
            num_envs=num_envs, 
            sim_device=sim_device, 
            headless=args.headless, 
            config=config
        )
        
        env_time = time.time() - start_time
        print(f"Environment creation took {env_time:.2f} seconds")
    except Exception as e:
        print(f"ERROR creating environment: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nAttempting to create fallback environment...")
        try:
            # Try with reduced settings
            fallback_config = config.copy()
            fallback_config['env']['num_envs'] = 1
            
            env = HarpyEnv(
                num_envs=1,
                sim_device=sim_device,
                headless=args.headless,
                config=fallback_config
            )
            print("Created fallback environment with reduced settings")
        except Exception as e2:
            print(f"CRITICAL ERROR creating fallback environment: {e2}")
            app.close()
            sys.exit(1)
    
    # Create trainer with robust error handling
    try:
        print("\nCreating PPO trainer...")
        trainer = PPOTrainer(env=env, device=device, config=config)
        print("PPO trainer created successfully")
    except Exception as e:
        print(f"ERROR creating trainer: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        app.close()
        sys.exit(1)
    
    # Start training with comprehensive error handling
    print(f"\nStarting training with {num_envs} environments for {iterations} iterations")
    print("Press Ctrl+C to stop training")
    
    try:
        # Train with performance monitoring
        start_time = time.time()
        trainer.train(num_iterations=iterations)
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Average time per iteration: {total_time/iterations:.2f} seconds")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        try:
            trainer.save_checkpoint(iteration=trainer.iteration + 1)
            print("Saved checkpoint after interruption")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save checkpoint even after error
        try:
            if hasattr(trainer, 'iteration'):
                trainer.save_checkpoint(iteration=trainer.iteration + 1)
                print("Saved emergency checkpoint after error")
        except:
            pass
    finally:
        # Comprehensive cleanup
        try:
            env.close()
            print("Environment closed")
        except Exception as e:
            print(f"Error closing environment: {e}")
        
        try:
            force_gpu_sync()
        except:
            pass
            
        try:
            app.close()
            print("Application closed")
        except Exception as e:
            print(f"Error closing application: {e}")

if __name__ == "__main__":
    main()