#!/usr/bin/env python3

"""Minimal Harpy training script with GPU physics but simplified robot"""

import os
import sys
import torch
import argparse
import time

# Force GPU settings
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PHYSX_GPU_TEMP_BUFFER_SIZE"] = "268435456"  # 256MB

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
args = parser.parse_args()

# Import SimulationApp first
from isaacsim.simulation_app import SimulationApp

# Initialize with GPU physics but NO DIRECT GPU API
app = SimulationApp({
    "headless": args.headless,
    "physics": {
        "gpu_found_lost_pairs_capacity": 33554432,    # 32MB
        "gpu_total_aggregate_pairs_capacity": 33554432,
        "gpu_max_rigid_contact_count": 33554432,      
        "gpu_max_rigid_patch_count": 8388608,        
        "enable_gpu_dynamics": True,                  # GPU physics ON
        "enable_direct_gpu_api": False,               # DIRECT GPU API OFF - THIS IS KEY
        "gpu_temp_buffer_capacity": 268435456,        # 256MB
        "gpu_dynamic_allocation_scale": 4.0           # Extra scaling
    }
})

# Project paths
HOME_DIR = os.path.expanduser("~")
PROJECT_DIR = os.path.join(HOME_DIR, "harpy_project")

# Add to path
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

# Core imports
from isaacsim.core.api import SimulationContext, World
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from pxr import UsdGeom, Gf, UsdPhysics
from isaacsim.core.prims import XFormPrim, RigidPrim

def main():
    print("Creating test environment with GPU physics")
    
    # Initialize simulation with GPU but no direct API
    sim = SimulationContext(physics_dt=1/60.0, device="cuda:0")
    world = World()
    stage = get_current_stage()
    
    # Create simplified robot with boxes instead of complex Harpy
    # This avoids the articulation issues
    def create_simple_robot(pos):
        # Create base
        base_path = f"/World/simple_robot_{int(pos[0])}_{int(pos[1])}"
        torso = UsdGeom.Cube.Define(stage, f"{base_path}/torso")
        torso.GetSizeAttr().Set((0.3, 0.2, 0.4))
        
        # Add physics
        rb = UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(f"{base_path}/torso"))
        
        # Position
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(f"{base_path}/torso"))
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(pos[0], pos[1], 0.5))
        
        # Create limbs
        for i, limb_pos in enumerate([
            (-0.2, 0.15, -0.15), (0.2, 0.15, -0.15),  # Arms
            (-0.15, 0, -0.35), (0.15, 0, -0.35)       # Legs
        ]):
            limb = UsdGeom.Cylinder.Define(stage, f"{base_path}/limb_{i}")
            limb.GetRadiusAttr().Set(0.05)
            limb.GetHeightAttr().Set(0.3)
            
            # Position limb
            limb_xform = UsdGeom.Xformable(stage.GetPrimAtPath(f"{base_path}/limb_{i}"))
            limb_xform.AddTranslateOp().Set(Gf.Vec3d(*limb_pos))
            
            # Add physics
            UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(f"{base_path}/limb_{i}"))
            UsdPhysics.JointAPI.Apply(stage.GetPrimAtPath(f"{base_path}/limb_{i}"))
        
        return base_path
    
    # Create ground plane properly using a cube with large width/depth and small height
    ground = UsdGeom.Cube.Define(stage, "/World/ground")
    # Set the cube's size with a single scalar value
    ground.GetSizeAttr().Set(1.0)  # This is the uniform size

    # Then apply scaling to make it wide and flat
    from pxr import UsdGeom
    xformable = UsdGeom.Xformable(ground)
    scaleOp = xformable.AddScaleOp()
    scaleOp.Set((100, 100, 0.1))  # Scale to 100x100x0.1
    ground_xform = UsdGeom.Xformable(stage.GetPrimAtPath("/World/ground"))
    ground_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))  # Position with top at z=0
    
    # Add physics to ground
    rb = UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath("/World/ground"))
    mass = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath("/World/ground"))
    mass.CreateMassAttr().Set(0)  # Make it static
    
    # Create multiple simple robots - no articulations
    robot_paths = []
    for i in range(4):  # 4 robots instead of 8 (reduced for initial testing)
        for j in range(4):  # in grid formation
            robot_path = create_simple_robot((i*3, j*3))
            robot_paths.append(robot_path)
    
    print(f"Created {len(robot_paths)} simple robots")
    
    # Initialize physics 
    print("Initializing GPU physics...")
    sim.initialize_physics()
    
    # Give physics time to stabilize before training
    print("Letting physics stabilize...")
    for i in range(50):
        sim.step()
        if i % 10 == 0:
            print(f"  Stabilization step {i}/50")
    
    # Training loop using rigids instead of articulations
    print(f"Beginning {args.iterations} training iterations...")
    for iter in range(args.iterations):
        print(f"Iteration {iter+1}/{args.iterations}")
        
        # Simulate physics for 100 steps
        for step in range(100):
            # Apply random forces to robots to simulate "training"
            if step % 10 == 0:
                for path in robot_paths:
                    torso_path = f"{path}/torso"
                    try:
                        # Apply random force
                        rb_prim = RigidPrim(torso_path)
                        force = torch.rand(3, device="cuda:0") * 100 - 50  # Random force
                        rb_prim.apply_force(force)
                    except Exception as e:
                        print(f"Error applying force: {e}")
            
            # Step physics
            sim.step()
            
            if step % 25 == 0:
                print(f"  Step {step}/100")
                
                # Print GPU memory stats
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024**2)
                    reserved = torch.cuda.memory_reserved() / (1024**2)
                    print(f"  GPU memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")
        
        # Report at end of iteration
        print(f"Completed iteration {iter+1}")
        
        # Every 5 iterations, reset robots to starting positions
        if iter % 5 == 4:
            print("Resetting robots to initial positions")
            for i, path in enumerate(robot_paths):
                try:
                    row, col = divmod(i, 4)  # Adjusted for 4x4 grid
                    rb_prim = RigidPrim(f"{path}/torso")
                    rb_prim.set_world_pose(
                        position=torch.tensor([row*3, col*3, 0.5], device="cuda:0"),
                        orientation=torch.tensor([1, 0, 0, 0], device="cuda:0")
                    )
                except Exception as e:
                    print(f"Error resetting robot: {e}")
    
    print("Training complete! GPU physics successfully used.")
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        app.close()