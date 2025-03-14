#!/usr/bin/env python3

"""
Helper script to fix PhysX settings in Isaac Sim
This can be imported into any simulation to fix RTX and PhysX conflicts
"""

import os
import torch

def fix_physics_settings(sim_context=None):
    """
    Fix physics settings for better GPU compatibility with RTX
    This can be called at any point to modify the simulation backend
    
    Args:
        sim_context: Optional SimulationContext instance. If not provided,
                     will attempt to get the current instance.
    """
    # Set environment variables
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["PHYSX_GPU_TEMP_BUFFER_SIZE"] = "268435456"  # 256MB
    os.environ["RTX_PREFER_CUDA_INTEROP"] = "0"
    os.environ["RTX_SCENE_DB_SIZE_LIMIT"] = "256"  # 256MB
    
    # Clean GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB allocated")
    
    # Get simulation context if not provided
    if sim_context is None:
        try:
            from isaacsim.core.api import SimulationContext
            sim_context = SimulationContext.instance()
        except Exception as e:
            print(f"Warning: Could not get simulation context: {e}")
            return False
    
    # Apply optimized physics settings
    try:
        from isaacsim.core.simulation_manager.sim_options import BackendOptions
        physics_options = BackendOptions()
        
        # Core physics settings
        physics_options.gpu_found_lost_pairs_capacity = 33554432      # 32MB
        physics_options.gpu_total_aggregate_pairs_capacity = 33554432 # 32MB
        physics_options.gpu_max_rigid_contact_count = 33554432        # 32MB
        physics_options.gpu_max_rigid_patch_count = 8388608           # 8MB
        
        # Critical settings
        physics_options.enable_gpu_dynamics = True
        physics_options.enable_direct_gpu_api = False  # KEY FIX
        physics_options.gpu_temp_buffer_capacity = 268435456  # 256MB
        physics_options.gpu_max_num_partitions = 2
        physics_options.gpu_heap_capacity = 536870912  # 512MB
        
        # Performance optimization
        physics_options.gpu_dynamics_debug_verification = False
        physics_options.gpu_dynamics_deterministic = False
        physics_options.gpu_dynamic_allocation_scale = 8.0
        
        # Apply the settings
        sim_context.set_backend_options(physics_options)
        print("Applied optimized physics settings")
        return True
    except Exception as e:
        print(f"Warning: Could not apply physics settings: {e}")
        return False

def optimize_renderer_settings():
    """
    Optimize renderer settings to avoid conflicts with physics simulation
    """
    try:
        import omni.kit.renderer
        renderer = omni.kit.renderer.get_renderer()
        
        # Disable ray tracing features that compete for GPU memory
        renderer.set_setting("rtxEnabled", False)
        renderer.set_setting("pathtracing.enabled", False)
        renderer.set_setting("pathtracing.enabledirt", False)
        renderer.set_setting("rtx.enableShadows", False)
        renderer.set_setting("rtx.enableRayTracedShadows", False)
        renderer.set_setting("rtx.enableRayTracedReflections", False)
        
        # Reduce memory used by renderer
        renderer.set_setting("renderer.resourcesGPUSlotsMB", 256)
        renderer.set_setting("renderer.texturesPerFrameMB", 128)
        renderer.set_setting("renderer.maxTextureResolution", 1024)
        
        print("Applied optimized renderer settings")
        return True
    except Exception as e:
        print(f"Warning: Could not optimize renderer settings: {e}")
        return False

if __name__ == "__main__":
    # This can be run directly to print diagnostic information
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix physics settings for Isaac Sim")
    parser.add_argument("--check", action="store_true", help="Check GPU memory and settings")
    args = parser.parse_args()
    
    if args.check:
        print("Checking GPU status:")
        if torch.cuda.is_available():
            print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            print(f"  Memory Reserved: {torch.cuda.memory_reserved()/1024**2:.2f}MB")
            print(f"  CUDA Version: {torch.version.cuda}")
        else:
            print("  No CUDA-capable GPU detected")
    else:
        # Apply fixes
        print("Applying physics and renderer fixes...")
        physics_fixed = fix_physics_settings()
        renderer_fixed = optimize_renderer_settings()
        
        print(f"Physics settings fixed: {physics_fixed}")
        print(f"Renderer settings fixed: {renderer_fixed}")
    
    print("Done!")