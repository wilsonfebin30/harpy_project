#!/usr/bin/env python3

"""
Very basic debug script to identify exactly where the issue is occurring.
This script doesn't try to do any training, just checks if environment setup works.
"""

import os
import sys
import time

# Import SimulationApp first - this must be done before other imports
from isaacsim.simulation_app import SimulationApp

# Initialize the simulation app
print("STEP 1: Starting SimulationApp")
app = SimulationApp({"headless": True})
print("STEP 2: SimulationApp initialized")

# Project paths
HOME_DIR = os.path.expanduser("~")
PROJECT_DIR = os.path.join(HOME_DIR, "harpy_project")

# Add project directory to Python path
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)
    print(f"STEP 3: Added {PROJECT_DIR} to sys.path")

print("STEP 4: Checking if harpy USD file exists")
harpy_usd_path = os.path.join(PROJECT_DIR, "assets", "harpy", "harpy.usd")
if not os.path.exists(harpy_usd_path):
    print(f"ERROR: Harpy USD file not found at: {harpy_usd_path}")
    print("Please check if the file exists and the path is correct")
    app.close()
    sys.exit(1)
else:
    print(f"STEP 5: Found Harpy USD file at: {harpy_usd_path}")

# Import basic physics modules
try:
    print("STEP 6: Importing basic physics modules")
    from isaacsim.core.api import World, SimulationContext
    print("STEP 7: Successfully imported World and SimulationContext")
except ImportError as e:
    print(f"ERROR importing physics modules: {e}")
    app.close()
    sys.exit(1)

# Try to create a basic simulation
try:
    print("STEP 8: Creating SimulationContext")
    sim = SimulationContext(physics_dt=1.0/60.0, rendering_dt=1.0/60.0, device="cuda:0")
    print("STEP 9: SimulationContext created")

    print("STEP 10: Initializing physics")
    sim.initialize_physics()
    print("STEP 11: Physics initialized")
    
    # Run a few steps
    print("STEP 12: Running 10 simulation steps")
    for i in range(10):
        sim.step()
        print(f"    Step {i+1}/10 completed")
        time.sleep(0.1)  # Small delay to see output
    
    print("STEP 13: Simulation steps completed successfully")
except Exception as e:
    print(f"ERROR in simulation: {e}")
    import traceback
    traceback.print_exc()

# Try to import HarpyEnv
try:
    print("\nSTEP 14: Trying to import HarpyEnv")
    from harpy_ppo_training import HarpyEnv
    print("STEP 15: Successfully imported HarpyEnv")
except Exception as e:
    print(f"ERROR importing HarpyEnv: {e}")
    import traceback
    traceback.print_exc()

# Try to create a very basic environment (without any training)
try:
    print("\nSTEP 16: Creating a simple environment with 1 robot")
    env = HarpyEnv(num_envs=1, sim_device="cuda:0", headless=True, config=None)
    print("STEP 17: Environment created successfully")
    
    # Reset environment
    print("STEP 18: Resetting environment")
    obs, _ = env.reset()
    print(f"STEP 19: Environment reset completed. Observation shape: {obs.shape}")
    
    # Take a few simple steps
    print("STEP 20: Taking 5 simple steps")
    import torch
    action = torch.zeros((1, env._num_actions), device="cuda:0")
    for i in range(5):
        action.normal_()  # Random normal action
        next_obs, reward, done, info = env.step(action)
        print(f"    Step {i+1}/5 - Reward: {reward.item():.4f}")
    
    print("STEP 21: Environment steps completed successfully")
    
    # Close environment
    env.close()
    print("STEP 22: Environment closed")
except Exception as e:
    print(f"\nERROR in environment creation or interaction: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
print("\nSTEP 23: Closing SimulationApp")
app.close()
print("STEP 24: SimulationApp closed. Debug complete!")
