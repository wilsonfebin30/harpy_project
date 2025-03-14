#!/usr/bin/env python3

"""
Minimal test script for Isaac Sim 4.5.0 to isolate GPU/memory issues
"""

import os
import sys
import time

# Import SimulationApp first - this must be done before other imports
from isaacsim.simulation_app import SimulationApp

# Initialize with minimum settings and enable more memory for the GPU
print("STEP 1: Starting SimulationApp with minimal settings...")
app = SimulationApp({
    "headless": True,
    "physics": {
        "gpu_found_lost_pairs_capacity": 4194304,
        "gpu_total_aggregate_pairs_capacity": 4194304,
        "gpu_max_rigid_contact_count": 4194304,
        "gpu_max_rigid_patch_count": 1048576
    }
})
print("STEP 2: SimulationApp initialized")

# Try to import core modules
try:
    print("STEP 3: Importing minimal core modules...")
    from isaacsim.core.api import SimulationContext
    print("STEP 4: Successfully imported SimulationContext")
    
    # Create a minimal simulation
    print("STEP 5: Creating minimal SimulationContext...")
    sim = SimulationContext(physics_dt=1.0/60.0, rendering_dt=1.0/60.0, device="cpu")
    print("STEP 6: SimulationContext created")

    # Initialize physics with CPU-only
    print("STEP 7: Initializing physics on CPU...")
    sim.initialize_physics()
    print("STEP 8: Physics initialized")
    
    # Run a few steps
    print("STEP 9: Running 3 simulation steps...")
    for i in range(3):
        sim.step()
        print(f"    Step {i+1}/3 completed")
    
    print("STEP 10: Simulation steps completed successfully")
    
except Exception as e:
    print(f"ERROR during test: {e}")
    import traceback
    traceback.print_exc()

print("STEP 11: Test complete, closing application...")
try:
    app.close()
    print("STEP 12: Application closed successfully")
except Exception as e:
    print(f"Error during close: {e}")

print("Test script completed!")