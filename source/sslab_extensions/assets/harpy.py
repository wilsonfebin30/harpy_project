# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
import isaaclab.actuators as acts
from isaaclab.assets.articulation import ArticulationCfg

# Define the path to the Harpy USD file
HARPY_USD_PATH = "/home/febin/harpy_project/assets/harpy/harpy.usd"

##
# Configuration
##

HARPY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/febin/harpy_project/assets/harpy/harpy.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),  # Starting higher for bipedal robot
        joint_pos={
            "FrontalLeftJoint": 0.0,
            "FrontalRightJoint": 0.0,
            "SagittalLeftJoint": 0.0,
            "SagittalRightJoint": 0.0,
            "KneeLeftJoint": 0.3,
            "KneeRightJoint": 0.3,
            "AnkleLeftJoint": 0.0,
            "AnkleRightJoint": 0.0,
            "FootLeftJoint": 0.0,
            "FootRightJoint": 0.0,
            "ThrusterLeftJoint": 0.0,
            "ThrusterRightJoint": 0.0,
        },
        joint_vel={".*Joint": 0.0},
        joint_pos_flight={".*Joint": 0.0},
        joint_vel_flight={".*Joint": 0.0},
    ),
    actuators={
        "hip_frontal": acts.DelayedPDActuatorCfg(
            joint_names_expr=["Frontal.*Joint"],
            effort_limit=350.0,
            velocity_limit=10.0,
            velocity_limit_sim=10.0,
            stiffness=800.0,
            damping=40.0,
            min_delay=0,
            max_delay=4
        ),
        "hip_sagittal": acts.DelayedPDActuatorCfg(
            joint_names_expr=["Sagittal.*Joint"],
            effort_limit=350.0,
            velocity_limit=10.0,
            velocity_limit_sim=10.0,
            stiffness=800.0,
            damping=40.0,
            min_delay=0,
            max_delay=4
        ),
        "knees": acts.DelayedPDActuatorCfg(
            joint_names_expr=["Knee.*Joint"],
            effort_limit=350.0,
            velocity_limit=10.0,
            velocity_limit_sim=10.0,
            stiffness=800.0,
            damping=40.0,
            min_delay=0,
            max_delay=4
        ),
        "ankles_feet": acts.DelayedPDActuatorCfg(
            joint_names_expr=["Ankle.*Joint", "Foot.*Joint"],
            effort_limit=350.0,
            velocity_limit=10.0,
            velocity_limit_sim=10.0,
            stiffness=800.0,
            damping=40.0,
            min_delay=0,
            max_delay=4
        ),
    },
)
