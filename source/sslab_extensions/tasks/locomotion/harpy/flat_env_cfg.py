# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import sslab_extensions.tasks.locomotion.harpy.mdp as harpy_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from sslab_extensions.assets.harpy import HARPY_CFG  # isort: skip

FLAT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=12,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(proportion=0.2, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25),
    },
)


@configclass
class HarpyActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["Frontal.*Joint", "Sagittal.*Joint", "Knee.*Joint", "Ankle.*Joint", "Foot.*Joint"], 
        scale=0.2, 
        use_default_offset=True, 
        preserve_order=True
    )

@configclass
class HarpyCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.25,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 2.0), lin_vel_y=(-0.5, 0.5), lin_vel_z=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)
        ),
    )

@configclass
class HarpyObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class HarpyEventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Body"),
            "mass_distribution_params": (-0.1, 0.1),
            "operation": "add",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.1, 0.1),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=harpy_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.5, 0.5),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
        },
    )

@configclass
class HarpyRewardsCfg:
    # -- task
    base_linear_velocity = RewardTermCfg(
        func=harpy_mdp.base_linear_velocity_reward,
        weight=5.0,
        params={
            "std": 1.0, 
            "ramp_rate": 0.5, 
            "ramp_at_vel": 0.5, 
            "asset_cfg": SceneEntityCfg("robot")
        },
    )
    
    foot_clearance = RewardTermCfg(
        func=harpy_mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names="Foot.*Joint"),
        },
    )
    
    gait = RewardTermCfg(
        func=harpy_mdp.BipedalGaitReward,
        weight=5.0,
        params={
            "std": 0.1,
            "max_err": 0.5,
            "velocity_threshold": 0.15,
            "foot_names": ["FootLeftJoint", "FootRightJoint"],
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    # -- penalties
    action_smoothness = RewardTermCfg(func=harpy_mdp.action_smoothness_penalty, weight=-1.0)
    
    base_motion = RewardTermCfg(
        func=harpy_mdp.base_motion_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    
    base_orientation = RewardTermCfg(
        func=harpy_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    
    foot_slip = RewardTermCfg(
        func=harpy_mdp.foot_slip_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Foot.*Joint"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Foot.*Joint"),
            "threshold": 0.5,
        },
    )
    
    joint_acc = RewardTermCfg(
        func=harpy_mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*Joint")},
    )
    
    joint_pos = RewardTermCfg(
        func=harpy_mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*Joint"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.15,
        },
    )
    
    joint_torques = RewardTermCfg(
        func=harpy_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*Joint")},
    )
    
    joint_vel = RewardTermCfg(
        func=harpy_mdp.joint_velocity_penalty,
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*Joint")},
    )


@configclass
class HarpyTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", 
            body_names=["Body", "harpy_model", "Frontal.*", "Sagittal.*", "Knee.*", "Ankle.*"]), 
            "threshold": 1.0
        },
    )
    
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot")},
        time_out=True,
    )


@configclass
class HarpyCurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(func=harpy_mdp.terrain_levels_vel)


@configclass
class HarpyFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Environment configuration for Harpy robot on flat terrain."""

    # Basic settings
    observations: HarpyObservationsCfg = HarpyObservationsCfg()
    actions: HarpyActionsCfg = HarpyActionsCfg()
    commands: HarpyCommandsCfg = HarpyCommandsCfg()

    # MDP setting
    rewards: HarpyRewardsCfg = HarpyRewardsCfg()
    terminations: HarpyTerminationsCfg = HarpyTerminationsCfg()
    events: HarpyEventCfg = HarpyEventCfg()
    curriculum: HarpyCurriculumCfg = HarpyCurriculumCfg()

    # Viewer
    viewer = ViewerCfg(eye=(10.0, -15.0, 5.0), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 10  # 50 Hz
        self.episode_length_s = 20.0
        
        # simulation settings
        self.sim.dt = 0.002  # 500 Hz
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        
        # Add physics settings from your physics section in training_config.yaml
        self.sim.gpu_found_lost_pairs_capacity = 16777216
        self.sim.gpu_total_aggregate_pairs_capacity = 16777216
        self.sim.gpu_max_rigid_contact_count = 16777216
        self.sim.gpu_max_rigid_patch_count = 4194304

        # Isaac Sim specific settings
        self.sim.enable_gpu_dynamics = True
        
        # update sensor update periods
        self.scene.contact_forces.update_period = self.sim.dt

        # switch robot to Harpy
        self.scene.robot = HARPY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=FLAT_TERRAIN_CFG,
            max_init_terrain_level=FLAT_TERRAIN_CFG.num_rows - 1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=True,
        )

        # no height scan
        self.scene.height_scanner = None


class HarpyFlatEnvCfg_PLAY(HarpyFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event during play
        self.events.push_robot = None
