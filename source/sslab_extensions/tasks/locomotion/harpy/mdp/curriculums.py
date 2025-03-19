# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions for Harpy's learning environment."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than a third of the distance of the terrain
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    grid_width = 4.0
    grid_length = 4.0

    distance_x = torch.abs(asset.data.root_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0])
    distance_y = torch.abs(asset.data.root_pos_w[env_ids, 1] - env.scene.env_origins[env_ids, 1])

    move_up = torch.logical_or(distance_x >= grid_width, distance_y >= grid_length)
    move_down = torch.logical_or(distance_x <= (grid_width/2), distance_y <= (grid_length/2))
    move_down *= ~move_up

    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

def modify_joints_around_default_event(env: ManagerBasedRLEnv, env_ids: Sequence[int]):
    """Curriculum that modifies the joint initialization around default values a given number of steps."""
    if env.common_step_counter > 15000:
        # obtain term settings
        term_cfg = env.event_manager.get_term_cfg("reset_joints_around_default")
        # update term settings
        term_cfg.params['position_range'] = (-0.2, 0.2)
        term_cfg.params['velocity_range'] = (-1.0, 1.0)
        env.event_manager.set_term_cfg("reset_joints_around_default", term_cfg)
    elif env.common_step_counter > 30000:
        # obtain term settings
        term_cfg = env.event_manager.get_term_cfg("reset_joints_around_default")
        # update term settings
        term_cfg.params['position_range'] = (-0.3, 0.3)
        term_cfg.params['velocity_range'] = (-2.0, 2.0)
        env.event_manager.set_term_cfg("reset_joints_around_default", term_cfg)
    elif env.common_step_counter > 45000:
        # obtain term settings
        term_cfg = env.event_manager.get_term_cfg("reset_joints_around_default")
        # update term settings
        term_cfg.params['position_range'] = (-0.45, 0.45)
        term_cfg.params['velocity_range'] = (-2.5, 2.5)
        env.event_manager.set_term_cfg("reset_joints_around_default", term_cfg)
