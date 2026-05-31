# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

##
# Pre-defined configs
##
from isaaclab_assets import (
    UNITREE_G1_29DOF_MIMIC_CFG,
)

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # uniform soft terrain (no curriculum) — suits jump trajectory training
    terrain = vel_mdp.SoftTerrain
    rigid_floor: TerrainImporterCfg = None  # type: ignore
    # robots
    robot: ArticulationCfg = UNITREE_G1_29DOF_MIMIC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore

    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
        collision_group=-1,
    )