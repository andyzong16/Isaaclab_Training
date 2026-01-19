# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math 
import carb

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CurriculumCfg

##
# Pre-defined configs
##

from .env_cfg import (
    G1ActionsCfg, 
    G1ObservationsCfg, 
    G1RewardsCfg, 
    G1SceneCfg,
    G1TerminationsCfg,
    G1CurriculumCfg, 
    G1EventCfg,
    G1CommandsCfg, 
)


@configclass
class G1WBCEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the motion tracking environment."""

    # Scene settings
    scene: G1SceneCfg = G1SceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: G1ObservationsCfg = G1ObservationsCfg()
    actions: G1ActionsCfg = G1ActionsCfg()
    commands: G1CommandsCfg = G1CommandsCfg()
    # MDP settings
    rewards: G1RewardsCfg = G1RewardsCfg()
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    events: G1EventCfg = G1EventCfg()
    curriculum: G1CurriculumCfg = G1CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.eye = (3.0, 3.0, 3.0)

        # # check if we are running headless
        # # if yes, we want to remove the dummy entities for performance reasons
        # carb_settings_iface = carb.settings.get_settings()
        # local_gui = carb_settings_iface.get("/app/window/enabled")
        # livestream_gui = carb_settings_iface.get("/app/livestream/enabled")
        # if not local_gui and not livestream_gui:
        #     self.scene.dummy_robot = None