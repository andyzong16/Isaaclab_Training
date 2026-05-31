# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.soft_contact.mdp as contact_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CurriculumCfg

##
# Pre-defined configs
##
from .env_cfg import (
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    SceneCfg,
    TerminationsCfg,
)


@configclass
class SoftcontactEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the motion tracking environment."""

    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 4.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # change terrain
        self.scene.terrain = contact_mdp.SoftTerrain

        # disable rigid contact
        # self.scene.terrain = vel_mdp.SoftTerrain
        self.scene.terrain.disable_collider = True  # enable soft terrain
        # self.actions.physics_callback.disable = True # disable soft contact

        # self.actions.physics_callback.backend = "2D"
        # self.events.randomize_stiffness.params["stiffness_range"] = (0.5, 15.0)
        # self.events.randomize_friction.params["friction_range"] = (0.2, 1.0)

        # viewer settings
        self.viewer = ViewerCfg(
            # eye=(-0.0, -0.7, 0.1),
            # lookat=(0.0, -0.0, 0.0),
            eye=(1.0, -2.0, 0.5),
            lookat=(1.0, -0.0, 0.2),
            resolution=(1920, 1080),
            origin_type="asset_root",
            asset_name="robot",
        )
