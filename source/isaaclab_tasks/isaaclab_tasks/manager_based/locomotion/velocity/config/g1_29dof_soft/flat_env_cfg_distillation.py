# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.envs.common import ViewerCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp

from .rough_env_cfg_distillation import G1RoughStudentEnvCfg, G1RoughTeacherEnvCfg


@configclass
class G1FlatTeacherEnvCfg(G1RoughTeacherEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Optionally override physics speed
        self.sim.dt = 0.005  # 200Hz
        self.decimation = 4  # 50Hz
        self.sim.render_interval = self.decimation

        # make curriculum soft terrain
        self.scene.terrain = vel_mdp.CurriculumSoftTerrain
        # self.scene.rigid_floor = vel_mdp.CurriculumSoftTerrainPlatform

        # no height scan
        self.scene.height_scanner = None  # type: ignore
        self.observations.policy.height_scan = None  # type: ignore
        self.observations.critic.height_scan = None  # type: ignore

        # select contact solver backend
        self.actions.physics_callback.backend = "3D-warp"
        # self.actions.physics_callback.backend = "3D"
        # self.actions.physics_callback.backend = "2D"
        # self.events.randomize_stiffness.params["stiffness_range"] = (0.5, 15.0)
        # self.events.randomize_friction.params["friction_range"] = (0.2, 1.0)

        # edit randomization
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # disable curriculum for walking only
        # self.curriculum.command_vel = None

        # edit command range
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # disable for non rough terrain
        self.terminations.terrain_out_of_bounds = None

        # rendering
        self.sim.render.enable_dlssg = True
        self.sim.render.dlss_mode = "performance"  # type: ignore
        self.viewer = ViewerCfg(
            eye=(0.0, 20.0, 0.5),
            lookat=(0.0, 0.0, 0.2),
            resolution=(1080, 720),
            origin_type="asset_root",
            asset_name="robot",
        )


class G1FlatEnvCfg_PLAY(G1FlatTeacherEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # change timestep
        # self.sim.dt = 1 / 200  # 200Hz
        # self.decimation = 4  # 50Hz
        self.sim.dt = 1 / 400  # 400Hz
        self.decimation = 8  # 50Hz
        self.sim.render_interval = self.decimation
        self.episode_length_s = 15.0

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 0.0

        # terrain with hole
        self.scene.terrain = vel_mdp.RigidSoftTerrain
        self.scene.rigid_floor = vel_mdp.SoftTerrainVisual

        # make soft terrain
        # self.scene.terrain = vel_mdp.SoftTerrain
        # self.scene.rigid_floor = vel_mdp.RigidPatch
        # self.scene.terrain.disable_collider = True  # enable soft terrain
        # self.scene.terrain = vel_mdp.RoughTerrain
        # self.actions.physics_callback.disable = True # disable soft contact

        # select contact solver backend
        self.actions.physics_callback.backend = "3D-warp"
        # self.actions.physics_callback.backend = "3D"
        # self.events.randomize_stiffness.params["stiffness_range"] = (0.577, 0.577)
        # self.events.randomize_friction.params["friction_range"] = (0.577, 0.577)
        # self.events.randomize_material_density.params["packing_ratio_range"] = (1.0, 1.0)
        # self.events.randomize_material_density.params["bulk_density_range"] = (1100.0, 1100.0)
        # self.events.randomize_stiffness.params["stiffness_range"] = (0.4, 0.4)
        # self.events.randomize_friction.params["friction_range"] = (0.4, 0.4)
        # self.events.randomize_material_density.params["packing_ratio_range"] = (1.0, 1.0)
        # self.events.randomize_material_density.params["bulk_density_range"] = (3000.0, 3000.0)

        # self.actions.physics_callback.backend = "2D"
        # self.events.randomize_stiffness.params["stiffness_range"] = (1.0, 1.0)
        # self.events.randomize_friction.params["friction_range"] = (0.3, 0.3)

        # disable curriculum
        self.curriculum.terrain_levels = None  # type: ignore
        self.curriculum.command_vel = None  # type: ignore

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # remove random events
        self.events.add_base_mass = None  # type: ignore
        self.events.push_robot = None  # type: ignore
        self.events.physics_material = None  # type: ignore
        self.events.scale_actuator_gains = None  # type: ignore

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        # self.commands.base_velocity.debug_vis = False

        # Randomization
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "yaw": (-math.pi, math.pi),
                # "yaw": (-math.pi / 2, -math.pi / 2),
                # "yaw": (-math.pi/4, -math.pi/4),
                # "yaw": (0, 0),
                # "yaw": (math.pi / 2, math.pi / 2),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rendering
        self.sim.render.enable_dlssg = True
        self.sim.render.dlss_mode = "performance"  # type: ignore
        self.viewer = ViewerCfg(
            # eye=(-0.0, -3.5, 0.5),
            # lookat=(0.0, -0.0, 0.2),
            eye=(3.5, 0.0, 0.5),
            lookat=(0.0, 0.0, 0.2),
            # resolution=(1920, 1080),
            resolution=(1080, 720),
            origin_type="asset_root",
            asset_name="robot",
        )

        # # rendering
        # self.viewer = ViewerCfg(
        #     eye=(-0.0, -15.0, 1.0),
        #     lookat=(0.0, -0.0, 1.0),
        #     resolution=(1920, 1080),
        #     # origin_type="asset_root",
        #     # asset_name="robot"
        # )


"""
Student
"""


@configclass
class G1FlatStudentEnvCfg(G1RoughStudentEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Optionally override physics speed
        self.sim.dt = 0.005  # 200Hz
        self.decimation = 4  # 50Hz
        self.sim.render_interval = self.decimation

        # make curriculum soft terrain
        self.scene.terrain = vel_mdp.CurriculumSoftTerrain
        # self.scene.rigid_floor = vel_mdp.CurriculumSoftTerrainPlatform

        # no height scan
        self.scene.height_scanner = None  # type: ignore
        self.observations.policy.height_scan = None  # type: ignore
        self.observations.critic.height_scan = None  # type: ignore

        # select contact solver backend
        self.actions.physics_callback.backend = "3D-warp"
        # self.actions.physics_callback.backend = "3D"
        # self.actions.physics_callback.backend = "2D"
        # self.events.randomize_stiffness.params["stiffness_range"] = (0.5, 15.0)
        # self.events.randomize_friction.params["friction_range"] = (0.2, 1.0)

        # edit randomization
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # disable curriculum for walking only
        # self.curriculum.command_vel = None

        # edit command range
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # disable for non rough terrain
        self.terminations.terrain_out_of_bounds = None

        # rendering
        self.sim.render.enable_dlssg = True
        self.sim.render.dlss_mode = "performance"  # type: ignore
        self.viewer = ViewerCfg(
            eye=(0.0, 5.0, 0.5),
            lookat=(0.0, 0.0, 0.2),
            # resolution=(1920, 1080),
            resolution=(1080, 720),
            origin_type="asset_root",
            asset_name="robot",
        )


class G1FlatEnvStudentCfg_PLAY(G1FlatStudentEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # change timestep
        # self.sim.dt = 1 / 200  # 200Hz
        # self.decimation = 4  # 50Hz
        self.sim.dt = 1 / 400  # 400Hz
        self.decimation = 8  # 50Hz
        self.sim.render_interval = self.decimation
        self.episode_length_s = 15.0

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 0.0

        # terrain with hole
        self.scene.terrain = vel_mdp.RigidSoftTerrain
        self.scene.rigid_floor = vel_mdp.SoftTerrainVisual

        # make soft terrain
        # self.scene.terrain = vel_mdp.SoftTerrain
        # self.scene.rigid_floor = vel_mdp.RigidPatch
        # self.scene.terrain.disable_collider = True  # enable soft terrain
        # self.scene.terrain = vel_mdp.RoughTerrain
        # self.actions.physics_callback.disable = True # disable soft contact

        # select contact solver backend
        self.actions.physics_callback.backend = "3D-warp"
        # self.actions.physics_callback.backend = "3D"
        # self.events.randomize_stiffness.params["stiffness_range"] = (0.577, 0.577)
        # self.events.randomize_friction.params["friction_range"] = (0.577, 0.577)
        # self.events.randomize_material_density.params["packing_ratio_range"] = (1.0, 1.0)
        # self.events.randomize_material_density.params["bulk_density_range"] = (1100.0, 1100.0)
        # self.events.randomize_stiffness.params["stiffness_range"] = (0.4, 0.4)
        # self.events.randomize_friction.params["friction_range"] = (0.4, 0.4)
        # self.events.randomize_material_density.params["packing_ratio_range"] = (1.0, 1.0)
        # self.events.randomize_material_density.params["bulk_density_range"] = (3000.0, 3000.0)

        # self.actions.physics_callback.backend = "2D"
        # self.events.randomize_stiffness.params["stiffness_range"] = (1.0, 1.0)
        # self.events.randomize_friction.params["friction_range"] = (0.3, 0.3)

        # disable curriculum
        self.curriculum.terrain_levels = None  # type: ignore
        self.curriculum.command_vel = None  # type: ignore

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # remove random events
        self.events.add_base_mass = None  # type: ignore
        self.events.push_robot = None  # type: ignore
        self.events.physics_material = None  # type: ignore
        self.events.scale_actuator_gains = None  # type: ignore

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        # self.commands.base_velocity.debug_vis = False

        # Randomization
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "yaw": (-math.pi, math.pi),
                # "yaw": (-math.pi / 2, -math.pi / 2),
                # "yaw": (-math.pi/4, -math.pi/4),
                # "yaw": (0, 0),
                # "yaw": (math.pi / 2, math.pi / 2),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rendering
        self.sim.render.enable_dlssg = True
        self.sim.render.dlss_mode = "performance"  # type: ignore
        self.viewer = ViewerCfg(
            # eye=(-0.0, -3.5, 0.5),
            # lookat=(0.0, -0.0, 0.2),
            eye=(3.5, 0.0, 0.5),
            lookat=(0.0, 0.0, 0.2),
            # resolution=(1920, 1080),
            resolution=(1080, 720),
            origin_type="asset_root",
            asset_name="robot",
        )

        # # rendering
        # self.viewer = ViewerCfg(
        #     eye=(-0.0, -15.0, 1.0),
        #     lookat=(0.0, -0.0, 1.0),
        #     resolution=(1920, 1080),
        #     # origin_type="asset_root",
        #     # asset_name="robot"
        # )
