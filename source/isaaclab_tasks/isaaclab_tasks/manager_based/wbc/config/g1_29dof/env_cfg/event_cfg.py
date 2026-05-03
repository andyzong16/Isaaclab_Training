# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.wbc.mdp as wbc_mdp

@configclass
class G1EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material, # type: ignore
        mode="startup",
        params={
            # Made some changes here
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),    # (1.0, 1.0)
            "dynamic_friction_range": (0.3, 1.2),   # (1.0, 1.0)
            "restitution_range": (0.0, 0.5),        # (0.0, 0.0)
            "num_buckets": 64,
        },
    )

    # TODO: Uncomment this to enable domain randomization for joint miscalibrations
    # add_joint_default_pos = EventTerm(
    #     func=mdp.randomize_default_joint_pos,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
    #         "distribution_params": (-0.01, 0.01),
    #         "operation": "add",
    #     },
    # )

    # TODO: Uncomment this to enable domain randomization for rigid body COM
    # base_com = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
    #     },
    # )

    # TODO: Uncomment this to enable domain randomization for pushing the robot
    # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(1.0, 3.0),
    #     params={
    #         "velocity_range": {
    #             "x": (-0.5, 0.5),
    #             "y": (-0.5, 0.5),
    #             "z": (-0.2, 0.2),
    #             "roll": (-0.52, 0.52),
    #             "pitch": (-0.52, 0.52),
    #             "yaw": (-0.78, 0.78),
    #         },
    #     },
    # )