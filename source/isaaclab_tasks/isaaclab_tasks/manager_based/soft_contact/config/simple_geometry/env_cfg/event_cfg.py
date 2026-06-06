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
import isaaclab_tasks.manager_based.soft_contact.mdp as contact_mdp

@configclass
class EventCfg:
    """Configuration for events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            # "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*"]),
            "static_friction_range": (0.1, 0.1),
            "dynamic_friction_range": (0.1, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    reset_base = EventTerm(
        func=contact_mdp.reset_root_state_uniform, 
        mode="reset",
        params={
            # "pose_range": {"x": (-10.0, 10.0), "y": (-10.0, 10.0), "z": (0.1, 0.1), "yaw": (0.0, 0.0)},
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (0.1, 0.1), "yaw": (0.0, 0.0)},
            "velocity_range": {
                # "x": (1.0, 1.0),
                "x": (0.0, 0.0),
                "y": (1.0, 1.0),
                # "y": (0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # randomize terrain friction
    randomize_friction = EventTerm(
        func=contact_mdp.randomize_terrain_friction,
        mode="reset",
        params={
            # "friction_range": (0.1, 1.0),
            "friction_range": (0.5, 0.5),
            "contact_solver_name": "physics_callback",
        },
    )

    # randomize terrain stiffness
    randomize_stiffness = EventTerm(
        func=contact_mdp.randomize_terrain_stiffness,
        mode="reset",
        params={
            # "stiffness_range": (0.3, 0.9),
            # "stiffness_range": (math.tan(math.radians(20)), math.tan(math.radians(40))),
            "stiffness_range": (math.tan(math.radians(40)), math.tan(math.radians(40))),
            # "stiffness_range": (math.tan(math.radians(30)), math.tan(math.radians(30))),
            # "stiffness_range": (math.tan(math.radians(30)), math.tan(math.radians(30))),
            "contact_solver_name": "physics_callback",
        },
    )
    # randomize material density
    randomize_material_density = EventTerm(
        func=contact_mdp.randomize_material_density,
        mode="reset",
        params={
            # "packing_ratio_range": (0.5, 1.0),
            # "bulk_density_range": (1000.0, 3000.0),

            "packing_ratio_range": (1.0, 1.0),
            "bulk_density_range": (2700.0, 2700.0),

            # "packing_ratio_range": (0.6, 0.6),
            # "bulk_density_range": (1500.0, 1500.0),

            # "packing_ratio_range": (0.6, 0.6),
            # "bulk_density_range": (1100.0, 1100.0),

            "contact_solver_name": "physics_callback",
        },
    )