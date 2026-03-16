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

    reset_base = EventTerm(
        func=contact_mdp.reset_root_state_uniform, 
        mode="reset",
        params={
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (-0.0, 0.0),
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
            "friction_range": (0.1, 1.0),
            "contact_solver_name": "physics_callback",
        },
    )

    # randomize terrain stiffness
    randomize_stiffness = EventTerm(
        func=contact_mdp.randomize_terrain_stiffness,
        mode="reset",
        params={
            "stiffness_range": (0.1, 0.6),
            "contact_solver_name": "physics_callback",
        },
    )