# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.soft_contact.mdp as contact_mdp
from isaaclab_tasks.manager_based.soft_contact import IntruderGeometryCfg, PhysicsCallbackActionCfg

@configclass    
class FootGeometryCfg(IntruderGeometryCfg):
    """Configuration for the intruder geometry used in soft contact modeling."""
    contact_edge_x: tuple[float, float] = (-0.1, 0.1)  # length in x direction (m)
    contact_edge_y: tuple[float, float] = (-0.05, 0.05)  # length in y direction (m)
    contact_edge_z: tuple[float, float] = (-0.04, 0.0)  # length in z direction (m)
    num_contact_points: int = 5 * 5

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    """
    Contact solver.
    """
    physics_callback = PhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*"],
        # backend="2D",
        backend="3D",
        intruder_geometry_cfg=FootGeometryCfg(),
        enable_ema_filter=True,
        contact_threshold=200.0,
        debug_vis=True,
    )

    # kinematics animator
    # velocity_setter = contact_mdp.VelocitySetActionCfg(
    #     asset_name="robot",
    #     velocity_stages=[
    #             {"step": 0, "lin_vel_z": (-0.04, -0.04), "ang_vel_z": (0.1, 0.1)},
    #             {"step": 150, "lin_vel_z": (0.04, 0.04), "ang_vel_z": (0.1, 0.1)},
    #             {"step": 300, "lin_vel_z": (-0.04, -0.04), "ang_vel_z": (0.1, 0.1)}, 
    #             {"step": 450, "lin_vel_z": (0.04, 0.04), "ang_vel_z": (0.1, 0.1)},
    #         ],
    # )