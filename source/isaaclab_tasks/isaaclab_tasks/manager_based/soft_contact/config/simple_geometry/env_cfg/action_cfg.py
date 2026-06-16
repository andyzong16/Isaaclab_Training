# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.soft_contact.mdp as contact_mdp
from isaaclab_tasks.manager_based.soft_contact import IntruderGeometryCfg, PhysicsCallbackActionCfg
from isaaclab_tasks.manager_based.soft_contact import (
    BoxColliderCfg,  # noqa: F401
    PhysicsCallbackActionCfg,
    PlaneColliderCfg,  # noqa: F401
    SphereColliderCfg,  # noqa: F401
)


# collider_cfg = PlaneColliderCfg(
#     contact_edge_x=(-0.1, 0.1),
#     contact_edge_y=(-0.05, 0.05),
#     contact_edge_z=(-0.04, 0.0),
#     resolution=(5, 5),
# )

collider_cfg = BoxColliderCfg(
    contact_edge_x=(-0.1, 0.1),
    contact_edge_y=(-0.05, 0.05),
    contact_edge_z=(-0.04, 0.04),
    # resolution=(2, 2),
    resolution=(5, 5),
)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    """
    Contact solver.
    """
    physics_callback = PhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*"],
        # backend="2D-warp",
        backend="3D-warp",
        # backend="2D",
        # backend="3D",
        # backend="spring-damper",
        intruder_geometry_cfg=collider_cfg,
        # enable_ema_filter=True,
        enable_ema_filter=False,
        contact_threshold=40.0,
        debug_vis=True,
        # debug_vis=False,
        contact_vis_scale=1000.0
    )

    # # kinematics animator
    # velocity_setter = contact_mdp.VelocitySetActionCfg(
    #     asset_name="robot",
    #     velocity_stages=[
    #             {"step": 0, "lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.1, 0.1), "lin_vel_z": (-0.1, -0.1), "ang_vel_z": (0.0, 0.0)},
    #             # {"step": 150, "lin_vel_y": (0.1, 0.1), "lin_vel_z": (0.1, 0.1), "ang_vel_z": (0.0, 0.0)},
    #             # {"step": 300, "lin_vel_y": (0.1, 0.1), "lin_vel_z": (-0.1, -0.1), "ang_vel_z": (0.0, 0.0)},
    #             # {"step": 450, "lin_vel_y": (0.1, 0.1), "lin_vel_z": (0.1, 0.1), "ang_vel_z": (0.0, 0.0)},
    #         ],
    # )
