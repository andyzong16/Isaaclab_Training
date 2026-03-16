# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.soft_contact import (
    BoxColliderCfg,  # noqa: F401
    PhysicsCallbackActionCfg,
    PlaneColliderCfg,  # noqa: F401
    SphereColliderCfg,  # noqa: F401
)

# g1_foot_geometry_cfg = PlaneColliderCfg(
#     contact_edge_x=(-0.065, 0.141),
#     contact_edge_y=(-0.0368, 0.0368),
#     contact_edge_z=(-0.03539, 0.0),
#     resolution=(5, 5),
# )

g1_foot_geometry_cfg = BoxColliderCfg(
    contact_edge_x=(-0.065, 0.141),
    contact_edge_y=(-0.0368, 0.0368),
    contact_edge_z=(-0.03539, 0.0),
    resolution=(5, 5),
)

# g1_foot_geometry_cfg = SphereColliderCfg(
#     radius=0.05,
#     center=(0.0, 0.0, 0.0),
#     resolution=(8, 8),
# )


@configclass
class G1ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )

    """
    Contact solver.
    """
    physics_callback = PhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*_ankle_roll_link"],
        # backend="2D",
        # backend="3D",
        backend="3D-warp",
        intruder_geometry_cfg=g1_foot_geometry_cfg,
        enable_ema_filter=True,
        contact_threshold=5.0,
        debug_vis=True,
    )
