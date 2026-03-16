# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_body_names,
    record_dtype,
    record_shape,
)

"""
body kinematics.
"""


@generic_io_descriptor(observation_type="BodyState", on_inspect=[record_shape, record_dtype, record_body_names])
def foot_pos_w(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    standing_position_foot_z: float = 0.039,
) -> torch.Tensor:
    """The flattened body poses of the asset w.r.t the env.scene.origin.

    Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their poses returned.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        The position of bodies in articulation [num_env, 3 * num_bodies].
        Output is stacked horizontally per body.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # access the body poses in world frame
    pose = asset.data.body_pose_w[:, asset_cfg.body_ids, :7]
    pose[..., :3] = pose[..., :3] - env.scene.env_origins.unsqueeze(1)

    pos = pose[..., :3]  # (num_envs, num_bodies, 3)
    quat = pose[..., 3:7]  # (num_envs, num_bodies, 4)
    rot = math_utils.matrix_from_quat(quat)  # (num_envs, num_bodies, 3, 3)

    local_pos = torch.tensor([0.0, 0.0, -standing_position_foot_z], device=pos.device).reshape(1, 1, 3)  # (1, 1, 3)
    pos_foot = pos + (rot @ local_pos.unsqueeze(-1)).squeeze(-1)  # (num_envs, num_bodies, 3)

    return pos_foot.reshape(env.num_envs, -1)


@generic_io_descriptor(observation_type="BodyState", on_inspect=[record_shape, record_dtype, record_body_names])
def foot_height(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    standing_position_foot_z: float = 0.039,
) -> torch.Tensor:
    """The flattened body poses of the asset w.r.t the env.scene.origin.

    Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their poses returned.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        The position of bodies in articulation [num_env, 3 * num_bodies].
        Output is stacked horizontally per body.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # access the body poses in world frame
    pose = asset.data.body_pose_w[:, asset_cfg.body_ids, :7]
    pose[..., :3] = pose[..., :3] - env.scene.env_origins.unsqueeze(1)

    pos = pose[..., :3]  # (num_envs, num_bodies, 3)
    quat = pose[..., 3:7]  # (num_envs, num_bodies, 4)
    rot = math_utils.matrix_from_quat(quat)  # (num_envs, num_bodies, 3, 3)

    local_pos = torch.tensor([0.0, 0.0, -standing_position_foot_z], device=pos.device).reshape(1, 1, 3)  # (1, 1, 3)
    pos_foot = pos + (rot @ local_pos.unsqueeze(-1)).squeeze(-1)  # (num_envs, num_bodies, 3)

    return pos_foot[:, :, 2].reshape(env.num_envs, -1)


"""
contact.
"""


def hard_contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("sensor")) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (num_envs, num_body_ids, 3)
    contact_forces = contact_forces.reshape(-1, contact_forces.shape[1] * contact_forces.shape[2])
    print(contact_forces)
    return contact_forces


def foot_hard_contact_forces(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("sensor")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.force_matrix_w[
        :, sensor_cfg.body_ids, :
    ]  # (num_envs, num_body_ids, num_filter, 3)
    friction_forces = contact_sensor.data.friction_forces_w[
        :, sensor_cfg.body_ids, :
    ]  # (num_envs, num_body_ids, num_filter, 3)
    total_contact_forces = (contact_forces + friction_forces).sum(dim=2)  # (num_envs, num_body_ids, 3)
    total_contact_forces = total_contact_forces.reshape(
        -1, total_contact_forces.shape[1] * total_contact_forces.shape[2]
    )
    # print(total_contact_forces)
    return total_contact_forces


"""
soft contact
"""


def foot_air_time(
    env: ManagerBasedRLEnv,
    action_term_name: str = "physics_callback",
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    action_term = env.action_manager.get_term(action_term_name)
    air_time = action_term.contact_solver.data.current_air_time
    return air_time


def foot_contact(
    env: ManagerBasedRLEnv,
    action_term_name: str = "physics_callback",
    threshold: float = 1.0,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    action_term = env.action_manager.get_term(action_term_name)
    contact_forces = action_term.contact_solver.contact_wrench[:, :, :3]  # (num_envs, num_body_ids, 3)
    contact = (torch.norm(contact_forces, dim=-1) > threshold).float()
    return contact


def foot_contact_forces(
    env: ManagerBasedRLEnv,
    action_term_name: str = "physics_callback",
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    action_term = env.action_manager.get_term(action_term_name)
    contact_forces = action_term.contact_solver.contact_wrench[:, :, :3]  # (num_envs, num_body_ids, 3)
    forces_flat = contact_forces.reshape(env.num_envs, -1)
    return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def foot_contact_forces_raw(
    env: ManagerBasedRLEnv,
    action_term_name: str = "physics_callback",
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    action_term = env.action_manager.get_term(action_term_name)
    contact_forces = action_term.contact_solver.contact_wrench[:, :, :3]  # (num_envs, num_body_ids, 3)
    forces_flat = contact_forces.reshape(env.num_envs, -1)
    return forces_flat
