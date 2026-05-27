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

def foot_contact_forces_raw(
    env: ManagerBasedRLEnv,
    action_term_name: str = "physics_callback",
    threshold: float = 1.0,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    action_term = env.action_manager.get_term(action_term_name)
    contact_forces = action_term.contact_solver.contact_wrench[:, :, :3]  # (num_envs, num_body_ids, 3)
    contact_forces = contact_forces * (contact_forces > threshold).float()
    forces_flat = contact_forces.reshape(env.num_envs, -1)
    return forces_flat

def foot_contact_forces_raw_hybrid(
    env: ManagerBasedRLEnv,
    rigid_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    soft_contact_sensor_name: str = "physics_callback",
    rigid_force_filter_threshold: float = 1.0,
    soft_force_filter_threshold: float = 1.0,
) -> torch.Tensor:
    """Hybrid foot contact forces observation selecting from the authoritative solver."""
    rigid_contact_sensor: ContactSensor = env.scene.sensors[rigid_contact_sensor_cfg.name]
    soft_contact_sensor = env.action_manager.get_term(soft_contact_sensor_name).contact_solver

    rigid_contact_forces = rigid_contact_sensor.data.net_forces_w[:, rigid_contact_sensor_cfg.body_ids, :]
    soft_contact_forces = soft_contact_sensor.contact_wrench[:, :, :3]

    is_soft = soft_contact_sensor.data.is_sensor_active  # [B, N_feet]

    forces = torch.where(is_soft.unsqueeze(-1), soft_contact_forces, rigid_contact_forces)
    # print("forces: ", forces)

    threshold = (
        torch.where(is_soft, soft_force_filter_threshold, rigid_force_filter_threshold)
        .unsqueeze(-1)
        .expand(-1, -1, 3)
        .reshape(env.num_envs, -1)
    )
    forces = forces.reshape(env.num_envs, -1)
    mask = torch.linalg.norm(forces, dim=-1, keepdim=True) > threshold
    forces = forces * mask.float()

    return forces