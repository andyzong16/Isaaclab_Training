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
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
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
soft contact state
"""


def foot_air_time(
    env: ManagerBasedRLEnv,
    action_term_name: str = "physics_callback",
    filter_time: float = 0.5,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    action_term = env.action_manager.get_term(action_term_name)
    air_time = action_term.contact_solver.data.current_air_time
    air_time = torch.where(air_time > filter_time, 0.0, air_time)  # remove the air time larger than filter_time
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
    threshold: float = 1.0,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    action_term = env.action_manager.get_term(action_term_name)
    contact_forces = action_term.contact_solver.contact_wrench[:, :, :3]  # (num_envs, num_body_ids, 3)
    contact_forces = contact_forces * (contact_forces > threshold).float()
    forces_flat = contact_forces.reshape(env.num_envs, -1)
    return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


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


def terrain_material_parameters(
    env: ManagerBasedRLEnv,
    action_term_name: str = "physics_callback",
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    action_term = env.action_manager.get_term(action_term_name)
    contact_solver = action_term.contact_solver
    friction_coef = contact_solver.terrain_friction
    rho_c = contact_solver.terrain_density / 3000.0  # max rho = 3000.0
    mu_int = contact_solver.terrain_stiffness
    return torch.stack([friction_coef, rho_c, mu_int], dim=-1)


"""
soft contact + rigid contact mixed
"""


class foot_air_time_hybrid(ManagerTermBase):
    """Hybrid foot air time observation with per-foot solver authority tracking."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.last_active_is_soft: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        rigid_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        soft_contact_sensor_name: str = "physics_callback",
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene.sensors[rigid_contact_sensor_cfg.name]
        soft_contact_sensor = env.action_manager.get_term(soft_contact_sensor_name).contact_solver

        air_time_rigid = contact_sensor.data.current_air_time[:, rigid_contact_sensor_cfg.body_ids]
        air_time_soft = soft_contact_sensor.data.current_air_time
        contact_time_rigid = contact_sensor.data.current_contact_time[:, rigid_contact_sensor_cfg.body_ids]
        contact_time_soft = soft_contact_sensor.data.current_contact_time

        # lazy init
        if self.last_active_is_soft is None:
            self.last_active_is_soft = torch.zeros(
                env.num_envs, air_time_rigid.shape[1], dtype=torch.bool, device=self.device
            )

        # update authority based on which solver reports contact
        self.last_active_is_soft[contact_time_rigid > 0] = False
        self.last_active_is_soft[contact_time_soft > 0] = True

        # select air_time from authoritative solver
        return torch.where(self.last_active_is_soft, air_time_soft, air_time_rigid)


class foot_contact_hybrid(ManagerTermBase):
    """Hybrid foot contact observation with per-foot solver authority tracking."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.last_active_is_soft: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        rigid_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        soft_contact_sensor_name: str = "physics_callback",
        rigid_force_threshold: float = 1.0,
        soft_force_threshold: float = 1.0,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene.sensors[rigid_contact_sensor_cfg.name]
        soft_contact_sensor = env.action_manager.get_term(soft_contact_sensor_name).contact_solver

        rigid_contact_forces = contact_sensor.data.net_forces_w[
            :, rigid_contact_sensor_cfg.body_ids, :
        ]  # (num_envs, num_body_ids, 3)
        rigid_contact = (torch.norm(rigid_contact_forces, dim=-1) > rigid_force_threshold).float()

        soft_contact_forces = soft_contact_sensor.contact_wrench[:, :, :3]  # (num_envs, num_body_ids, 3)
        soft_contact = (torch.norm(soft_contact_forces, dim=-1) > soft_force_threshold).float()

        # lazy init
        if self.last_active_is_soft is None:
            self.last_active_is_soft = torch.zeros(
                env.num_envs, rigid_contact.shape[1], dtype=torch.bool, device=self.device
            )

        # update authority based on which solver reports contact
        self.last_active_is_soft[rigid_contact > 0] = False
        self.last_active_is_soft[soft_contact > 0] = True

        # select contact from authoritative solver
        return torch.where(self.last_active_is_soft, soft_contact, rigid_contact)


class foot_contact_forces_hybrid(ManagerTermBase):
    """Hybrid foot contact forces observation with per-foot solver authority tracking."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.last_active_is_soft: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        rigid_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        soft_contact_sensor_name: str = "physics_callback",
        rigid_force_filter_threshold: float = 1.0,
        soft_force_filter_threshold: float = 1.0,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene.sensors[rigid_contact_sensor_cfg.name]
        soft_contact_sensor = env.action_manager.get_term(soft_contact_sensor_name).contact_solver

        rigid_contact_forces = contact_sensor.data.net_forces_w[
            :, rigid_contact_sensor_cfg.body_ids, :
        ]  # (num_envs, num_body_ids, 3)
        soft_contact_forces = soft_contact_sensor.contact_wrench[:, :, :3]  # (num_envs, num_body_ids, 3)

        # lazy init (use per-foot authority, before reshape)
        if self.last_active_is_soft is None:
            self.last_active_is_soft = torch.zeros(
                env.num_envs, rigid_contact_forces.shape[1], dtype=torch.bool, device=self.device
            )

        # update authority: use contact_time from underlying sensors
        contact_time_rigid = contact_sensor.data.current_contact_time[:, rigid_contact_sensor_cfg.body_ids]
        contact_time_soft = soft_contact_sensor.data.current_contact_time
        self.last_active_is_soft[contact_time_rigid > 0] = False
        self.last_active_is_soft[contact_time_soft > 0] = True

        # select forces from authoritative solver (per foot, then threshold & process)
        forces = torch.where(
            self.last_active_is_soft.unsqueeze(-1), soft_contact_forces, rigid_contact_forces
        )  # (num_envs, num_body_ids, 3)

        # expand per-foot threshold to match (num_envs, num_body_ids * 3) flattened forces
        threshold = (
            torch.where(self.last_active_is_soft, soft_force_filter_threshold, rigid_force_filter_threshold)
            .unsqueeze(-1)
            .expand(-1, -1, 3)
            .reshape(env.num_envs, -1)
        )
        forces = forces.reshape(env.num_envs, -1)
        forces = forces * (forces > threshold).float()

        return torch.sign(forces) * torch.log1p(torch.abs(forces))


class terrain_material_parameters_hybrid(ManagerTermBase):
    """Hybrid terrain material parameters observation with per-foot solver authority tracking."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.last_active_is_soft: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        rigid_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        soft_contact_sensor_name: str = "physics_callback",
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene.sensors[rigid_contact_sensor_cfg.name]
        soft_contact_sensor = env.action_manager.get_term(soft_contact_sensor_name).contact_solver

        # lazy init
        if self.last_active_is_soft is None:
            num_feet = contact_sensor.data.net_forces_w[:, rigid_contact_sensor_cfg.body_ids, :].shape[1]
            self.last_active_is_soft = torch.zeros(env.num_envs, num_feet, dtype=torch.bool, device=self.device)

        # update authority using contact_time
        contact_time_rigid = contact_sensor.data.current_contact_time[:, rigid_contact_sensor_cfg.body_ids]
        contact_time_soft = soft_contact_sensor.data.current_contact_time
        self.last_active_is_soft[contact_time_rigid > 0] = False
        self.last_active_is_soft[contact_time_soft > 0] = True

        # aggregate per-env: soft if ANY foot is on soft terrain
        on_soft_ground = self.last_active_is_soft.any(dim=-1).float()

        mu_rigid = 1.0
        friction_rigid = 1.0
        rho_c_rigid = 3000.0

        friction_coef = soft_contact_sensor.terrain_friction * on_soft_ground + (1 - on_soft_ground) * friction_rigid
        rho_c = (soft_contact_sensor.terrain_density / 3000.0) * on_soft_ground + (1 - on_soft_ground) * (
            rho_c_rigid / 3000.0
        )
        mu_int = soft_contact_sensor.terrain_stiffness * on_soft_ground + (1 - on_soft_ground) * mu_rigid
        return torch.stack([friction_coef, rho_c, mu_int], dim=-1)
