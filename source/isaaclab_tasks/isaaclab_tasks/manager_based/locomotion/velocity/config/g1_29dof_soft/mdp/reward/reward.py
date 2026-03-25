# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def find_physx_mjwarp_mapping(mjwarp_joint_names, physx_joint_names):
    """
    Finds the mapping between PhysX and MJWarp joint names.
    Returns a tuple of two lists: (mjc_to_physx, physx_to_mjc).
    """
    mjc_to_physx = []
    physx_to_mjc = []
    for j in mjwarp_joint_names:
        if j in physx_joint_names:
            mjc_to_physx.append(physx_joint_names.index(j))

    for j in physx_joint_names:
        if j in mjwarp_joint_names:
            physx_to_mjc.append(mjwarp_joint_names.index(j))

    return mjc_to_physx, physx_to_mjc


"""
whole-body centroidal momentum penalties.
"""


class angular_momentum_l2(ManagerTermBase):
    """
    compute the L2 norm of the whole-body centroidal (pelvis) angular momentum.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        import sys

        # TODO: figure out non sys path way to import cusadi
        sys.path.append("../")
        import casadi
        from cusadi import CASADI_FUNCTION_DIR, CusadiFunction

        super().__init__(cfg, env)
        self.centroidal_ang_momentum = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=self.device)
        self.cusadi_func = CusadiFunction(
            casadi.Function.load(os.path.join(CASADI_FUNCTION_DIR, "g1_29dof_ang_momentum_func.casadi")),  # type: ignore
            num_instances=env.num_envs,
        )

        # get joint mapping index
        assert len(cfg.params["physx_joint_names"]) == len(cfg.params["mjw_joint_names"]), (
            "PhysX and MJWarp joint name lists must have the same length."
        )
        self.mjc_to_physx, self.physx_to_mjc = find_physx_mjwarp_mapping(
            cfg.params["mjw_joint_names"], cfg.params["physx_joint_names"]
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        physx_joint_names: list[str],
        mjw_joint_names: list[str],
    ) -> torch.Tensor:

        asset: Articulation = env.scene[asset_cfg.name]
        base_pos = asset.data.root_pos_w - env.scene.env_origins  # (num_envs, 3)
        base_quat = asset.data.root_quat_w  # (num_envs, 4)
        # align physx joint order to mjw order
        joint_pos = asset.data.joint_pos.clone()[:, self.mjc_to_physx]  # (num_envs, num_dofs)

        base_lin_vel = asset.data.root_lin_vel_w  # (num_envs, 3)
        base_ang_vel = asset.data.root_ang_vel_w  # (num_envs, 3)
        # align physx joint order to mjw order
        joint_vel = asset.data.joint_vel.clone()[:, self.mjc_to_physx]  # (num_envs, num_dofs)

        q_pos = torch.cat([base_pos, base_quat, joint_pos], dim=-1)
        q_vel = torch.cat([base_lin_vel, base_ang_vel, joint_vel], dim=-1)
        self.cusadi_func.evaluate([q_pos.double(), q_vel.double()])

        # whole body centroidal angular momentum wrt global frame
        self.centroidal_ang_momentum = self.cusadi_func.getDenseOutput(0).squeeze(-1).float()
        return torch.sqrt(torch.square(self.centroidal_ang_momentum).sum(dim=-1))


"""
contact reward
"""


def feet_air_time_positive_biped_hybrid(
    env,
    command_name: str,
    threshold: float,
    rigid_contact_sensor_cfg: SceneEntityCfg,
    soft_contact_sensor_name: str = "physics_callback",
) -> torch.Tensor:
    rigid_contact_sensor: ContactSensor = env.scene.sensors[rigid_contact_sensor_cfg.name]
    soft_contact_sensor = env.action_manager.get_term(soft_contact_sensor_name).contact_solver

    # combine air time: whichever solver is active has the correct (smaller) value
    air_time_rigid = rigid_contact_sensor.data.current_air_time[:, rigid_contact_sensor_cfg.body_ids]
    air_time_soft = soft_contact_sensor.data.current_air_time
    air_time = torch.minimum(air_time_rigid, air_time_soft)

    # combine contact time: whichever solver is active has the non-zero value
    contact_time_rigid = rigid_contact_sensor.data.current_contact_time[:, rigid_contact_sensor_cfg.body_ids]
    contact_time_soft = soft_contact_sensor.data.current_contact_time
    contact_time = torch.maximum(contact_time_rigid, contact_time_soft)  # pick whichever active

    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)

    linear_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    angular_norm = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    total_norm = linear_norm + angular_norm
    reward *= total_norm > 0.05

    return reward


def feet_slide_hybrid(
    env,
    rigid_contact_sensor_cfg: SceneEntityCfg,
    soft_contact_sensor_name: str = "physics_callback",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rigid_contact_threshold: float = 1.0,
    soft_contact_threshold: float = 5.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[rigid_contact_sensor_cfg.name]
    contact_solver = env.action_manager.get_term(soft_contact_sensor_name).contact_solver

    rigid_contacts = (
        contact_sensor.data.net_forces_w_history[:, :, rigid_contact_sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > rigid_contact_threshold
    )
    soft_contacts = (
        contact_solver.data.net_forces_w_history[:, :, :, :].norm(dim=-1).max(dim=1)[0] > soft_contact_threshold
    )
    contacts = torch.maximum(rigid_contacts, soft_contacts)  # pick whichever active

    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


def no_fly_hybrid(
    env: ManagerBasedRLEnv,
    rigid_contact_sensor_cfg: SceneEntityCfg,
    soft_contact_sensor_name: str = "physics_callback",
    rigid_contact_threshold: float = 5.0,
    soft_contact_threshold: float = 5.0,
    command_name: str = "base_velocity",
    velocity_threshold: float = 1.5,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[rigid_contact_sensor_cfg.name]
    contact_solver = env.action_manager.get_term(soft_contact_sensor_name).contact_solver

    rigid_contact = (
        torch.max(
            torch.norm(contact_sensor.data.net_forces_w_history[:, :, rigid_contact_sensor_cfg.body_ids], dim=-1), dim=1
        )[0]
        > rigid_contact_threshold
    )
    soft_contact = (
        torch.max(torch.norm(contact_solver.data.net_forces_w_history, dim=-1), dim=1)[0] > soft_contact_threshold
    )
    is_contact = torch.maximum(rigid_contact, soft_contact)  # pick whichever active

    linear_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    is_active = linear_norm < velocity_threshold
    reward = torch.sum(is_contact, dim=-1) < 0.5
    return reward * is_active


def foot_force_hybrid(
    env: ManagerBasedRLEnv,
    rigid_contact_sensor_cfg: SceneEntityCfg,
    soft_contact_sensor_name: str = "physics_callback",
    threshold: float = 500,
    max_reward: float = 400,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[rigid_contact_sensor_cfg.name]
    contact_solver = env.action_manager.get_term(soft_contact_sensor_name).contact_solver

    # sum: mutually exclusive per terrain type
    rigid_fz = contact_sensor.data.net_forces_w[:, rigid_contact_sensor_cfg.body_ids, 2]
    soft_fz = contact_solver.data.net_forces_w[:, :, 2]
    reward = (rigid_fz + soft_fz).norm(dim=-1)

    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    return reward.clamp(min=0, max=max_reward)


def reward_soft_landing_hybrid(
    env: ManagerBasedRLEnv,
    rigid_contact_sensor_cfg: SceneEntityCfg,
    soft_contact_sensor_name: str = "physics_callback",
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize high impact forces at landing to encourage soft footfalls."""
    contact_sensor: ContactSensor = env.scene.sensors[rigid_contact_sensor_cfg.name]
    contact_solver = env.action_manager.get_term(soft_contact_sensor_name).contact_solver

    # forces: sum (mutually exclusive per terrain)
    rigid_forces = contact_sensor.data.net_forces_w[:, rigid_contact_sensor_cfg.body_ids, :]  # [B, N, 3]
    soft_forces = contact_solver.data.net_forces_w  # [B, N, 3]
    forces = rigid_forces + soft_forces

    # first_contact: OR (landing event on either terrain)
    rigid_first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, rigid_contact_sensor_cfg.body_ids
    ]  # [B, N]
    soft_first_contact = contact_solver.compute_first_contact(env.step_dt)  # [B, N]
    first_contact = torch.maximum(rigid_first_contact, soft_first_contact)

    force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
    landing_impact = force_magnitude * first_contact.float()  # [B, N]
    cost = torch.sum(landing_impact, dim=1)  # [B]

    num_landings = torch.sum(first_contact.float())
    mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
    env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force

    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        active = ((linear_norm + angular_norm) > command_threshold).float()
        cost = cost * active
    return cost
