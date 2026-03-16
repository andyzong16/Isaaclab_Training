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
