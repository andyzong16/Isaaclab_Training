# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import warp as wp


from .soft_contact_model_data import SoftContactData
from .material import Material3DRFTCfg
from .collider import Collider, ColliderCfg, PlaneColliderCfg
from .kernels import (
    compute_contact_point_lin_vel_w,
    compute_contact_point_pos_w,
    compute_contact_wrench,
    compute_intrusion_angle,
    compute_normal_direction_w,
    compute_r_direction_w,
    compute_resistive_force,
    compute_t_direction_w,
    compute_tilt_angle,
    compute_twist_angle,
    compute_v_direction_w,
    expand_vec3_to_3d_vec3,
    reset,
    transform_global_wrench_to_body,
    update_array_with_index,
    zero_wrench,
)


# Backward compatibility alias
IntruderGeometryCfg = PlaneColliderCfg


class RFT_3D:
    def __init__(
        self,
        num_envs: int,
        num_bodies: int,
        device: torch.device | str,
        dt: float,
        material_cfg: Material3DRFTCfg,
        collider_cfg: ColliderCfg,
        history_length: int = 3,
        contact_threshold: float = 10,
        enable_ema_filter: bool = True,
    ) -> None:
        """
        Soft contact model based on 3D RFT proposed in
        https://www.pnas.org/doi/10.1073/pnas.2214017120

        Args:
            num_envs: number of parallel environments
            num_bodies: number of bodies using soft contact model per env
            device: torch device
            dt: simulation time step
            history_length: length of history for force tracking
            material_cfg: material configuration
            collider_cfg: collider geometry configuration
        """

        self.cfg = material_cfg
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self.device = device
        self.dt = dt
        self.c_r = 100 / (1 / self.dt)  # 100/f (e.g. f=2000hz -> 0.05)
        self.history_length = history_length
        self.enable_ema_filter = enable_ema_filter
        self.contact_threshold = contact_threshold

        # Build collider geometry
        self.collider = Collider(collider_cfg, num_envs, num_bodies, device)
        self.num_contact_points = self.collider.num_contact_points
        self.surface_area = self.collider.surface_area

        self._data: SoftContactData = SoftContactData()

        self.create_buffers()
        self.initialize_data()

        print("-" * 40)
        print("3D RFT soft contact.")
        print("backend: warp")
        print(f"Number of envs: {self.num_envs}")
        print(f"Number of bodies per env: {self.num_bodies}")
        print(f"Number of contact points per body: {self.num_contact_points}")
        print(f"Contact surface area per body: {self.surface_area:.2f} m^2")
        print(f"mu int: {self.cfg.mu_int:.2f}")
        print(f"rho c: {self.cfg.rho_c:.2f} kg/m^3")
        print(f"mu_surf: {self.cfg.dynamic_friction_coef:.2f}")
        print("-" * 40)
        print("\n")

        self.capture()

    """
    make graph of contact evaluation
    """

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._eval_contacts()
            self.graph = capture.graph
        else:
            self.graph = None

    def create_buffers(self):
        """
        create warp arrays
        """
        # torch buffers
        self.body_pos_torch = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_quat_torch = torch.zeros((self.num_envs, self.num_bodies, 4), device=self.device)
        self.body_lin_vel_torch = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_ang_vel_torch = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)

        # body state copied from torch
        self.body_pos = wp.from_torch(self.body_pos_torch, dtype=wp.vec3f)
        self.body_quat = wp.from_torch(self.body_quat_torch, dtype=wp.quatf)
        self.body_lin_vel = wp.from_torch(self.body_lin_vel_torch, dtype=wp.vec3f)
        self.body_ang_vel = wp.from_torch(self.body_ang_vel_torch, dtype=wp.vec3f)

        # contact points (provided by collider)
        self.contact_point_local = self.collider.contact_point_local
        self.normal_dir_local = self.collider.normal_dir_local
        self.contact_point_pos = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.contact_point_lin_vel = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.contact_point_lin_vel_prev = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )

        # characteristics angle
        self.contact_point_tilt_angle = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.float32, device=self.device
        )
        self.contact_point_intrusion_angle = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.float32, device=self.device
        )
        self.contact_point_twist_angle = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.float32, device=self.device
        )

        # local coordinate
        self.n_dir = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.r_dir = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.t_dir = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.z_dir = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.v_dir = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.n_rtz_dir = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )

        # contact forces
        self.contact_point_force = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.contact_point_torque = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.contact_force = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self.contact_torque = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self.contact_force_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self.contact_torque_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)

        # cache for EMA filter
        self.alpha_unfiltered = wp.zeros(
            (self.num_envs, self.num_bodies * self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.alpha_filtered = wp.zeros(
            (self.num_envs, self.num_bodies * self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.resitive_force = wp.zeros(
            (self.num_envs, self.num_bodies * self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.tau_r = wp.zeros(
            (self.num_envs, self.num_bodies * self.num_contact_points), dtype=wp.float32, device=self.device
        )

        # material parameters
        self.static_friction_coef = wp.full(
            self.num_envs, self.cfg.static_friction_coef, dtype=wp.float32, device=self.device
        )
        self.dynamic_friction_coef = wp.full(
            self.num_envs, self.cfg.dynamic_friction_coef, dtype=wp.float32, device=self.device
        )
        self.rho_c = wp.full(self.num_envs, self.cfg.rho_c, dtype=wp.float32, device=self.device)
        self.mu_int = wp.full(self.num_envs, self.cfg.mu_int, dtype=wp.float32, device=self.device)
        self.coef_1 = wp.array1d(self.cfg.coef_1, dtype=wp.float32, device=self.device)
        self.coef_2 = wp.array1d(self.cfg.coef_2, dtype=wp.float32, device=self.device)
        self.coef_3 = wp.array1d(self.cfg.coef_3, dtype=wp.float32, device=self.device)

        # timestamps
        self._timestamp = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._timestamp_last_update = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Bind torch buffers to warp buffers
        self.torch_contact_point_pos = wp.to_torch(self.contact_point_pos)
        self.torch_contact_force = wp.to_torch(self.contact_force)
        self.torch_contact_torque = wp.to_torch(self.contact_torque)
        self.torch_contact_force_b = wp.to_torch(self.contact_force_b)
        self.torch_contact_torque_b = wp.to_torch(self.contact_torque_b)
        self.torch_contact_point_force = wp.to_torch(self.contact_point_force)
        self.torch_contact_point_torque = wp.to_torch(self.contact_point_torque)


    def initialize_data(self) -> None:
        """
        Initialize soft contact data.
        """
        self._data.net_forces_w = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self._data.net_forces_w_history = torch.zeros(
            (self.num_envs, self.history_length, self.num_bodies, 3), device=self.device
        )
        self._data.force_matrix_w = torch.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device
        )
        self._data.force_matrix_w_history = torch.zeros(
            (self.num_envs, self.history_length, self.num_bodies, self.num_contact_points, 3), device=self.device
        )
        self._data.last_air_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.current_air_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.last_contact_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.current_contact_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)

    """
    properties.
    """

    @property
    def data(self) -> SoftContactData:
        return self._data

    @property
    def contact_wrench(self) -> torch.Tensor:
        contact_wrench = torch.cat(
            (self.torch_contact_force, self.torch_contact_torque), dim=-1
        )  # (num_envs, num_bodies, 6)
        return contact_wrench

    @property
    def contact_wrench_b(self) -> torch.Tensor:
        contact_wrench_b = torch.cat(
            (self.torch_contact_force_b, self.torch_contact_torque_b), dim=-1
        )  # (num_envs, num_bodies, 6)
        return contact_wrench_b

    @property
    def contact_point_wrench(self) -> torch.Tensor:
        contact_point_wrench = torch.cat(
            (self.torch_contact_point_force, self.torch_contact_point_torque), dim=-1
        )  # (num_envs, num_bodies, num_contact_points, 6)
        return contact_point_wrench

    """
    operations.
    """

    def update(
        self, body_pos: torch.Tensor, body_quat: torch.Tensor, body_lin_vel: torch.Tensor, body_ang_vel: torch.Tensor
    ):
        """
        Update soft contact model.

        Args:
            body_pos: intruder position. (num_envs, num_bodies, 3)
            body_quat: intruder orientation in quaternion form. (num_envs, num_bodies, 4)
            body_lin_vel: intruder linear velocity wrt global frame. (num_envs, num_bodies, 3)
            body_ang_vel: intruder angular velocity wrt global frame. (num_envs, num_bodies, 3)
        """
        # copy to torch buffer
        self.body_pos_torch[:] = body_pos
        self.body_quat_torch[:] = body_quat[:, :, [1, 2, 3, 0]]  # convert (w, x, y, z) to (x, y, z, w)
        self.body_lin_vel_torch[:] = body_lin_vel
        self.body_ang_vel_torch[:] = body_ang_vel

        # evaluate contact forces
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._eval_contacts()

        # update timestamp and data
        self._timestamp += self.dt
        self._update_data(torch.arange(self.num_envs, device=self.device))
        self._timestamp_last_update[:] = self._timestamp[:]

    def randomize_ground_stiffness(self, env_ids: torch.Tensor, mu_int: torch.Tensor) -> None:
        """
        Update ground stiffness (N/m) for each env.
        Implementation is similar to terrain curriculum used in terrain importer class.
        This can be triggered by curriculum manager.

        Args:
            env_ids: tensor of env ids to update
            mu_int: tensor of mu_int values (len(env_ids), )
        """
        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[wp.from_torch(env_ids, dtype=wp.int64), wp.from_torch(mu_int, dtype=wp.float32), self.mu_int],
            device=self.device,
        )

    def update_material_density(self, env_ids: torch.Tensor, packing_density: torch.Tensor, bulk_density: torch.Tensor) -> None:
        """
        Update material density for each env.
        This can be triggered by event manager.

        Args:
            env_ids: tensor of env ids to update
            packing_density: tensor of packing densities (len(env_ids), )
            bulk_density: tensor of bulk densities (len(env_ids), )
        """
        rho_c = bulk_density * packing_density
        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[wp.from_torch(env_ids, dtype=wp.int64), wp.from_torch(rho_c, dtype=wp.float32), self.rho_c],
            device=self.device,
        )

    def update_friction_params(
        self, env_ids: torch.Tensor, static_friction_coef: torch.Tensor, dynamic_friction_coef: torch.Tensor
    ) -> None:
        """
        Update friction coefficients for each env.
        This can be triggered by event manager.

        Args:
            env_ids: tensor of env ids to update
            static_friction_coef: tensor of static friction coefficients (len(env_ids), )
            dynamic_friction_coef: tensor of dynamic friction coefficients (len(env_ids), )
        """
        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[
                wp.from_torch(env_ids, dtype=wp.int64),
                wp.from_torch(static_friction_coef, dtype=wp.float32),
                self.static_friction_coef,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[
                wp.from_torch(env_ids, dtype=wp.int64),
                wp.from_torch(dynamic_friction_coef, dtype=wp.float32),
                self.dynamic_friction_coef,
            ],
            device=self.device,
        )

    """
    helper functions.
    """

    def _update_data(self, env_ids: torch.Tensor) -> None:
        """
        Update soft contact data.
        Majority of implementations are from IsaacLab's contact sensor class.

        Args:
            env_ids: tensor of env ids to update
        """
        self._data.net_forces_w[env_ids, :, :] = self.torch_contact_force[env_ids, :, :]  # type: ignore
        self._data.force_matrix_w[env_ids, :, :, :] = self.torch_contact_point_force[env_ids, :, :, :]  # type: ignore
        if self.history_length > 0:
            self._data.net_forces_w_history[env_ids] = self._data.net_forces_w_history[env_ids].roll(shifts=1, dims=1)  # type: ignore
            self._data.net_forces_w_history[env_ids, 0] = self._data.net_forces_w[env_ids]  # type: ignore

            self._data.force_matrix_w_history[env_ids] = self._data.force_matrix_w_history[env_ids].roll(
                shifts=1, dims=1
            )  # type: ignore
            self._data.force_matrix_w_history[env_ids, 0] = self._data.force_matrix_w[env_ids]  # type: ignore

        # track air time (see contact sensor class)
        elapsed_time = self._timestamp[env_ids] - self._timestamp_last_update[env_ids]
        is_contact = torch.norm(self._data.net_forces_w[env_ids, :, :], dim=-1) > self.contact_threshold  # type: ignore
        is_first_contact = (self._data.current_air_time[env_ids] > 0) * is_contact  # type: ignore
        is_first_detached = (self._data.current_contact_time[env_ids] > 0) * ~is_contact  # type: ignore
        # -- update the last contact time if body has just become in contact
        self._data.last_air_time[env_ids] = torch.where(  # type: ignore
            is_first_contact,
            self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1),  # type: ignore
            self._data.last_air_time[env_ids],  # type: ignore
        )
        # -- increment time for bodies that are not in contact
        self._data.current_air_time[env_ids] = torch.where(  # type: ignore
            ~is_contact,
            self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1),
            0.0,  # type: ignore
        )
        # -- update the last contact time if body has just detached
        self._data.last_contact_time[env_ids] = torch.where(  # type: ignore
            is_first_detached,
            self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1),  # type: ignore
            self._data.last_contact_time[env_ids],  # type: ignore
        )
        # -- increment time for bodies that are in contact
        self._data.current_contact_time[env_ids] = torch.where(  # type: ignore
            is_contact,
            self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1),
            0.0,  # type: ignore
        )

    def _eval_contacts(self) -> None:
        """
        Update contact points' kinematic states and compute contact forces.
        """
        # step1: calculate contact pos, lin vel, and surface normal
        # compute contact points in global frame
        wp.launch(
            kernel=compute_contact_point_pos_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.body_pos, self.body_quat, self.contact_point_local, self.contact_point_pos],
            device=self.device,
        )

        # compute contact point linear velocity in global frame
        wp.launch(
            kernel=compute_contact_point_lin_vel_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[
                self.body_pos,
                self.body_lin_vel,
                self.body_ang_vel,
                self.contact_point_pos,
                self.contact_point_lin_vel,
            ],
            device=self.device,
        )

        # step2: Find local coordinate frame {r, theta, z}
        # compute unit vectors (z, n, v, r, t)
        # see S7 eq.4 from https://www.pnas.org/doi/10.1073/pnas.2214017120

        # compute contact normal
        # currently, we only consider surface normal at the bottom of foot which faces to -z in local frame.
        wp.launch(
            kernel=compute_normal_direction_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.body_quat, self.normal_dir_local, self.n_dir],
            device=self.device,
        )

        Z_DIR = wp.vec3f(0.0, 0.0, 1.0)
        wp.launch(
            kernel=expand_vec3_to_3d_vec3,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[Z_DIR, self.z_dir],
            device=self.device,
        )
        wp.launch(
            kernel=compute_v_direction_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.contact_point_lin_vel, self.v_dir],
            device=self.device,
        )
        wp.launch(
            kernel=compute_r_direction_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.n_dir, self.z_dir, self.v_dir, self.r_dir, 1e-10],
            device=self.device,
        )
        wp.launch(
            kernel=compute_t_direction_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.z_dir, self.r_dir, self.t_dir],
            device=self.device,
        )

        # step3: compute characteristic angles

        # compute contact point velocity angle (gamma)
        # see S7 eq.6 from https://www.pnas.org/doi/10.1073/pnas.2214017120
        wp.launch(
            kernel=compute_intrusion_angle,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.z_dir, self.v_dir, self.r_dir, self.contact_point_intrusion_angle],
            device=self.device,
        )

        # compute contact point tilt angle (beta)
        # see S7 eq.5 from https://www.pnas.org/doi/10.1073/pnas.2214017120
        wp.launch(
            kernel=compute_tilt_angle,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.z_dir, self.n_dir, self.r_dir, self.t_dir, self.contact_point_tilt_angle],
            device=self.device,
        )

        # compute twist angle (psi)
        # see S7 eq.7 from https://www.pnas.org/doi/10.1073/pnas.2214017120
        wp.launch(
            kernel=compute_twist_angle,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.z_dir, self.n_dir, self.r_dir, self.t_dir, self.n_rtz_dir, self.contact_point_twist_angle],
            device=self.device,
        )

        # step4: compute resistive force alpha (N/m^3) and point forces (N)
        # see eq.1 in https://www.pnas.org/doi/10.1073/pnas.2214017120

        wp.launch(
            kernel=compute_resistive_force,
            dim=(self.num_envs, self.num_bodies * self.num_contact_points),
            inputs=[
                self.contact_point_pos.reshape((self.num_envs, -1)),
                self.contact_point_lin_vel.reshape((self.num_envs, -1)),
                self.contact_point_lin_vel_prev.reshape((self.num_envs, -1)),
                self.contact_point_tilt_angle.reshape((self.num_envs, -1)),
                self.contact_point_intrusion_angle.reshape((self.num_envs, -1)),
                self.contact_point_twist_angle.reshape((self.num_envs, -1)),
                self.r_dir.reshape((self.num_envs, -1)),
                self.t_dir.reshape((self.num_envs, -1)),
                self.z_dir.reshape((self.num_envs, -1)),
                self.n_rtz_dir.reshape((self.num_envs, -1)),
                self.rho_c,
                self.mu_int,
                self.dynamic_friction_coef,
                self.coef_1,
                self.coef_2,
                self.coef_3,
                self.tau_r,
                self.c_r,
                self.collider.dA,
                self.num_contact_points,
                self.alpha_unfiltered,
                self.alpha_filtered,
                self.resitive_force,
            ],
        )

        # Step5: sum up contact point forces to get net contact force and torque on the body, and transform to body frame
        # clear contact wrench before atomic add
        wp.launch(
            kernel=zero_wrench,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                self.contact_force,
                self.contact_torque,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=compute_contact_wrench,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[
                self.body_pos,
                self.contact_point_pos,
                self.resitive_force.reshape((self.num_envs, self.num_bodies, self.num_contact_points)),
                self.contact_point_force,
                self.contact_point_torque,
                self.contact_force,
                self.contact_torque,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=transform_global_wrench_to_body,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                self.body_quat,
                self.contact_force,
                self.contact_torque,
                self.contact_force_b,
                self.contact_torque_b,
            ],
            device=self.device,
        )

        wp.copy(self.contact_point_lin_vel_prev, self.contact_point_lin_vel)

    """
    reset.
    """

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        wp.launch(
            kernel=reset,
            dim=(len(env_ids), self.num_bodies * self.num_contact_points),
            inputs=[
                env_ids,
                self.alpha_unfiltered,
                self.alpha_filtered,
                self.tau_r,
                self.contact_point_lin_vel_prev.reshape((self.num_envs, -1)),
            ],
            device=self.device,
        )


if __name__ == "__main__":
    num_envs = 4096
    num_bodies = 2
    device = "cuda"
    dt = 1 / 200
    material_cfg = Material3DRFTCfg(
        coef_1=[
            0.00212,
            -0.02320,
            -0.20890,
            -0.43083,
            -0.00259,
            0.48872,
            -0.00415,
            0.07204,
            -0.02750,
            -0.08772,
            0.01992,
            -0.45961,
            0.40799,
            -0.10107,
            -0.06576,
            0.05664,
            -0.09269,
            0.01892,
            0.01033,
            0.15120,
        ],
        coef_2=[
            -0.06796,
            -0.10941,
            0.04725,
            -0.06914,
            -0.05835,
            -0.65880,
            -0.11985,
            -0.25739,
            -0.26834,
            0.02692,
            -0.00736,
            0.63758,
            0.08997,
            0.21069,
            0.04748,
            0.20406,
            0.18519,
            0.04934,
            0.13527,
            -0.33207,
        ],
        coef_3=[
            -0.02634,
            -0.03436,
            0.45256,
            0.00835,
            0.02553,
            -1.31290,
            -0.05532,
            0.06790,
            -0.16404,
            0.02287,
            0.02927,
            0.95406,
            -0.00131,
            -0.11028,
            0.01487,
            -0.20770,
            0.10911,
            -0.04097,
            0.07881,
            -0.27519,
        ],
    )
    collider_cfg = PlaneColliderCfg(
        contact_edge_x=(-0.1, 0.1),
        contact_edge_y=(-0.05, 0.05),
        contact_edge_z=(-0.02, 0.0),
        resolution=(5, 5),
    )

    rft_3d = RFT_3D(
        num_envs=num_envs,
        num_bodies=num_bodies,
        device=device,
        dt=dt,
        material_cfg=material_cfg,
        collider_cfg=collider_cfg,
    )

    body_pos = torch.zeros((num_envs, num_bodies, 3), device=device)
    body_quat = torch.zeros((num_envs, num_bodies, 4), device=device)
    body_quat[..., 0] = 1.0
    body_lin_vel = torch.zeros((num_envs, num_bodies, 3), device=device)
    body_ang_vel = torch.zeros((num_envs, num_bodies, 3), device=device)
    rft_3d.update(body_pos, body_quat, body_lin_vel, body_ang_vel)
    rft_3d.reset()
