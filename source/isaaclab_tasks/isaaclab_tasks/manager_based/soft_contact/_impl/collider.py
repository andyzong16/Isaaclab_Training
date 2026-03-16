# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

import torch
import warp as wp

from isaaclab.utils import configclass

from .kernels import (
    box_contact_points,
    expand_1d_vec3_to_3d_vec3,
    plane_contact_points,
    sphere_contact_points,
)

"""
Collider configuration classes.
"""


@configclass
class ColliderCfg:
    """Base collider configuration. Subclass per shape."""

    def __post_init__(self):
        print("ColliderCfg is not yet implemented.")


@configclass
class PlaneColliderCfg(ColliderCfg):
    """
    Single planar face collider (e.g. foot sole).

    Contact points are sampled as an (nx, ny) grid on the XY plane
    at z = contact_edge_z[0] (bottom of the geometry).
    Surface normal points in -Z direction in body frame.
    """

    contact_edge_x: tuple[float, float] = MISSING  # type: ignore
    """(min, max) bounds in body-frame X (m)."""
    contact_edge_y: tuple[float, float] = MISSING  # type: ignore
    """(min, max) bounds in body-frame Y (m)."""
    contact_edge_z: tuple[float, float] = MISSING  # type: ignore
    """(min, max) bounds in body-frame Z (m). Depth = z[1] - z[0]."""
    resolution: tuple[int, int] = (5, 5)
    """(nx, ny) grid resolution on the face."""


@configclass
class BoxColliderCfg(ColliderCfg):
    """
    6-face box collider.

    Contact points are sampled as an (nx, ny) grid on each of the 6 faces.
    Total contact points = nx * ny * 6.
    """

    contact_edge_x: tuple[float, float] = MISSING  # type: ignore
    """(min, max) bounds in body-frame X (m)."""
    contact_edge_y: tuple[float, float] = MISSING  # type: ignore
    """(min, max) bounds in body-frame Y (m)."""
    contact_edge_z: tuple[float, float] = MISSING  # type: ignore
    """(min, max) bounds in body-frame Z (m)."""
    resolution: tuple[int, int] = (5, 5)
    """(nx, ny) grid resolution per face."""


@configclass
class SphereColliderCfg(ColliderCfg):
    """
    Sphere collider.

    Contact points are sampled using a spherical coordinate grid.
    Total contact points = n_theta * n_phi.
    """

    radius: float = MISSING  # type: ignore
    """Sphere radius (m)."""
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Center of the sphere in body frame (m)."""
    resolution: tuple[int, int] = (8, 8)
    """(n_theta, n_phi) angular grid resolution."""


# Backward compatibility alias
IntruderGeometryCfg = PlaneColliderCfg


"""
Collider runtime class.
"""


class Collider:
    """
    Generates contact point positions and surface normals in body frame.

    Each collider shape dispatches to its own warp kernel for building
    the contact point grid. The solver reads the outputs via properties
    and does not need to know the collider shape.
    """

    def __init__(
        self,
        cfg: ColliderCfg,
        num_envs: int,
        num_bodies: int,
        device: torch.device | str,
    ) -> None:
        self._cfg = cfg
        self._num_envs = num_envs
        self._num_bodies = num_bodies
        self._device = device

        if isinstance(cfg, PlaneColliderCfg):
            self._setup_plane(cfg)
        elif isinstance(cfg, BoxColliderCfg):
            self._setup_box(cfg)
        elif isinstance(cfg, SphereColliderCfg):
            self._setup_sphere(cfg)
        else:
            raise ValueError(f"Unsupported collider config type: {type(cfg)}")

    # ------------------------------------------------------------------
    # Properties (read by the solver)
    # ------------------------------------------------------------------

    @property
    def contact_point_local(self) -> wp.array:
        """Body-frame contact positions. Shape: (N, B, C) dtype=vec3f."""
        return self._contact_point_local

    @property
    def normal_dir_local(self) -> wp.array:
        """Body-frame surface normals. Shape: (N, B, C) dtype=vec3f."""
        return self._normal_dir_local

    @property
    def num_contact_points(self) -> int:
        """Total number of contact points per body."""
        return self._num_contact_points

    @property
    def surface_area(self) -> float:
        """Total contact surface area (m^2)."""
        return self._surface_area

    @property
    def dA(self) -> wp.array:
        """Per-contact-point area element. Shape: (C,) dtype=float32."""
        return self._dA

    # ------------------------------------------------------------------
    # Plane collider setup
    # ------------------------------------------------------------------

    def _setup_plane(self, cfg: PlaneColliderCfg) -> None:
        """
        Build contact point grid for a single planar face.

        Generates an (nx, ny) grid on the XY rectangle defined by
        contact_edge_x/y at z = -foot_depth. Normal = (0, 0, -1).
        """
        nx, ny = cfg.resolution
        self._num_contact_points = nx * ny

        lx = cfg.contact_edge_x[1] - cfg.contact_edge_x[0]
        ly = cfg.contact_edge_y[1] - cfg.contact_edge_y[0]
        self._surface_area = lx * ly

        foot_depth = cfg.contact_edge_z[1] - cfg.contact_edge_z[0]

        # Uniform dA for a single face
        dA_val = self._surface_area / self._num_contact_points
        self._dA = wp.full(
            (self._num_contact_points,), dA_val, dtype=wp.float32, device=self._device
        )

        # Allocate warp arrays
        self._contact_point_local = wp.zeros(
            (self._num_envs, self._num_bodies, self._num_contact_points),
            dtype=wp.vec3f,
            device=self._device,
        )
        self._normal_dir_local = wp.zeros(
            (self._num_envs, self._num_bodies, self._num_contact_points),
            dtype=wp.vec3f,
            device=self._device,
        )

        # Flat 1D arrays for the kernel to write into
        contact_pts_flat = wp.zeros(
            (self._num_contact_points,), dtype=wp.vec3f, device=self._device
        )
        contact_nrm_flat = wp.zeros(
            (self._num_contact_points,), dtype=wp.vec3f, device=self._device
        )

        edge_x = wp.vec2f(
            (wp.float32(cfg.contact_edge_x[0]), wp.float32(cfg.contact_edge_x[1]))
        )
        edge_y = wp.vec2f(
            (wp.float32(cfg.contact_edge_y[0]), wp.float32(cfg.contact_edge_y[1]))
        )

        wp.launch(
            kernel=plane_contact_points,
            dim=(ny, nx),
            inputs=[edge_x, edge_y, -foot_depth, nx, ny, contact_pts_flat, contact_nrm_flat],
            device=self._device,
        )

        # Expand to (num_envs, num_bodies, num_contact_points)
        wp.launch(
            kernel=expand_1d_vec3_to_3d_vec3,
            dim=(self._num_envs, self._num_bodies, self._num_contact_points),
            inputs=[contact_pts_flat, self._contact_point_local],
            device=self._device,
        )
        wp.launch(
            kernel=expand_1d_vec3_to_3d_vec3,
            dim=(self._num_envs, self._num_bodies, self._num_contact_points),
            inputs=[contact_nrm_flat, self._normal_dir_local],
            device=self._device,
        )

    # ------------------------------------------------------------------
    # Box collider setup
    # ------------------------------------------------------------------

    def _setup_box(self, cfg: BoxColliderCfg) -> None:
        """
        Build contact point grids for all 6 faces of a box.

        Each face has (nx, ny) points. Total = 6 * nx * ny.
        Normals point outward from each face.
        """
        nx, ny = cfg.resolution
        self._num_contact_points = 6 * nx * ny

        lx = cfg.contact_edge_x[1] - cfg.contact_edge_x[0]
        ly = cfg.contact_edge_y[1] - cfg.contact_edge_y[0]
        lz = cfg.contact_edge_z[1] - cfg.contact_edge_z[0]
        self._surface_area = 2.0 * (lx * ly + ly * lz + lz * lx)

        # Per-face areas
        face_areas = [
            lx * ly,   # -Z
            lx * ly,   # +Z
            lx * lz,   # -Y
            lx * lz,   # +Y
            ly * lz,   # -X
            ly * lz,   # +X
        ]
        dA_list = []
        pts_per_face = nx * ny
        for fa in face_areas:
            dA_list.extend([fa / pts_per_face] * pts_per_face)
        self._dA = wp.array(dA_list, dtype=wp.float32, device=self._device)

        # Allocate expanded warp arrays
        self._contact_point_local = wp.zeros(
            (self._num_envs, self._num_bodies, self._num_contact_points),
            dtype=wp.vec3f,
            device=self._device,
        )
        self._normal_dir_local = wp.zeros(
            (self._num_envs, self._num_bodies, self._num_contact_points),
            dtype=wp.vec3f,
            device=self._device,
        )

        # Flat 1D arrays for the kernel to write into
        contact_pts_flat = wp.zeros(
            (self._num_contact_points,), dtype=wp.vec3f, device=self._device
        )
        contact_nrm_flat = wp.zeros(
            (self._num_contact_points,), dtype=wp.vec3f, device=self._device
        )

        edge_x = wp.vec2f(
            (wp.float32(cfg.contact_edge_x[0]), wp.float32(cfg.contact_edge_x[1]))
        )
        edge_y = wp.vec2f(
            (wp.float32(cfg.contact_edge_y[0]), wp.float32(cfg.contact_edge_y[1]))
        )
        edge_z = wp.vec2f(
            (wp.float32(cfg.contact_edge_z[0]), wp.float32(cfg.contact_edge_z[1]))
        )

        wp.launch(
            kernel=box_contact_points,
            dim=(6, ny, nx),
            inputs=[edge_x, edge_y, edge_z, nx, ny, contact_pts_flat, contact_nrm_flat],
            device=self._device,
        )

        # Expand to (num_envs, num_bodies, num_contact_points)
        wp.launch(
            kernel=expand_1d_vec3_to_3d_vec3,
            dim=(self._num_envs, self._num_bodies, self._num_contact_points),
            inputs=[contact_pts_flat, self._contact_point_local],
            device=self._device,
        )
        wp.launch(
            kernel=expand_1d_vec3_to_3d_vec3,
            dim=(self._num_envs, self._num_bodies, self._num_contact_points),
            inputs=[contact_nrm_flat, self._normal_dir_local],
            device=self._device,
        )

    # ------------------------------------------------------------------
    # Sphere collider setup
    # ------------------------------------------------------------------

    def _setup_sphere(self, cfg: SphereColliderCfg) -> None:
        """
        Build contact point grid for a sphere using spherical coordinates.

        Points are sampled on a (n_theta, n_phi) grid. Normals point
        radially outward.
        """
        n_theta, n_phi = cfg.resolution
        self._num_contact_points = n_theta * n_phi
        self._surface_area = 4.0 * math.pi * cfg.radius ** 2

        # Uniform dA for sphere
        dA_val = self._surface_area / self._num_contact_points
        self._dA = wp.full(
            (self._num_contact_points,), dA_val, dtype=wp.float32, device=self._device
        )

        # Allocate expanded warp arrays
        self._contact_point_local = wp.zeros(
            (self._num_envs, self._num_bodies, self._num_contact_points),
            dtype=wp.vec3f,
            device=self._device,
        )
        self._normal_dir_local = wp.zeros(
            (self._num_envs, self._num_bodies, self._num_contact_points),
            dtype=wp.vec3f,
            device=self._device,
        )

        # Flat 1D arrays for the kernel
        contact_pts_flat = wp.zeros(
            (self._num_contact_points,), dtype=wp.vec3f, device=self._device
        )
        contact_nrm_flat = wp.zeros(
            (self._num_contact_points,), dtype=wp.vec3f, device=self._device
        )

        center = wp.vec3f(
            wp.float32(cfg.center[0]),
            wp.float32(cfg.center[1]),
            wp.float32(cfg.center[2]),
        )

        wp.launch(
            kernel=sphere_contact_points,
            dim=(n_theta, n_phi),
            inputs=[
                wp.float32(cfg.radius),
                center,
                n_theta,
                n_phi,
                contact_pts_flat,
                contact_nrm_flat,
            ],
            device=self._device,
        )

        # Expand to (num_envs, num_bodies, num_contact_points)
        wp.launch(
            kernel=expand_1d_vec3_to_3d_vec3,
            dim=(self._num_envs, self._num_bodies, self._num_contact_points),
            inputs=[contact_pts_flat, self._contact_point_local],
            device=self._device,
        )
        wp.launch(
            kernel=expand_1d_vec3_to_3d_vec3,
            dim=(self._num_envs, self._num_bodies, self._num_contact_points),
            inputs=[contact_nrm_flat, self._normal_dir_local],
            device=self._device,
        )