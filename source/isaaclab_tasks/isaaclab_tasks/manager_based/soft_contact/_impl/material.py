# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass


"""
2D RFT material parameters.
"""


@configclass
class MaterialCfg:
    """
    Material configuration for soft contact model.

    A00 - D10: quasistatic RFT Fourier coefficients.
    lam, rho: dynamic RFT parameters.
    static_friction_coef, dynamic_friction_coef: friction coefficients.
    kf: tangential force model parameter.
    kh, beta_d, bh: horizontal stroke resistive force model parameters.
    """

    # quasistatic RFT fourier coefficients
    A00: float = MISSING  # type: ignore
    A10: float = MISSING  # type: ignore
    B11: float = MISSING  # type: ignore
    B01: float = MISSING  # type: ignore
    B_11: float = MISSING  # type: ignore
    C11: float = MISSING  # type: ignore
    C01: float = MISSING  # type: ignore
    C_11: float = MISSING  # type: ignore
    D10: float = MISSING  # type: ignore

    # dynamic RFT parameters
    lam: float = MISSING  # type: ignore
    rho: float = MISSING  # type: ignore

    # material properties
    static_friction_coef: float = MISSING  # type: ignore # TODO remove
    dynamic_friction_coef: float = MISSING  # type: ignore

    # coulomb friction tangential force model parameters
    kf: float = MISSING  # type: ignore

    # horizontal stroke resistive force model parameters
    kh: float = MISSING  # type: ignore
    beta_d: float = MISSING  # type: ignore
    bh: float = MISSING  # type: ignore


@configclass
class PoppySeedLPCfg(MaterialCfg):
    A00: float = 0.051
    A10: float = 0.047
    B11: float = 0.053
    B01: float = 0.083
    B_11: float = 0.020
    C11: float = -0.026
    C01: float = 0.057
    C_11: float = 0.0
    D10: float = 0.025

    # dynamic RFT parameters
    lam: float = 1.0
    rho: float = 638.0 * (1e-6)  # kg/mm^3 to kg/cm^3

    # material properties
    static_friction_coef: float = 1.0  # TODO remove
    dynamic_friction_coef: float = 0.5

    # tangential force model parameters
    kf: float = 10.0

    # horizontal stroke resistive force model parameters
    kh: float = 50.0
    beta_d: float = 0.5
    bh: float = 1.0


@configclass
class PoppySeedCPCfg(MaterialCfg):
    A00: float = 0.094
    A10: float = 0.092
    B11: float = 0.092
    B01: float = 0.151
    B_11: float = 0.035
    C11: float = -0.039
    C01: float = 0.086
    C_11: float = 0.018
    D10: float = 0.046

    lam: float = 1.0
    rho: float = 638.0 * (1e-6)  # kg/mm^3 to kg/cm^3

    static_friction_coef: float = 1.0  # TODO remove
    dynamic_friction_coef: float = 0.5

    kf: float = 10.0

    # horizontal stroke resistive force model parameters
    kh: float = 50.0
    beta_d: float = 0.5
    bh: float = 1.0


"""
3D RFT material parameters.
"""


@configclass
class Material3DRFTCfg:
    """
    Material parameters for 3D RFT soft contact model.
    See https://www.pnas.org/doi/10.1073/pnas.2214017120 supplementary material S3.

    c1^k, c2^k, c3^k are the coefficients of 3rd order polynomial that computes the resistive force per volume.
    static_friction_coef: static friction coefficient
    dynamic_friction_coef: dynamic friction coefficient
    rho_c: critical media density (effective media density = packing fraction * grain density)
    mu_int: media internal friction coefficient
    """

    # c1^k
    coef_1: list[float] = [
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
    ]
    # c2^k
    coef_2: list[float] = [
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
    ]
    # c3^k
    coef_3: list[float] = [
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
    ]

    # 3D RFT media specific properties
    static_friction_coef: float = 1.0  # TODO remove
    dynamic_friction_coef: float = 0.3
    mu_int: float = 0.3  # media internal friction coefficient
    rho_c: float = 3000.0  # critical media density (effective media density = packing fraction * grain density)
