# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from .soft_contact_model import IntruderGeometryCfg
from . import physics_callback_actions

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR


BLUE_ARROW_Z_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow": sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/prop/arrow_z.usd",
            scale=(0.1, 0.1, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        )
    }
)


@configclass
class PhysicsCallbackActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = physics_callback_actions.PhysicsCallbackAction
    body_names: list[str] = MISSING # type: ignore
    """List of joint names or regex expressions that the action will be mapped to."""
    backend: Literal["2D", "3D"] = "3D"
    """The RFT backend to use. Options are '2D' or '3D'."""
    disable: bool = False
    """Whether to disable this action term."""
    enable_ema_filter: bool = True
    """Whether to enable an exponential moving average filter on the input actions."""
    contact_threshold: float = 10.0
    """Threshold for contact detection (N)."""
    intruder_geometry_cfg: IntruderGeometryCfg = IntruderGeometryCfg()

    contact_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_Z_MARKER_CFG.replace( # type: ignore
        prim_path="/Visuals/Contact/grf", 
    )
    contact_vis_max_force: float = 100.0
    contact_visualizer_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3) # type: ignore