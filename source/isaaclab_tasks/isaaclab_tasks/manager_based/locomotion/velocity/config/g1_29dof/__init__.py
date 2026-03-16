# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# gym.register(
#     id="Isaac-Velocity-Rough-G1-29dof-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
#     },
# )


# gym.register(
#     id="Isaac-Velocity-Rough-G1-29dof-Play-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
#     },
# )


gym.register(
    id="Isaac-Velocity-Flat-G1-29dof-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "rsl_rl_recurrent_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerRecurrentCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:G1FlatDistillationRunnerCfg"
        ),
        "holosoma_agent_cfg_entry_point": f"{agents.__name__}.holosoma_agent_fastsac_cfg:G1HolosomaFastSACAgentCfg",
        # "holosoma_agent_cfg_entry_point": f"{agents.__name__}.holosoma_agent_ppo_cfg:G1HolosomaPPOAgentCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-G1-29dof-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "rsl_rl_recurrent_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerRecurrentCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:G1FlatDistillationRunnerCfg"
        ),
        "holosoma_agent_cfg_entry_point": f"{agents.__name__}.holosoma_agent_fastsac_cfg:G1HolosomaFastSACAgentCfg",
        # "holosoma_agent_cfg_entry_point": f"{agents.__name__}.holosoma_agent_ppo_cfg:G1HolosomaPPOAgentCfg",
    },
)
