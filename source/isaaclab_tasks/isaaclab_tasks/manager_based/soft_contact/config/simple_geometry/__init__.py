import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Soft-Contact-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.soft_contact_env_cfg:SoftcontactEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)