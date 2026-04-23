# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPAEModelCfg,
    RslRlMLPModelCfg,
    RslRlMLPVAEModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRNNEncoderModelCfg,
    RslRlTCNAttentionModelCfg,
    RslRlTCNModelCfg,
)


@configclass
class RslRlPpoAEAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the PPO algorithm with AutoEncoder."""

    class_name: str = "PPOAE"
    decoder_loss_coef: float = 1.0
    loss_type: Literal["mse", "huber"] = "mse"


@configclass
class RslRlPpoVAEAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the PPO algorithm with Variational AutoEncoder."""

    class_name: str = "PPOVAE"
    decoder_loss_coef: float = 1.0
    loss_type: Literal["mse", "huber"] = "mse"
    kl_loss_coef: float = 0.2
    kl_clip: float = 0.0


@configclass
class RslRlPpoDistillationAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the PPO algorithm with Variational AutoEncoder."""

    class_name: str = "PPODistillation"
    imitation_loss_coef: float = 1.0
    encoder_loss_coef: float = 1.0
    decoder_loss_coef: float = 1.0
    loss_type: Literal["mse", "huber"] = "mse"
    ppo_learning_start: int = 0


@configclass
class G1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30_000
    save_interval = 500
    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic", "dynamics_privileged", "terrain_privileged"],
        "encoder": ["proprioceptive_history", "dynamics_privileged", "terrain_privileged"],
        "decoder": ["terrain_privileged"],
    }

    # VAE
    actor = RslRlMLPVAEModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        encoder_obs_set="encoder",
        encoder_output_dim=64,
        encoder_hidden_dims=[256, 128],
        encoder_activation="elu",
        encoder_obs_normalization=False,
        decoder_obs_set="decoder",
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoVAEAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        decoder_loss_coef=1.0,
        loss_type="mse",
        # kl_loss_coef=0.2,
        kl_loss_coef=0.01,
        # kl_loss_coef=0.005,
        kl_clip=0.0,
    )

    logger = "wandb"
    wandb_project = "g1_29dof_soft_vae_teacher"
    experiment_name = "g1_29dof_soft_vae_teacher"


@configclass
class G1PPORunnerCfgFinetune(G1PPORunnerCfg):
    max_iterations = 15_000
    logger = "wandb"
    wandb_project = "g1_29dof_soft_encoder_decoder_teacher_finetune"
    experiment_name = "g1_29dof_soft_encoder_decoder_teacher_finetune"


@configclass
class G1DistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30_000
    save_interval = 500
    obs_groups = {
        "student": ["policy"],
        "teacher": ["policy"],
        # "encoder": ["proprioceptive_history", "dynamics_privileged", "terrain_privileged"],  # RNN
        "encoder": ["policy_history", "dynamics_privileged", "terrain_privileged"],  # TCN
        "decoder": ["terrain_privileged"],
        "proprioceptive_history": ["proprioceptive_history"],
    }
    teacher = RslRlMLPVAEModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        encoder_obs_set="encoder",
        encoder_output_dim=64,
        encoder_hidden_dims=[256, 128],
        encoder_activation="elu",
        encoder_obs_normalization=False,
        decoder_obs_set="decoder",
    )
    # TCN student
    student = RslRlTCNModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        encoder_obs_set="proprioceptive_history",
        encoder_output_dim=64,
        encoder_hidden_dims=[256, 128],
        encoder_activation="elu",
    )
    # GRU student
    # student = RslRlRNNEncoderModelCfg(
    #     hidden_dims=[512, 256, 128],
    #     activation="elu",
    #     obs_normalization=False,
    #     distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
    #     encoder_obs_set="proprioceptive_history",
    #     encoder_obs_normalization=False,
    #     rnn_type="gru",
    #     rnn_hidden_dim=64,
    #     rnn_num_layers=2,
    # )

    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )

    # pure distillation
    # algorithm = RslRlDistillationAlgorithmCfg(
    #     num_learning_epochs=2,
    #     learning_rate=1.0e-3,
    #     gradient_length=15,
    # )

    # distillation with PPO loss
    algorithm = RslRlPpoDistillationAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        imitation_loss_coef=1.0,
        encoder_loss_coef=1.0,
        decoder_loss_coef=1.0,
        loss_type="mse",
        ppo_learning_start=1000,
    )

    logger = "wandb"
    wandb_project = "g1_29dof_soft_vae_student_tcn"
    experiment_name = "g1_29dof_soft_vae_student_tcn"

    # wandb_project = "g1_29dof_soft_vae_student_gru"
    # experiment_name = "g1_29dof_soft_vae_student_gru"
