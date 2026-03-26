import gymnasium as gym


gym.register(
    id="Perceptive-UnitreeBase-G1-29dof-Velocity-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:PerceptiveUnitreeBaseG1RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:PerceptiveUnitreeBaseG1RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:PerceptiveUnitreeBaseG1PPORunnerCfg",
    },
)

gym.register(
    id="Perceptive-UnitreeBase-G1-29dof-Velocity-ZeroCmd-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:PerceptiveUnitreeBaseG1ZeroCmdPlayEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:PerceptiveUnitreeBaseG1ZeroCmdPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:PerceptiveUnitreeBaseG1PPORunnerCfg",
    },
)

gym.register(
    id="Perceptive-UnitreeBase-G1-29dof-Velocity-Fwd005-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:PerceptiveUnitreeBaseG1Fwd005PlayEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:PerceptiveUnitreeBaseG1Fwd005PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:PerceptiveUnitreeBaseG1PPORunnerCfg",
    },
)

gym.register(
    id="Perceptive-UnitreeBase-G1-29dof-Velocity-Fwd010-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:PerceptiveUnitreeBaseG1Fwd010PlayEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:PerceptiveUnitreeBaseG1Fwd010PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:PerceptiveUnitreeBaseG1PPORunnerCfg",
    },
)
