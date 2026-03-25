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
