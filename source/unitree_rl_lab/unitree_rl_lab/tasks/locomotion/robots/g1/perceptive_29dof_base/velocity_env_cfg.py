from importlib import import_module

from isaaclab.utils import configclass

_BASE_ENV_CFG_MODULE = import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg")
RobotEnvCfg = _BASE_ENV_CFG_MODULE.RobotEnvCfg
RobotPlayEnvCfg = _BASE_ENV_CFG_MODULE.RobotPlayEnvCfg


@configclass
class PerceptiveUnitreeBaseG1RobotEnvCfg(RobotEnvCfg):
    """Project-owned alias of the faithful Unitree G1-29dof locomotion env."""


@configclass
class PerceptiveUnitreeBaseG1RobotPlayEnvCfg(RobotPlayEnvCfg):
    """Project-owned alias of the faithful Unitree G1-29dof locomotion play env."""


@configclass
class _PerceptiveUnitreeBaseG1FixedCommandPlayEnvCfg(PerceptiveUnitreeBaseG1RobotPlayEnvCfg):
    """Diagnostic play env with fixed commands and no play-time pushes."""

    FIXED_LIN_VEL_X = 0.0
    FIXED_LIN_VEL_Y = 0.0
    FIXED_ANG_VEL_Z = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (self.FIXED_LIN_VEL_X, self.FIXED_LIN_VEL_X)
        self.commands.base_velocity.ranges.lin_vel_y = (self.FIXED_LIN_VEL_Y, self.FIXED_LIN_VEL_Y)
        self.commands.base_velocity.ranges.ang_vel_z = (self.FIXED_ANG_VEL_Z, self.FIXED_ANG_VEL_Z)
        self.events.push_robot = None


@configclass
class PerceptiveUnitreeBaseG1ZeroCmdPlayEnvCfg(_PerceptiveUnitreeBaseG1FixedCommandPlayEnvCfg):
    """Zero-command diagnostic play env."""


@configclass
class PerceptiveUnitreeBaseG1Fwd005PlayEnvCfg(_PerceptiveUnitreeBaseG1FixedCommandPlayEnvCfg):
    """Fixed forward 0.05 m/s diagnostic play env."""

    FIXED_LIN_VEL_X = 0.05


@configclass
class PerceptiveUnitreeBaseG1Fwd010PlayEnvCfg(_PerceptiveUnitreeBaseG1FixedCommandPlayEnvCfg):
    """Fixed forward 0.10 m/s diagnostic play env."""

    FIXED_LIN_VEL_X = 0.10
