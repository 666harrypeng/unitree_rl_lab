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
