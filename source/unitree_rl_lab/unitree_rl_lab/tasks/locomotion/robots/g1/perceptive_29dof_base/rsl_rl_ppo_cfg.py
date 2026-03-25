from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class PerceptiveUnitreeBaseG1PPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "perceptive_unitree_base_g1_29dof_velocity_v0"
