# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import gymnasium as gym
import json
import math
import pathlib
import statistics
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import inspect
import os
import shutil
import torch
from copy import deepcopy
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner  # TODO: Consider printing the experiment name in the terminal.

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _collect_rsl_rl_metrics(runner) -> dict[str, float]:
    """Collect a compact set of scalar metrics from the current logger state."""

    metrics: dict[str, float] = {}
    log = runner.logger

    if len(log.rewbuffer) > 0:
        metrics["Train/mean_reward"] = float(statistics.mean(log.rewbuffer))
    if len(log.lenbuffer) > 0:
        metrics["Train/mean_episode_length"] = float(statistics.mean(log.lenbuffer))

    if log.ep_extras:
        for key in log.ep_extras[0]:
            values = []
            for ep_info in log.ep_extras:
                if key not in ep_info:
                    continue
                value = ep_info[key]
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor([value], device=log.device)
                if len(value.shape) == 0:
                    value = value.unsqueeze(0)
                values.append(value.to(log.device))
            if values:
                metric_name = key if "/" in key else "Episode/" + key
                metrics[metric_name] = float(torch.cat(values).mean().item())

    return metrics


def _best_checkpoint_score(metrics: dict[str, float]) -> tuple:
    """Score checkpoints using locomotion-relevant metrics rather than reward alone."""

    time_out = metrics.get("Episode_Termination/time_out")
    bad_orientation = metrics.get("Episode_Termination/bad_orientation")
    error_vel_xy = metrics.get("Metrics/base_velocity/error_vel_xy")
    error_vel_yaw = metrics.get("Metrics/base_velocity/error_vel_yaw")
    mean_len = metrics.get("Train/mean_episode_length")
    mean_reward = metrics.get("Train/mean_reward")

    return (
        (time_out if time_out is not None else -math.inf),
        -(bad_orientation if bad_orientation is not None else math.inf),
        -(error_vel_xy if error_vel_xy is not None else math.inf),
        -(error_vel_yaw if error_vel_yaw is not None else math.inf),
        (mean_len if mean_len is not None else -math.inf),
        (mean_reward if mean_reward is not None else -math.inf),
    )


def _enable_best_model_checkpointing(runner) -> None:
    """Patch the runner so every training run also maintains a best-model checkpoint."""

    runner._pl_best_metrics = None
    runner._pl_best_score = None
    runner._pl_best_source_checkpoint = None

    original_log = runner.logger.log
    original_save = runner.save

    def wrapped_log(*args, **kwargs):
        runner._pl_best_metrics = _collect_rsl_rl_metrics(runner)
        return original_log(*args, **kwargs)

    def wrapped_save(path: str, infos: dict | None = None):
        original_save(path, infos)

        metrics = runner._pl_best_metrics
        if not metrics:
            return

        score = _best_checkpoint_score(metrics)
        if runner._pl_best_score is None or score > runner._pl_best_score:
            runner._pl_best_score = score
            runner._pl_best_source_checkpoint = path

            best_path = os.path.join(runner.logger.log_dir, "best_model.pt")
            original_save(best_path, {"source_checkpoint": path, "metrics": metrics})

            best_meta_path = os.path.join(runner.logger.log_dir, "best_model_meta.json")
            with open(best_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source_checkpoint": path,
                        "best_model_path": best_path,
                        "iter": runner.current_learning_iteration,
                        "score": list(score),
                        "metrics": metrics,
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )

            print(f"[INFO] Updated best model checkpoint: {best_path}")

    runner.logger.log = wrapped_log
    runner.save = wrapped_save


def _sanitize_rsl_rl_cfg(cfg: dict) -> dict:
    """Remove deprecated runner/model keys before handing config to rsl-rl 5.x."""
    cfg = deepcopy(cfg)
    deprecated_model_keys = ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std")
    for model_key in ("actor", "critic"):
        model_cfg = cfg.get(model_key)
        if isinstance(model_cfg, dict):
            for deprecated_key in deprecated_model_keys:
                model_cfg.pop(deprecated_key, None)
    policy_cfg = cfg.get("policy")
    if isinstance(policy_cfg, dict) and not policy_cfg:
        cfg["policy"] = {}
    return cfg


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner_cfg_dict = _sanitize_rsl_rl_cfg(agent_cfg.to_dict())
    runner = OnPolicyRunner(env, runner_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    _enable_best_model_checkpointing(runner)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
