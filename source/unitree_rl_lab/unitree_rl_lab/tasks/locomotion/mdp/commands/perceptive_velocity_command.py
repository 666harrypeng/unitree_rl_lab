from __future__ import annotations

from dataclasses import MISSING

import torch
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .velocity_command import UniformLevelVelocityCommandCfg


class PerceptiveUniformVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command sampler with standing envs and per-axis non-zero floors."""

    cfg: "PerceptiveUniformLevelVelocityCommandCfg"

    def _sample_axis(
        self,
        count: int,
        low: float,
        high: float,
        floor: float,
        zero_prob: float,
    ) -> torch.Tensor:
        values = torch.empty(count, device=self.device)
        if count == 0:
            return values

        zero_mask = torch.rand(count, device=self.device) <= zero_prob
        values[zero_mask] = 0.0

        active_mask = ~zero_mask
        if not torch.any(active_mask):
            return values

        num_active = int(active_mask.sum().item())

        if floor <= 0.0:
            values[active_mask] = torch.empty(num_active, device=self.device).uniform_(low, high)
            return values

        if low >= 0.0:
            values[active_mask] = torch.empty(num_active, device=self.device).uniform_(max(low, floor), high)
            return values

        if high <= 0.0:
            values[active_mask] = torch.empty(num_active, device=self.device).uniform_(low, min(high, -floor))
            return values

        pos_extent = max(high - floor, 0.0)
        neg_extent = max(abs(low) - floor, 0.0)
        total_extent = pos_extent + neg_extent
        if total_extent <= 0.0:
            values[active_mask] = 0.0
            return values

        active_values = torch.empty(num_active, device=self.device)
        choose_positive = torch.rand(num_active, device=self.device) < (pos_extent / total_extent)

        if torch.any(choose_positive):
            active_values[choose_positive] = torch.empty(int(choose_positive.sum().item()), device=self.device).uniform_(
                floor, high
            )
        if torch.any(~choose_positive):
            active_values[~choose_positive] = -torch.empty(
                int((~choose_positive).sum().item()), device=self.device
            ).uniform_(floor, abs(low))

        values[active_mask] = active_values
        return values

    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        r = torch.empty(len(env_ids), device=self.device)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

        moving_mask = ~self.is_standing_env[env_ids]
        moving_env_ids = env_ids[moving_mask]
        if len(moving_env_ids) == 0:
            return

        floors = self.cfg.floors
        zero_prob = self.cfg.zero_prob

        self.vel_command_b[moving_env_ids, 0] = self._sample_axis(
            len(moving_env_ids),
            self.cfg.ranges.lin_vel_x[0],
            self.cfg.ranges.lin_vel_x[1],
            floors.lin_vel_x,
            zero_prob.lin_vel_x,
        )
        self.vel_command_b[moving_env_ids, 1] = self._sample_axis(
            len(moving_env_ids),
            self.cfg.ranges.lin_vel_y[0],
            self.cfg.ranges.lin_vel_y[1],
            floors.lin_vel_y,
            zero_prob.lin_vel_y,
        )
        self.vel_command_b[moving_env_ids, 2] = self._sample_axis(
            len(moving_env_ids),
            self.cfg.ranges.ang_vel_z[0],
            self.cfg.ranges.ang_vel_z[1],
            floors.ang_vel_z,
            zero_prob.ang_vel_z,
        )


@configclass
class PerceptiveUniformLevelVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    class_type: type = PerceptiveUniformVelocityCommand

    @configclass
    class Floors:
        lin_vel_x: float = 0.0
        lin_vel_y: float = 0.0
        ang_vel_z: float = 0.0

    @configclass
    class ZeroProb:
        lin_vel_x: float = 0.0
        lin_vel_y: float = 0.0
        ang_vel_z: float = 0.0

    floors: Floors = MISSING
    zero_prob: ZeroProb = MISSING
