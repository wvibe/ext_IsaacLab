# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP components for the Go2 jump-augmented velocity task.

Extends the SE(2) velocity command with a jump command: environments receive either a
planar walking command or a pure-vertical jump command (mutually exclusive). A jump is
commanded as an apex *height gain* [m] rather than a velocity: height is the integral
of vertical velocity and directly defines a jump, so it cannot be satisfied by merely
standing up quickly (the leg-extension limit of the Go2 is ~0.10 m, well below the
commanded range).

Heights are measured relative to the local ground (height scanner), so the task remains
well-posed on rough terrain: climbing stairs raises the base and the ground together and
is not mistaken for a jump, while a jump raises only the base.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class JumpVelocityCommand(UniformVelocityCommand):
    """Velocity command generator with a windowed, height-based jump command.

    The command is 5-dimensional: ``(lin_vel_x, lin_vel_y, ang_vel_z, jump_height, jump_residual)``.
    The first three components keep the same order as :class:`UniformVelocityCommand` so all
    existing reward and observation terms remain compatible.

    Commands are mutually exclusive: with probability :attr:`JumpVelocityCommandCfg.rel_jump_envs`
    an environment receives a pure-vertical jump command (planar components zeroed, heading
    control disabled), otherwise a regular planar command with zero jump component.

    Heights are measured as the base height above the local ground (:attr:`base_rel_height`).
    When :attr:`JumpVelocityCommandCfg.height_scanner_name` is set, the ground is estimated as
    the median of the height-scanner ray hits, which makes the measurement terrain-relative:
    walking up stairs raises base and ground together and does not register as a jump. Without
    a scanner (flat terrain), the environment origin is used as the ground.

    Jump semantics are impulsive rather than sustained: time is divided into windows of
    :attr:`JumpVelocityCommandCfg.jump_period` seconds. Within each window, the generator
    records the peak relative base height [m] (:attr:`jump_peak_h`). The tracking error
    (:attr:`jump_error`) compares this peak against the target apex
    ``ref_height + jump_height``, so a single takeoff that matches the commanded apex yields
    full reward for the rest of the window. Peak tracking is armed only once the base has been
    near the ground within the window, so the spawn-drop transient and a jump straddling a
    window boundary are not credited. The peak is parked at ground level (zero) at window
    start so that all base motion, including below the standing height, shapes the reward:
    parking at the standing height instead was observed to create a zero-gradient pocket in
    which a crouched robot could explore without the reward ever responding.

    The reference height is the *constant* nominal standing height
    (:attr:`JumpVelocityCommandCfg.ref_height`), not the height at window start. This is
    deliberate: a state-dependent reference could be gamed by crouching before the window
    boundary and "achieving" the target by simply standing back up.

    The last command element is the signed jump residual ``target apex - jump_peak_h`` [m].
    It makes the task Markovian: positive means "a jump is still owed", zero means "achieved",
    negative means "overshot" (the penalty is locked in for the rest of the window, since the
    peak can only increase).

    Success accounting: each *completed* jump-commanded window is judged delivered when the
    absolute jump error at its end is below :attr:`JumpVelocityCommandCfg.jump_success_threshold`.
    The delivered fraction is logged as ``Metrics/jump_success_rate`` (jump windows only, so it
    is not diluted by walking environments). The unified ``Metrics/success_rate`` is overridden
    to require both the planar thresholds of the base class and the delivery of every completed
    jump window, so jump environments no longer trivially pass it with a zero planar command.
    """

    cfg: JumpVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: JumpVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # buffer for the jump command: commanded apex height gain above the reference height [m]
        self.jump_height_command = torch.zeros(self.num_envs, 1, device=self.device)
        # windowed peak of the base height above the local ground [m]
        self._jump_peak_h = torch.zeros(self.num_envs, device=self.device)
        # peak tracking is armed only once the base has been near the ground within the window;
        # this masks the spawn-drop transient and jumps straddling a window boundary
        self._jump_armed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # time left in the current jump window [s]
        self._jump_window_left = torch.zeros(self.num_envs, device=self.device)
        # per-episode tracking-error accounting for the jump height
        self.metrics["error_jump_height"] = torch.zeros(self.num_envs, device=self.device)
        self._error_jump_sum = torch.zeros(self.num_envs, device=self.device)
        # per-episode jump-window bookkeeping: completed jump-commanded windows and deliveries
        self._jump_window_count = torch.zeros(self.num_envs, device=self.device)
        self._jump_window_success = torch.zeros(self.num_envs, device=self.device)
        # annotate the extra command elements for export
        self.cfg.element_names = ["lin_vel_x", "lin_vel_y", "ang_vel_z", "jump_height", "jump_residual"]

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 5).

        Components: linear x-y velocity [m/s], angular z velocity [rad/s], commanded apex
        height gain [m], signed jump residual [m].
        """
        residual = self._target_apex_h - self._jump_peak_h
        return torch.cat([self.vel_command_b, self.jump_height_command, residual.unsqueeze(1)], dim=1)

    @property
    def base_rel_height(self) -> torch.Tensor:
        """Base height above the local ground [m], shape (num_envs,).

        Uses the median height-scanner ray hit as the ground estimate when a scanner is
        configured; otherwise falls back to the environment origin (exact on flat terrain).
        Non-hitting rays are replaced by the nominal ground under the base before the median.
        """
        base_h = self.robot.data.root_pos_w.torch[:, 2]
        if self.cfg.height_scanner_name is not None:
            sensor = self._env.scene.sensors[self.cfg.height_scanner_name]
            hits_z = sensor.data.ray_hits_w.torch[..., 2]
            fallback = (base_h - self.cfg.ref_height).unsqueeze(1)
            hits_z = torch.where(torch.isfinite(hits_z), hits_z, fallback)
            ground_h = hits_z.median(dim=1).values
            return base_h - ground_h
        return base_h - self._env.scene.env_origins[:, 2]

    @property
    def jump_peak_h(self) -> torch.Tensor:
        """Peak base height above the local ground [m] reached in the current jump window."""
        return self._jump_peak_h

    @property
    def jump_error(self) -> torch.Tensor:
        """Signed jump tracking error [m], shape (num_envs,).

        For jump environments this is ``jump_peak_h - target apex`` (symmetric: undershoot
        and overshoot are both errors). For walking environments (zero jump command) the
        peak is only penalized beyond :attr:`JumpVelocityCommandCfg.walk_height_deadband`
        above the reference height, which exempts gait oscillation and posture changes.
        """
        is_jump = self.jump_height_command[:, 0] > 1.0e-6
        jump_err = self._jump_peak_h - self._target_apex_h
        walk_err = (self._jump_peak_h - (self.cfg.ref_height + self.cfg.walk_height_deadband)).clamp_min(0.0)
        return torch.where(is_jump, jump_err, walk_err)

    @property
    def _target_apex_h(self) -> torch.Tensor:
        """Target apex base height above the local ground [m], per environment."""
        return self.cfg.ref_height + self.jump_height_command[:, 0]

    """
    Operations
    """

    def compute(self, dt: float):
        # advance the jump windows; expired windows restart the peak so a new jump is owed
        self._jump_window_left -= dt
        expired = self._jump_window_left <= 0.0
        if expired.any():
            # judge completed jump-commanded windows before the peak is restarted
            jump_window = expired & (self.jump_height_command[:, 0] > 1.0e-6)
            self._jump_window_count[jump_window] += 1.0
            delivered = self.jump_error.abs() < self.cfg.jump_success_threshold
            self._jump_window_success[jump_window & delivered] += 1.0
            self._jump_window_left[expired] = self.cfg.jump_period
            self._jump_armed[expired] = False
            # park the peak at ground level, NOT at ref_height: parking higher creates a
            # zero-gradient pocket where a crouched robot can move without the reward ever
            # seeing it (observed failure mode on rough terrain: crouch-and-wait lock-in)
            self._jump_peak_h[expired] = 0.0
        super().compute(dt)
        # arm the peak tracking once the base is near the ground: the spawn drop and a jump
        # straddling a window boundary must not be credited as achieved height
        rel_h = self.base_rel_height
        self._jump_armed |= rel_h <= self.cfg.ref_height + self.cfg.arm_height_threshold
        # record the peak relative base height for armed environments
        self._jump_peak_h = torch.where(self._jump_armed, torch.maximum(self._jump_peak_h, rel_h), self._jump_peak_h)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # finalize the jump-error metric before the base class logs and zeros the metrics
        ids = slice(None) if env_ids is None else env_ids
        denom = self._step_count[ids].clamp_min(1.0)
        self.metrics["error_jump_height"][ids] = self._error_jump_sum[ids] / denom
        # per-env binary jump compliance: every completed jump window was delivered
        # (vacuously true for environments that completed no jump window)
        jump_ok = self._jump_window_success[ids] >= self._jump_window_count[ids]
        # planar success with the same formula as the base class (its buffers are still intact)
        mean_error_xy = self._error_xy_sum[ids] / denom
        mean_error_yaw = self._error_yaw_sum[ids] / denom
        planar_ok = (mean_error_xy < self.cfg.vel_xy_success_threshold) & (
            mean_error_yaw < self.cfg.vel_yaw_success_threshold
        )
        extras = super().reset(env_ids)
        # override the unified success rate: planar thresholds AND jump-window delivery
        self._env.extras["log"]["Metrics/success_rate"] = (planar_ok & jump_ok).float().mean().item()
        # window-weighted jump success over the resetting environments (jump windows only)
        total_windows = self._jump_window_count[ids].sum()
        if total_windows > 0:
            self._env.extras["log"]["Metrics/jump_success_rate"] = (
                self._jump_window_success[ids].sum() / total_windows
            ).item()
        self._error_jump_sum[ids] = 0.0
        self._jump_window_count[ids] = 0.0
        self._jump_window_success[ids] = 0.0
        return extras

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        super()._update_metrics()
        self._error_jump_sum += torch.abs(self.jump_error)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # mutually exclusive commands: an environment either walks (planar) or jumps (pure vertical)
        is_jump = torch.empty(len(env_ids), device=self.device).uniform_(0.0, 1.0) <= self.cfg.rel_jump_envs
        jump_height = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges.jump_height)
        self.jump_height_command[env_ids, 0] = jump_height * is_jump.float()
        # jump environments: zero the planar command and disable heading control
        jump_ids = torch.as_tensor(env_ids, device=self.device)[is_jump]
        self.vel_command_b[jump_ids] = 0.0
        self.is_heading_env[jump_ids] = False
        # start a fresh, disarmed jump window (peak parked at ground level, see compute())
        self._jump_window_left[env_ids] = self.cfg.jump_period
        self._jump_armed[env_ids] = False
        self._jump_peak_h[env_ids] = 0.0

    def _update_command(self):
        super()._update_command()
        # standing environments receive a fully zero command, including the jump component
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.jump_height_command[standing_env_ids, :] = 0.0

    def _debug_vis_callback(self, event):
        # goal arrow: planar command rotated by base yaw only (so body pitch during a jump
        # does not tilt a purely planar command), plus the *remaining* jump obligation as the
        # vertical component -- it points up until the jump is achieved, then collapses.
        # current arrow: measured world-frame velocity (what the robot actually does).
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.torch.clone()
        base_pos_w[:, 2] += 0.5
        # remaining jump obligation [m]; gated to jump environments so walking dogs never
        # show a spurious up-arrow when their base dips below the reference height
        is_jump = self.jump_height_command[:, 0] > 1.0e-6
        residual_up = (self._target_apex_h - self._jump_peak_h).clamp_min(0.0) * is_jump.float()
        # rotate the planar command into the world frame with the yaw component only
        base_yaw_quat = math_utils.yaw_quat(self.robot.data.root_quat_w.torch)
        planar_cmd = torch.cat([self.vel_command_b[:, :2], torch.zeros_like(residual_up).unsqueeze(1)], dim=1)
        goal_vec_w = math_utils.quat_apply(base_yaw_quat, planar_cmd)
        goal_vec_w[:, 2] = residual_up
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_world_vector_to_arrow(goal_vec_w)
        vel_arrow_scale, vel_arrow_quat = self._resolve_world_vector_to_arrow(self.robot.data.root_lin_vel_w.torch)
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_world_vector_to_arrow(self, vec_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts a world-frame 3D vector to arrow scale and orientation (with pitch).

        Near-zero vectors shrink the whole arrow instead of only its length, so a completed
        jump command vanishes rather than rendering as a flattened disc.
        """
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        norm = torch.linalg.norm(vec_w, dim=1)
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(vec_w.shape[0], 1)
        arrow_scale[:, 0] *= norm * 3.0
        shrink = (norm / 0.1).clamp(max=1.0)
        arrow_scale[:, 1] *= shrink
        arrow_scale[:, 2] *= shrink
        # yaw from the planar components, pitch from the vertical one (negative pitch tilts +x upward)
        yaw = torch.atan2(vec_w[:, 1], vec_w[:, 0])
        pitch = -torch.atan2(vec_w[:, 2], torch.linalg.norm(vec_w[:, :2], dim=1))
        zeros = torch.zeros_like(yaw)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, pitch, yaw)
        return arrow_scale, arrow_quat


@configclass
class JumpVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the jump-augmented velocity command generator."""

    class_type: type[JumpVelocityCommand] = JumpVelocityCommand

    rel_jump_envs: float = 0.25
    """The sampled probability of environments that receive a pure-vertical jump command.
    Defaults to 0.25."""

    jump_period: float = 3.0
    """Length of a jump window [s]. Within each window the peak relative base height is
    tracked; at the window boundary the peak restarts and a new jump is owed. Defaults to 3.0."""

    ref_height: float = 0.27
    """Nominal standing base height above the local ground [m]. The target apex height is
    ``ref_height + jump_height``. Kept constant (not state-dependent) so the target cannot
    be lowered by crouching before the window starts. Defaults to 0.27 (measured settled
    standing height of the Go2)."""

    height_scanner_name: str | None = None
    """Name of the height-scanner sensor used to estimate the local ground height. If None,
    the environment origin is used as the ground, which is exact on flat terrain only.
    Defaults to None."""

    arm_height_threshold: float = 0.05
    """Height margin above :attr:`ref_height` [m] below which the base must pass before peak
    tracking is armed in a window. Masks the spawn-drop transient and prevents a single jump
    straddling a window boundary from being credited twice. Defaults to 0.05."""

    walk_height_deadband: float = 0.10
    """Peak base height above the reference height [m] tolerated for walking environments
    (zero jump command) before it is treated as an unwanted jump. Covers gait oscillation
    and posture changes; on rough terrain it should also cover the ground-estimate ripple
    when crossing steps. Defaults to 0.10."""

    jump_success_threshold: float = 0.10
    """Threshold on the absolute jump error [m] at the end of a completed window for the
    window to count as a delivered jump in the success metrics. Defaults to 0.10."""

    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        """Uniform distribution ranges for the velocity commands."""

        jump_height: tuple[float, float] = (0.20, 0.45)
        """Range for the commanded apex height gain [m], applied to jump environments only.

        The lower bound must exceed the pure leg-extension margin (~0.10 m for the Go2) so a
        commanded jump always requires liftoff. The upper bound may exceed the physically
        achievable apex (~0.35 m from actuator limits): the exponential reward kernel keeps a
        useful gradient near the limit, pushing the policy toward maximum effort. Defaults
        to (0.20, 0.45)."""

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""


def track_jump_peak_exp(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward matching the windowed peak base height to the commanded apex (exponential kernel).

    Uses :attr:`JumpVelocityCommand.jump_error`: for jump environments the error is symmetric
    around the target apex height, so both refusing to jump and overshooting reduce the reward
    for the rest of the window (the peak is monotone within a window, so an overshoot cannot
    be taken back). For walking environments any peak beyond the deadband is penalized, which
    discourages spurious jumps without punishing normal gait or posture changes.
    """
    command_term: JumpVelocityCommand = env.command_manager.get_term(command_name)
    return torch.exp(-torch.square(command_term.jump_error) / std**2)
