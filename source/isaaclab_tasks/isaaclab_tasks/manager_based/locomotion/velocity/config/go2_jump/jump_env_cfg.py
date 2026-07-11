# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configurations for the Go2 jump-augmented velocity task (rough and flat)."""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg

from ..go2.flat_env_cfg import PhysicsCfg
from ..go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
from . import jump_mdp


@configclass
class JumpRewardsCfg(RewardsCfg):
    """Reward terms for the jump task: adds windowed jump-apex tracking to the common table.

    The kernel width is the perception radius of the reward: with ``std=0.1`` a non-jumping
    robot sits >3 sigma from every target and the gradient is numerically dead, which let a
    fear-crouch policy persist on rough terrain. ``std=0.2`` keeps crouch->stand->hop->jump
    monotonically and perceptibly rewarded from anywhere in the reachable state space, and
    the weight makes that slope competitive with the terrain-noise floor of the advantage.
    """

    track_jump_peak_exp = RewTerm(
        func=jump_mdp.track_jump_peak_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": 0.2},
    )


@configclass
class UnitreeGo2RoughJumpEnvCfg(UnitreeGo2RoughEnvCfg):
    """Go2 rough-terrain velocity task with an impulsive, height-based jump command.

    Differences from :class:`UnitreeGo2RoughEnvCfg`:

    * The command term is replaced with :class:`~.jump_mdp.JumpVelocityCommand`. Commands are
      mutually exclusive: 75% planar walking, 25% pure-vertical jump with a commanded apex
      height gain in (0.20, 0.45) m above the nominal standing height. One jump is owed per
      3 s window.
    * The ``lin_vel_z_l2`` penalty is removed. Vertical motion is governed by the
      :func:`~.jump_mdp.track_jump_peak_exp` reward, which for walking environments penalizes
      base-height peaks beyond a deadband and for jump environments symmetrically tracks the
      commanded apex height.
    * Heights are measured relative to the local ground estimated from the height scanner,
      so climbing stairs is not mistaken for a jump and jump targets stay reachable anywhere
      in a terrain cell. The walking deadband is widened to 0.15 m to absorb the ground-median
      ripple when crossing tall steps.
    """

    rewards: JumpRewardsCfg = JumpRewardsCfg()

    def __post_init__(self):
        # post init of parent (robot asset, action scale, reward weights, terminations)
        super().__post_init__()

        # replace the command generator with the jump-augmented version
        # (same planar ranges as the base template in velocity_env_cfg.py)
        self.commands.base_velocity = jump_mdp.JumpVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.02,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            rel_jump_envs=0.25,
            jump_period=3.0,
            ref_height=0.27,
            height_scanner_name="height_scanner",
            arm_height_threshold=0.05,
            walk_height_deadband=0.15,
            jump_success_threshold=0.10,
            debug_vis=True,
            ranges=jump_mdp.JumpVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.0, 1.0),
                heading=(-math.pi, math.pi),
                jump_height=(0.20, 0.45),
            ),
        )

        # vertical motion is commanded now, so the flat penalty on it must go:
        # jumping would otherwise be taxed at -2.0 * vz^2 per step and never pay off
        self.rewards.lin_vel_z_l2 = None


@configclass
class UnitreeGo2FlatJumpEnvCfg(UnitreeGo2RoughJumpEnvCfg):
    """Flat-terrain variant of the jump task (stage 1 of the curriculum).

    On flat ground the environment origin is an exact ground reference, so no height
    scanner is needed and the jump-apex signal is clean; the rough variant is stage 2.
    Terrain and observation trimming mirrors :class:`UnitreeGo2FlatEnvCfg`.
    """

    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards (same values as the official flat config)
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # the scene has no scanner anymore: fall back to the exact env-origin ground
        # and restore the tighter flat-terrain walking deadband
        self.commands.base_velocity.height_scanner_name = None
        self.commands.base_velocity.walk_height_deadband = 0.10


@configclass
class UnitreeGo2FlatJumpEnvCfg_PLAY(UnitreeGo2FlatJumpEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # more jump commands so the behavior is easy to observe
        self.commands.base_velocity.rel_jump_envs = 0.5


@configclass
class UnitreeGo2RoughJumpEnvCfg_PLAY(UnitreeGo2RoughJumpEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # more jump commands so the behavior is easy to observe
        self.commands.base_velocity.rel_jump_envs = 0.5
