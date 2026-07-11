# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL agent configurations for the Go2 jump-augmented velocity task."""

from isaaclab.utils.configclass import configclass

from ..go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg


@configclass
class UnitreeGo2JumpPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_jump"


@configclass
class UnitreeGo2FlatJumpPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_jump_flat"
        self.max_iterations = 1000
