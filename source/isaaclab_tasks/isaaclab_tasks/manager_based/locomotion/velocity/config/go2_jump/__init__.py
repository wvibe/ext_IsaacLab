# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-Jump-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump_env_cfg:UnitreeGo2FlatJumpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:UnitreeGo2FlatJumpPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-Jump-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump_env_cfg:UnitreeGo2FlatJumpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:UnitreeGo2FlatJumpPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-Jump-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump_env_cfg:UnitreeGo2RoughJumpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:UnitreeGo2JumpPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-Jump-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump_env_cfg:UnitreeGo2RoughJumpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:UnitreeGo2JumpPPORunnerCfg",
    },
)
