# 周末学习 · 第 1 期周日进展记录 —— Allegro 转方块 + Isaac Sim 6 / PhysX

> 记录日期：2026-07-12（周日）
> 关联文档：
> - [weekend1-sunday-allegro-plan.md](weekend1-sunday-allegro-plan.md)（当日总计划）
> - [isaacsim6-physx-install-guide.md](isaacsim6-physx-install-guide.md)（Sim6 + PhysX 安装，已执行）
> - [go2-jump-task-retrospective.md](go2-jump-task-retrospective.md)（周六 Go2 复盘）
> - [isaaclab-cli-cheatsheet.md](isaaclab-cli-cheatsheet.md)（CLI 速查）
>
> 主线任务（定案）：`Isaac-Repose-Cube-Allegro-Direct-v0` + `physics=newton_mjwarp`
> 最佳 checkpoint：`logs/rsl_rl/allegro_hand/2026-07-12_11-36-46_allegro_baseline_resume/model_2500.pt`

本文汇总 2026-07-12 全日进展：环境补装、任务选型、基线训练与续训、B2 源码精读（问题 1–3），以及尚未做完的 B4/B5。  
**停笔时点**：晚间；B2 问题 4（ActuatorCfg）与 B4 实验留到下次。

---

## 1. 当天完成了什么（总览）

| 块 | 状态 | 结论 |
|---|---|---|
| Isaac Sim 6.0.1.0 + PhysX 后端补装 | ✅ | 本项目 `env_isaaclab` 内 pip 装完；Newton 回归 + PhysX Cartpole 验证通过 |
| B0 冒烟 / 后端定案 | ✅ | Direct + Newton 可用；Manager + `physics=physx` 因无 PresetCfg 报错 |
| B1 基线训练 | ✅ | ~1014 iter 已明显学会转方块；误续训到 ~2736，效果更好 |
| B3 初步观赏 | ⚠️ 部分 | viser 因 websockets 降级挂过一次，已修复；`--real-time` 可正常看；精看/录屏未做 |
| B2 源码精读 | ✅ 问题 1–3 / ⏳ 问题 4 | 逐函数读完 `inhand_manipulation_env.py`；问题 4（Actuator）留给 C2 前热身 |
| B4 实验 C1/C2/C3 | ⏳ 未做 | 基线已就绪，下次优先 C2 |
| B5 收尾 / 第 1 期总表 | ⏳ 未做 | 见文末「下次从哪里继续」 |

---

## 2. 环境侧：Isaac Sim 6 + PhysX（上午）

### 2.1 装法

- 位置：本仓库 venv `env_isaaclab/`（uv 创建，Python 3.12.11）
- 版本：`isaacsim[all,extscache]==6.0.1.0`（跟随 `source/isaaclab/setup.py` 钉版本）
- **关键经验**：走代理下载大包反复超时；绕过代理直连 `pypi.nvidia.com`→`pypi.nvidia.cn` CDN 约 100 MB/s，55 秒装完
- 命令骨架：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  UV_HTTP_TIMEOUT=600 \
  uv pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com
```

### 2.2 安装副作用与修复

| 包 | 装前 | 装后（被 isaacsim 改动） | 处理 |
|---|---|---|---|
| torch 三件套 | 2.10.0+cu128 | 2.11.0 默认轮 | 重钉回 cu128 ✅ |
| mujoco / mujoco-warp | 3.8.1 | 3.8.0 / 3.8.0.3 | 未干预；Newton 回归正常 |
| websockets | 16.1 | **12.0** | 升回 16.1，否则 viser 炸 ✅ |

EULA：终端 `printf 'Yes\n' | python -c "import isaacsim"` 持久接受，无需 GUI。

### 2.3 训练验证（安装指南 Step 5）

| 后端 | 任务 | 结果 |
|---|---|---|
| Newton MJWarp | Cartpole Direct，4096 envs | ✅ ~950k steps/s，正常收敛 |
| PhysX | Cartpole Direct（本地 wind 调参版） | 能跑但不收敛——调参在 Newton 下做的，跨求解器不迁移 |
| PhysX | Cartpole Manager `Isaac-Cartpole-v0` | ✅ success 1.0，~655k steps/s |
| CLI 语法定案 | — | **`physics=physx`**（不是 `presets=physx`） |

吞吐基线对照（Cartpole Direct，4096 envs，RTX 5090）：**Newton ~950k vs PhysX ~655k**（PhysX ≈ Newton 的 69%）。

快照：装前 `~/pre-isaacsim-freeze-0712.txt`，装后 `plan/env-snapshot-20260712.txt`。

---

## 3. Allegro B0：任务选型与后端定案

### 3.1 list_envs 实际任务名

| 任务 ID | 架构 | 物理预设 |
|---|---|---|
| `Isaac-Repose-Cube-Allegro-Direct-v0` | Direct（`InHandManipulationEnv`） | PresetCfg：physx / **newton_mjwarp** / ovphysx |
| `Isaac-Repose-Cube-Allegro-v0` | ManagerBased | **写死 PhysxCfg**，无 `physics=` 切换 |
| `Isaac-Repose-Cube-Allegro-NoVelObs-v0` 等 | ManagerBased 变体 | 同上 |
| DexSuite Kuka-Allegro / Warp 实验版 | 本期不碰 | — |

### 3.2 冒烟结果

**路线 A（成功）**：

```bash
./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Repose-Cube-Allegro-Direct-v0 \
    --num_envs=256 --max_iterations=10 --headless physics=newton_mjwarp
```

- 跑通；稳态约 **12.8k steps/s**（256 envs）
- 有 inertia / density warning，不影响训练
- 首次启动含 CUDA graph ~34s，属正常冷启动

**路线 B（失败原因已澄清）**：

```bash
# ❌ 会报 Unknown preset(s): physx
./isaaclab.sh train ... --task=Isaac-Repose-Cube-Allegro-v0 ... physics=physx
```

Manager 版 `inhand_env_cfg.py` 里物理是直接 `PhysxCfg(...)`，没有 `PresetCfg`，CLI `physics=` 找不到可切换节点。若硬要跑 Manager，应**去掉** `physics=`（默认就是 PhysX），但会拉起 Kit，与周末 kit-less 主线不一致。

### 3.3 定案

**今天全程**：`Isaac-Repose-Cube-Allegro-Direct-v0` + `physics=newton_mjwarp`

理由：B0 已通过；与 Cartpole/Go2 的 kit-less Newton 工作流一致；Direct cfg 的 Hydra 覆盖路径更浅，适合后续 C1/C2 实验。

---

## 4. 基线训练结果（B1）

### 4.1 两次 run

| Run | 目录 | Iter 范围 | 备注 |
|---|---|---|---|
| 基线 | `logs/rsl_rl/allegro_hand/2026-07-12_11-20-47_allegro_baseline` | 0 → ~1014 | 手动中断；checkpoint 到 `model_1000.pt` |
| 续训 | `logs/rsl_rl/allegro_hand/2026-07-12_11-36-46_allegro_baseline_resume` | 1000 → ~2736 | 从 `model_1000.pt` resume；因 `max_iterations` 语义误解多训到 3000 目标 |

**续训命令备忘**（`--max_iterations` = **总目标 iter**，不是「再训多少步」）：

```bash
./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Repose-Cube-Allegro-Direct-v0 \
    --num_envs=4096 --max_iterations=2000 --headless physics=newton_mjwarp \
    --resume --load_run 2026-07-12_11-20-47_allegro_baseline \
    --checkpoint model_1000.pt --run_name allegro_baseline_resume
# 若写 max_iterations=3000 → 会从 1000 训到 3000（再 2000 步）
```

### 4.2 指标曲线（关键节点）

| Iter | Mean reward | Episode length | Success rate | Consecutive successes |
|---|---|---|---|---|
| 0 | -6.0 | 16 | ~0% | 0 |
| 99 | 34 | 137 | ~3% | 0.04 |
| 199 | 99 | 290 | ~0% | 0.03 |
| 499 | 157 | 275 | 13% | 0.19 |
| **1014**（基线停） | **452** | **232** | **71%** | **1.49** |
| **2736**（续训末） | **1293** | **229** | **93%** | **5.31** |

吞吐：4096 envs 约 **100–107k steps/s**（RTX 5090）。

### 4.3 对照官方 benchmark

`test/benchmarking/configs.yaml` 对 Direct Allegro 的 500-iter 及格线：

| 指标 | 及格线 (500 iter) | 本机 1014 / 2736 |
|---|---|---|
| Mean reward | > 200 | 452 / **1293** ✅ |
| Episode length | > 150 | 232 / 229 ✅ |
| Success rate | （无硬线） | 71% / **93%** ✅ |

### 4.4 阶段解读

1. **0–200**：学会托住方块（episode length 爬到接近满长 ~300）
2. **200–500**：能托但还不太会转（success 仍低）
3. **500–1000**：突破期——开始稳定完成朝向重定向
4. **1000–2700**：链式转方块能力显著增强（consecutive 1.5 → 5.3）

**结论**：基线已是可用策略——多数时候不掉、能连续转多次目标。可进入 B3 精看与 B4 改造实验。

---

## 5. 代码地图（B2 精读入口，已摸清）

### 5.1 注册与入口

| 角色 | 路径 |
|---|---|
| Gym 注册 | `source/isaaclab_tasks/.../direct/allegro_hand/__init__.py` → `Isaac-Repose-Cube-Allegro-Direct-v0` |
| Env 逻辑 | `.../direct/inhand_manipulation/inhand_manipulation_env.py`（Allegro/Shadow 共用） |
| Env cfg | `.../direct/allegro_hand/allegro_hand_env_cfg.py` |
| PPO cfg | `.../direct/allegro_hand/agents/rsl_rl_ppo_cfg.py` |
| 手资产 | `source/isaaclab_assets/.../robots/allegro.py` → `ALLEGRO_HAND_CFG` |

Manager 对照（学 MDP 模块化时读，本期不训）：

- `.../manager_based/manipulation/inhand/inhand_env_cfg.py`
- `.../inhand/mdp/rewards.py`、`terminations.py`、`commands/orientation_command.py`

### 5.2 任务本质与关键超参（Direct）

- **任务**：16 DoF Allegro 掌心朝上托 DexCube，转到随机目标朝向；成功则刷新目标；掉落则 reset
- **观测**：`obs_type=full`，124 维；**`asymmetric_obs=False`**（无非对称 AC；Shadow OpenAI 变体才有）
- **动作**：16 维关节位置目标
- **旋转距离**：`rotation_distance` = `2 * asin(||quat_diff.xyz||)`（四元数 **xyzw**）
- **success_tolerance**：0.2 rad；**reach_goal_bonus**：250；**fall_dist**：0.24；**fall_penalty**：0
- **手指 ImplicitActuator**：stiffness **3.0**，damping **0.1**，effort_limit_sim **0.5**

### 5.3 账本框架（三行）

| 类别 | 项 | 量级 |
|---|---|---|
| 主账 | `rot_rew = 1/(rot_dist+rot_eps) * rot_reward_scale` | scale=1.0，eps=0.1 |
| 奖金 | 朝向误差 ≤ success_tolerance → +reach_goal_bonus | +250 |
| 罚金 | `action_penalty_scale * ‖a‖²`；掉落只终止不额外罚 | -0.0002；fall_penalty=0 |

---

## 6. 踩过的坑（当天）

| # | 坑 | 根因 | 正确做法 |
|---|---|---|---|
| 1 | Manager Allegro + `physics=physx` 报 Unknown preset | 无 PresetCfg，物理写死 | 去掉 `physics=`，或改用 Direct + `physics=newton_mjwarp` |
| 2 | viser `No module named websockets.asyncio` | isaacsim 把 websockets 降到 12.0 | `uv pip install 'websockets>=13,<17'` |
| 3 | 续训多跑了约 1000 iter | `--max_iterations` 是总目标，不是「再训 N 步」 | 从 1000 到 2000 应写 `--max_iterations=2000` |
| 4 | play `--load_run` + `--checkpoint model_2500.pt` 找不到文件 | play 里写了 `--checkpoint` 就**忽略** load_run，把文件名当 cwd 相对路径 | 只写 `--load_run`（自动取最新），或 `--checkpoint` 给完整相对路径 |
| 5 | play 太快看不清 | 默认不限速 | 加 `--real-time`；建议 `--num_envs=1` |

**Play 正确命令（当前最佳策略）**：

```bash
./isaaclab.sh play --rl_library rsl_rl --task=Isaac-Repose-Cube-Allegro-Direct-v0 \
    --num_envs=1 physics=newton_mjwarp --visualizer viser --real-time \
    --load_run 2026-07-12_11-36-46_allegro_baseline_resume
# 或：
# --checkpoint logs/rsl_rl/allegro_hand/2026-07-12_11-36-46_allegro_baseline_resume/model_2500.pt
```

TensorBoard：

```bash
./isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl/allegro_hand --port 6006
```

重点看：`Metrics/success_rate`、`Episode/consecutive_successes`、`Train/mean_episode_length`、`Train/mean_reward`。

---

## 7. B2 源码精读笔记（2026-07-12 晚间 · 问题 1–3 已完成）

主读：`source/isaaclab_tasks/.../direct/inhand_manipulation/inhand_manipulation_env.py`  
对照 cfg：`.../direct/allegro_hand/allegro_hand_env_cfg.py`  
资产（问题 4 入口，未热身）：`source/isaaclab_assets/.../robots/allegro.py`

### 7.1 问题 1 · 非对称 AC → ✅

- `asymmetric_obs = False`，`state_space = 0`；观测只有 `{"policy": obs}`
- 非对称时才有 `JointWrenchSensor` + `compute_full_state`（指尖力）→ Shadow OpenAI 变体才开
- 记法：力给 critic 无害、给 actor 毁 sim-to-real

### 7.2 问题 2 · 四元数旋转距离 → ✅

数据路径（验收口令）：

1. L325：`object_rot = root_quat_w`（**xyzw**）
2. L422：`quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))`
3. L423：**asin 版** `2*asin(‖quat_diff[:, 0:3]‖)` —— `0:3`=xyz，确认 xyzw
4. L452：`rot_rew = 1/(|rot_dist|+rot_eps)*rot_reward_scale`（倒数 shaping）
5. 观测同源相对 quat：L361

asin/acos 双覆盖等价；asin 用模长防符号，acos 需 `|w|`。

### 7.3 问题 3 · 账本与 consecutive → ✅

| 账目 | cfg | 值 | 分类 |
|---|---|---|---|
| 旋转倒数 shaping | `rot_reward_scale`, `rot_eps` | 1.0, 0.1 | 主账·收入（末端暴利，上限约 10/步） |
| 位置距掌心 | `dist_reward_scale` | -10.0 | 主账·支出 |
| 动作惩罚 | `action_penalty_scale` | -0.0002 | 主账·支出 |
| 达成 bonus | `reach_goal_bonus` | **250** | 奖金（≈25 步满额 shaping） |
| 掉落罚金 | `fall_penalty`, `fall_dist` | **0**, 0.24 | 实际不罚款（机会成本） |
| success 阈值 | `success_tolerance` | **0.2** rad | 元规则 |

consecutive：`_reset_target_pose` **只刷新 `goal_rot`**，方块原地继续。掉落在 `_get_dones`（≥0.24 m）终止。

### 7.4 读码新发现（滚动）

- L72 `goal_rot[:, 0]=1.0`：像 wxyz identity 遗留；xyzw 应为 `[:, 3]=1`（运行时会被覆盖）
- `fall_penalty=0`：掉落经济学 ≠ Cartpole 当场负奖

### 7.5 问题 4 · ActuatorCfg → ⏳ 下次 C2 前 5 分钟

| | Go2 腿 | Allegro 指 |
|---|---|---|
| 模型 | DCMotorCfg | ImplicitActuatorCfg |
| stiffness / damping / effort | 25 / 0.5 / 23.5 | **3.0 / 0.1 / 0.5** |

---

## 8. 下次从哪里继续

### 8.1 优先：B4-C2（必做）

指 stiffness ×0.5 → 先写预测 → 训 → vs 基线 → 填跨肢体对照表。  
Hydra 候选：`env.robot_cfg.actuators.fingers.stiffness=1.5`（以报错合法键为准）；或改 `allegro.py` 并留 git diff。

### 8.2 可选：C1（容差 ±1 倍）/ C3（质量随机，Direct 需改代码）

### 8.3 B3 精看 + B5

```bash
./isaaclab.sh play --rl_library rsl_rl --task=Isaac-Repose-Cube-Allegro-Direct-v0 \
    --num_envs=1 physics=newton_mjwarp --visualizer viser --real-time \
    --load_run 2026-07-12_11-36-46_allegro_baseline_resume
```

- [ ] 精看行为 + 录屏；第 1 期总表；三句话总结；hands.py（可选）  
- Play：**不要**同时写 `--load_run` + 裸 `--checkpoint model_xxx.pt`

最佳 checkpoint：`logs/rsl_rl/allegro_hand/2026-07-12_11-36-46_allegro_baseline_resume/model_2500.pt`

---

## 9. 第 1 期三任务吞吐草表（草稿）

| 任务 | 后端 | envs | 约 steps/s | 观测 / 动作 |
|---|---|---|---|---|
| Cartpole Direct | Newton / PhysX | 4096 | ~950k / ~655k | 4 / 1 |
| Go2 Flat/Rough | Newton | 4096–8192 | （见周六） | ~48–235 / 12 |
| Allegro Direct | Newton | 4096 | **~105k** | **124 / 16** |

---

## 10. 遗留问题（滚动）

- [x] Newton 支持 Allegro Direct；PhysX CLI=`physics=physx`；非对称 AC=否
- [x] B2 问题 1–3 精读落档（本文 §7）
- [ ] B2 问题 4 热身 + **C2 stiffness×0.5** + 对照表
- [ ] C1/C3（可选）；B3 精看/录屏；B5 总表与三句话
- [ ] L72 goal_rot 初始化约定；GUI hands.py；OVPhysX / 柔性朝向 / `--video`（远期）

---

## 11. 一句话结论（停笔 2026-07-12 晚）

已完成：**Sim6/PhysX 补装**、**Allegro 基线（~2700 iter，success~93%，consecutive~5.3）**、**B2 问题 1–3**（asin 旋转距离、倒数 shaping、只刷新目标的 consecutive）。  
**下次优先**：问题 4 热身 → **B4-C2** → 再视时间 C1 / B3 / B5。
