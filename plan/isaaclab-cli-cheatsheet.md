# IsaacLab 3 命令行速查表(本机实测版)

> 适用环境:kit-less pip 安装(无 Isaac Sim / PhysX / Kit GUI),Newton/MJWarp 物理后端,viser 可视化。
> 所有命令均在仓库根目录执行;`./isaaclab.sh` 会自动选用 `env_isaaclab/` 虚拟环境的 Python。
> 参数以本机代码为准(`scripts/reinforcement_learning/rsl_rl/{train,play}.py` + AppLauncher)。

---

## 0. 一图流:命令结构

```
./isaaclab.sh  <train|play>  --rl_library rsl_rl  --task=<任务名>  [脚本参数]  [AppLauncher参数]  [Hydra覆盖...]
```

三类参数可以混写,解析互不干扰:

| 类别 | 形式 | 例子 |
|---|---|---|
| 脚本参数 | `--key value` | `--num_envs=4096` `--run_name foo` |
| AppLauncher 参数 | `--key` | `--headless` `--visualizer viser` |
| Hydra 覆盖 | `key=value`(无 `--`) | `physics=newton_mjwarp` `'env.sim.physics.num_substeps=2'` |

---

## 1. 训练套餐(train)

### 1.1 正式训练(无可视化,吃满 GPU)

```bash
./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs=4096 --headless physics=newton_mjwarp --run_name my_exp
```

- `--num_envs`:PPO 是 on-policy,env 数直接决定样本量。**4096 起步,显存富余就 8192**(实测 8192 的 302 iter ≈ 4096 的 600 iter;单进程仅 ~3-4 GB 显存)。
- `--run_name`:日志目录后缀,**每个实验都起名**,否则只有时间戳难以辨认。
- `physics=newton_mjwarp`:本机唯一可用后端(`physics=physx` 需要完整 Isaac Sim,会直接报错)。

### 1.2 rough 地形训练(防 Newton NaN)

```bash
./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Velocity-Rough-Unitree-Go2-v0 \
    --num_envs=8192 --headless physics=newton_mjwarp --run_name my_rough \
    'env.sim.physics.num_substeps=2'
```

- `num_substeps=2`:接触密集地形上 Newton 会低概率发散出 NaN(实测 iter 196 崩过);substeps=2 换 ~20% 速度买稳定,rough 必加。

### 1.3 小规模观察训练(带实时可视化,学习效果差、仅供理解流程)

```bash
./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs=8 --headless physics=newton_mjwarp --visualizer viser --run_name tiny_watch
```

- viser 起本地 web 服务(默认 http://localhost:8080),浏览器实时看。
- **注意**:8 个 env 学不出有效策略(梯度噪声太大),这个套餐只用于看训练管线如何运转。

### 1.4 断点续训(resume)

```bash
./isaaclab.sh train --rl_library rsl_rl --task=<任务名> \
    --num_envs=8192 --headless physics=newton_mjwarp --run_name my_exp_resume \
    --resume --load_run 2026-07-11_16-58-56_my_exp --checkpoint model_150.pt
```

- `--load_run` / `--checkpoint` 都支持正则,缺省 `.*` = 各自取**最新**(目录按字母序、checkpoint 按数字);所以"从最新断点继续"只需 `--resume`。
- **奖励函数改过语义后不要 resume**——旧策略是需要反学习的负资产,从头训更快(跳跃任务实证)。

### 1.5 限制迭代数 / 固定随机种子

```bash
./isaaclab.sh train ... --max_iterations 300 --seed 42
```

- `--max_iterations` 覆盖 agent 配置里的默认值(Go2 rough 默认 1500);快速对照实验用 300 足够看出曲线形态。

### 1.6 录视频训练(kit-less 下待验证)

```bash
./isaaclab.sh train ... --video --video_length 200 --video_interval 2000
```

---

## 2. 回放/推理套餐(play)

### 2.1 标准回放:最新 run 的最新 checkpoint

```bash
./isaaclab.sh play --rl_library rsl_rl --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
    --num_envs 8 --headless --real-time physics=newton_mjwarp --visualizer viser
```

- **用 `-Play-v0` 变体任务**:env 数少、关闭观测噪声与随机推搡、禁用课程——专为目测设计。
- `--real-time`:按真实时间步进(否则以最快速度跑,肉眼看不清动作)。
- play 会自动导出 JIT/ONNX 到 checkpoint 同目录的 `exported/`。

### 2.2 回放指定 run / 指定 checkpoint(观察学习进程)

```bash
# 看训练早期(model_50)、中期(model_500)、最终(model_1499)的差别
./isaaclab.sh play --rl_library rsl_rl --task <Play任务名> \
    --num_envs 8 --headless --real-time physics=newton_mjwarp --visualizer viser \
    --load_run 2026-07-11_22-18-50_jump_v3_rough_fix2 --checkpoint model_50.pt
```

- checkpoint 落盘频率由 agent 配置的 `save_interval` 决定(Go2 系列默认每 50 iter)。

### 2.3 陷阱:rough 模型 ≠ flat 模型

rough 策略网络输入含 height_scan(observation 235/237 维),flat 是 48/50 维——**加载 rough checkpoint 必须用 rough 的 Play 任务**,否则报 `size mismatch for mlp.0.weight`。

---

## 3. Hydra 覆盖速查(踩坑实录)

覆盖路径 = 配置类的属性链,`env.` 前缀对应 EnvCfg,`agent.` 前缀对应 PPO RunnerCfg。

| 想改什么 | 写法 |
|---|---|
| 物理后端预设 | `physics=newton_mjwarp`(顶层预设,无前缀) |
| 物理 substeps | `'env.sim.physics.num_substeps=2'` |
| 某奖励项权重 | `'env.rewards.feet_air_time.weight=1.0'` |
| 删掉某奖励项 | `'env.rewards.flat_orientation_l2=null'` |
| 指令采样范围(tuple) | `'env.commands.base_velocity.ranges.lin_vel_x=(-1.5, 1.5)'` |
| 事件参数(dict) | `'env.events.push_robot.params.velocity_range={"x": (-1.5, 1.5), "y": (-1.5, 1.5)}'` |
| PPO 迭代数 | `'agent.max_iterations=2000'` |
| 熵系数 | `'agent.algorithm.entropy_coef=0.01'` |

**三条铁律**(均为实测踩坑):

1. **dict/tuple 值必须是合法 Python 字面量**——IsaacLab 用 `ast.literal_eval` 解析,dict 键必须带引号:`{"x": ...}` 对,`{x: ...}` 会**静默变成字符串**,直到事件首次触发才炸 `AttributeError: 'str' object has no attribute 'get'`(E3 踩坑)。
2. **整条覆盖用单引号包裹**,防 shell 吃掉括号/引号/空格。
3. 覆盖只在启动时生效,**不写回配置文件**;要复现实验请把命令行存档(TensorBoard 的 run 目录里有 `params/` 快照可查)。

---

## 4. 日志与 TensorBoard

```bash
# 日志目录结构:logs/rsl_rl/<experiment_name>/<时间戳_run_name>/
ls logs/rsl_rl/unitree_go2_flat/

# 起 TensorBoard(看所有 run 对比曲线)
./isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl/unitree_go2_flat --port 6006
```

**看板速读**(rsl_rl 的 per-item 记账):

| 面板 | 含义 | 健康形态 |
|---|---|---|
| `Episode_Reward/<term>` | 各奖励项的每 episode 均值(已乘权重) | 跟踪项爬升、惩罚项稳定小额 |
| `Metrics/<cmd>/error_*` | 指令跟踪误差(物理单位) | 单调下降 |
| `Metrics/success_rate` | 任务成功率(注意各任务定义不同) | 上升;**警惕"白送成功"的语义陷阱** |
| `Episode_Termination/*` | 终止原因占比 | time_out 占主,base_contact 低 |
| `Curriculum/terrain_levels` | 地形课程等级(rough) | 先降后升(先学走再上难度) |
| `Mean action std` | 策略探索度 | 从 1.0 缓慢收敛到 ~0.3-0.9 |

**曲线形态诊断**(比绝对值更重要):平线 = 梯度堵死(去查奖励管道),缓升 = 正常学习,快速收敛后目测不符 = 奖励被作弊行为填满。

---

## 5. 实用工具命令

```bash
# 列出已注册任务(验证自定义任务是否被自动发现)
# 注:官方 scripts/environments/list_envs.py 需要 Isaac Sim 的 EXP_PATH,kit-less 下会崩;用内联替代:
./isaaclab.sh -p -c "
import gymnasium as gym, isaaclab_tasks
print('\n'.join(sorted(s for s in gym.registry if 'Go2' in s)))
"

# 零动作冒烟环境配置(不加载策略;官方 zero_agent.py/random_agent.py 在 kit-less 下不可用)
./isaaclab.sh -p -c "
import torch, isaaclab_tasks
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
cfg = load_cfg_from_registry('<任务名>', 'env_cfg_entry_point')
cfg.scene.num_envs = 8
cfg = resolve_presets(cfg, ('newton_mjwarp',)) or cfg
env = ManagerBasedRLEnv(cfg=cfg)
for _ in range(100):
    env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
print('OK, obs dims:', {k: v.shape[-1] for k, v in env.observation_manager.compute().items()})
env.close()
"

# 内联 Python(用仓库环境,调试 API/写对照实验)
./isaaclab.sh -p -c "import isaaclab; print(isaaclab.__version__)"

# 跑指定测试文件
./isaaclab.sh -p -m pytest source/isaaclab/test/xxx.py

# pre-commit 全量检查(提交前必跑)
./isaaclab.sh -f

# GPU 监控(注意:Util% 高 ≠ 吃满,要看功耗和显存;单训练进程 ~3-4GB,可 2-3 路并发实验)
watch -n 2 nvidia-smi
```

---

## 6. AppLauncher 通用参数(所有脚本共享)

| 参数 | 用途 | 备注 |
|---|---|---|
| `--headless` | 不开 GUI | kit-less 机器**必加** |
| `--visualizer viser` | web 实时可视化 | 可与 `--headless` 共存;别名 `--viz` |
| `--device cuda:0` | 指定计算设备 | 默认 `cuda:0`;多卡分流实验用 |
| `--deterministic` | PyTorch 确定性模式 | 复现实验;有速度代价 |
| `--enable_cameras` | 启用相机传感器 | 视觉任务才需要 |
| `--livestream 1` | WebRTC 推流 | 需要 Kit,本机不可用 |
| `--max_visible_envs N` | 可视化只画前 N 个 env | 大规模训练+viser 观察时省资源 |

---

## 7. 本周期常用任务名备忘

| 任务 | 训练 | 回放 |
|---|---|---|
| Go2 平地行走 | `Isaac-Velocity-Flat-Unitree-Go2-v0` | `...-Flat-Unitree-Go2-Play-v0` |
| Go2 复杂地形 | `Isaac-Velocity-Rough-Unitree-Go2-v0` | `...-Rough-Unitree-Go2-Play-v0` |
| Go2 平地+跳跃(自定义) | `Isaac-Velocity-Flat-Unitree-Go2-Jump-v0` | `...-Flat-...-Jump-Play-v0` |
| Go2 rough+跳跃(自定义) | `Isaac-Velocity-Rough-Unitree-Go2-Jump-v0` | `...-Rough-...-Jump-Play-v0` |
| cartpole 入门 | `Isaac-Cartpole-v0` | 同名 |
| Allegro 转方块(周日) | `Isaac-Repose-Cube-Allegro-v0` | 同名 |

实验名(= 日志目录一级):`unitree_go2_flat` / `unitree_go2_rough` / `unitree_go2_jump_flat` / `unitree_go2_jump` / `cartpole_*`。
