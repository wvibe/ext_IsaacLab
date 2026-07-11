# Isaac Lab 3.0 周末学习计划 · 第 1 期:跑通、读懂、改动 Cartpole 与基础工作流

> **前置状态**:环境已装好并验证(见 `plan/` 下的安装指南 v2 与安装记录)。
> 仓库 `~/vibe/sim/IsaacLab3`,分支 `release/3.0.0-beta2`,newton 1.2.1,Cartpole 4096 envs 实测 ~106 万 steps/s。
> **本期目标(两天)**:①建立"任务=配置+环境类"的代码心智模型;②能自己修改奖励/观测/随机化并用肉眼+曲线验证效果;③固化一套 SSH 远程开发的看效果工作流。
> **工作方式**:IDE 通过 SSH 连接远程机;remote desktop 已可用(`--visualizer newton` 已验证)。
> **给 Agent 的角色定位**:本文档主要由**人**学习执行;Agent 负责查文件路径、解释代码、批量跑对照实验,但**每个"动手改"环节请用户亲手改亲手跑**——这是学习不是交付。

---

## 第 0 节 · 远程可视化工作流(先固化这个,后面全程复用)

四种看效果的方式,按使用频率排序:

### 方式 1:训练录像(日常主力,纯 SSH 不出 IDE)

训练时周期性录制 rollout 视频,存到 `logs/`,在 IDE 文件树里直接点开播放:

```bash
./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Cartpole-Direct-v0 \
    --num_envs=4096 --headless physics=newton_mjwarp \
    --video --video_length 200 --video_interval 2000
```

> 若报不认识 `--video` 参数或 kit-less 下录像报错,退回传统入口试一次:
> `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=... --headless --video ...`
> 仍不行则记录报错,改用方式 2/3,并把该问题记入"本期遗留问题"。

### 方式 2:TensorBoard + SSH 端口转发(曲线监控,常开)

远程机上:

```bash
./isaaclab.sh -p -m tensorboard.main --logdir logs/ --port 6006
```

本地终端(或 IDE 的 SSH 配置里加转发规则):

```bash
ssh -L 6006:localhost:6006 <user>@<远程机>
```

本地浏览器开 `http://localhost:6006`。VS Code/Cursor 类 IDE 通常会自动探测远程端口并弹"转发"提示,点一下即可,连命令都省了。

### 方式 3:play 回放 checkpoint(验收训练成果)

训练完不用重训,直接加载最新 checkpoint 在少量环境上回放,配合 remote desktop 看:

```bash
./isaaclab.sh play --rl_library rsl_rl --task=Isaac-Cartpole-Direct-v0 \
    --num_envs=16 physics=newton_mjwarp --visualizer newton
# 若 play 子命令不存在:./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task=... --num_envs 16
# play 也支持 --video,可把回放录成视频走方式 1 的看法
```

### 方式 4:实时可视化训练(仅调试期,remote desktop)

即你已验证的命令:小环境数 + `--visualizer newton`。吞吐会掉两个数量级(2756 vs 106 万 steps/s),所以**只用于"看看策略/环境行为对不对",不用于正式训练**。

另:环境里已装 `rerun-sdk 0.34.1`,`--visualizer rerun` 理论上是更适合 SSH 场景的轻量方案(独立窗口/网页,可端口转发)。留作第 2 天的探索项,见 Session 6。

**固化的节奏**:改代码 → 16 envs + visualizer 快速肉眼确认逻辑没错 → 4096 envs headless + video 正式训练 → TensorBoard 看曲线 → play 回放验收。

---

## 第 1 天(周六)

### Session 1 · 定位并通读 Cartpole Direct 任务(约 2 小时,上午)

**1.1 找到代码。**任务名到代码的映射靠 gym 注册表,让 Agent 帮你执行并解释输出:

```bash
# 列出全部任务,圈出 Cartpole / Ant / Anymal / Allegro 相关的名字,存入 plan/task-list.txt
./isaaclab.sh -p scripts/environments/list_envs.py | tee plan/task-list.txt

# 找到 Cartpole direct 的实现文件(beta2 重构过目录,以 grep 结果为准)
grep -rn "Isaac-Cartpole-Direct-v0" source/isaaclab_tasks/ --include="*.py" -l
```

顺着注册处的 `entry_point` 打开环境文件(通常是 `source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/` 下的 `cartpole_env.py` + 同目录 cfg)。

**1.2 通读,带着五个问题。**读的时候在纸上(或让 Agent 出一张表)回答:

1. `CartpoleEnvCfg` 里声明了哪些东西?(scene、robot 的 ArticulationCfg、sim 的 dt/decimation、episode 长度、action/observation 维度)
2. `_get_observations()` 返回了哪 4 个量?各自的物理含义?
3. `_get_rewards()` 由哪几项加权组成?每项的系数在哪定义?
4. `_get_dones()` 什么条件下终止 episode?(超时 vs 小车出界/杆倒)
5. `_reset_idx()` 里初始状态是怎么随机化的?(这就是最朴素的域随机化)

**1.3 对照 3.0 新约定验证两件事**(让 Agent 陪你在代码里找证据):

- 数据类返回的是 warp array 还是 torch tensor?环境代码里哪里做了 `wp.to_torch` 或等价转换?
- 搜一处四元数使用,确认它是 xyzw 顺序(或者确认 Cartpole 太简单根本没用四元数——那就到 `source/isaaclab/isaaclab/utils/math.py` 里看官方 math 模块的约定声明)。

**✅ 本节验收**:能不看代码向 Agent 复述"一个 direct 任务由哪几个方法构成、数据怎么流动",并让 Agent 挑错。

### Session 2 · 四个动手实验(约 3 小时,下午)——本周末的核心

每个实验都是同一个循环:**改一处 → 16 envs 可视化确认 → 4096 envs 正式训 → 曲线/录像对比基线**。先把上午跑过的原版结果留作基线(logs 目录改个名,如 `logs/rsl_rl/cartpole_baseline`)。

**实验 A:改奖励权重,看行为变化。**把"存活奖励"或"杆角度惩罚"的系数改大/改小 5 倍(cfg 里的 `rew_scale_*` 类字段)。预期可观察:角度惩罚加大 → 杆更笔直但小车可能更晃;存活项独大 → 收敛更快但姿态更松。**学到的**:奖励塑形的敏感性——这是你将来抓取任务 80% 的调参时间所在。

**实验 B:砍一个观测,看学习变难。**从观测里去掉小车速度(记得同步改 cfg 里的 observation 维度)。预期:仍能学会但收敛显著变慢/不稳——策略失去了速度信息只能从位置差分里"猜"。**学到的**:观测设计与部分可观测性的直觉;也顺手体验了"改观测要动几个地方"(这是新手最常见的报错来源,故意让你踩一次)。

**实验 C:加大初始随机化,看鲁棒性换成本。**把 `_reset_idx()` 里初始杆角度的随机范围扩大(比如 ±0.25 rad → ±1.0 rad)。预期:训练前期 reward 更低、最终策略更皮实(play 时用手动更极端的初值考验它)。**学到的**:域随机化的基本权衡,抓取任务里物体初始位姿随机化就是同一件事。

**实验 D:改物理参数,初见 sim 差异。**在 cfg 里把仿真 `dt` 减半(或把杆的质量/阻尼改一档),用同一份策略 checkpoint play——大概率行为退化。**学到的**:策略对物理参数的过拟合,这就是为什么存在域随机化和 sim-to-sim 验证;也为下周 PhysX vs Newton 对比埋下伏笔。

**✅ 本节验收**:TensorBoard 里有 5 条曲线(基线 + ABCD),每条你都能说清"改了什么、预期什么、实际看到什么、为什么"。建议让 Agent 把这张对照表写进 `plan/weekend1-experiments.md`。

### Session 3 · checkpoint 与回放工作流(约 40 分钟,收尾)

- 弄清 `logs/rsl_rl/<task>/<时间戳>/` 目录结构:checkpoint(`model_*.pt`)、配置快照、TensorBoard event 文件各在哪。
- 练一次指定 checkpoint 回放:play 命令加 `--checkpoint <路径>`(具体参数名以 `--help` 为准),分别回放"训练早期"和"最终"两个 checkpoint,肉眼看策略从瞎晃到站稳的演化——这个对比对建立 RL 直觉非常值。
- 把方式 1 的 `--video` 参数在你机器上实测一遍,确认录像落盘路径,写进工作流备忘。

---

## 第 2 天(周日)

### Session 4 · Manager-based 工作流解剖(约 2 小时,上午)

Direct 你已经熟了,现在看另一种风格——你同事的 2.x 项目和官方复杂任务都是这个。

**4.1** 找到 Cartpole 的 **manager-based 版本**(list_envs 输出里不带 Direct 字样的那个),打开它的 cfg 文件。这次没有 `_get_rewards()` 方法了,取而代之的是:

- `ObservationsCfg`:一组 `ObsTerm(func=mdp.xxx, ...)`
- `RewardsCfg`:一组 `RewTerm(func=mdp.xxx, weight=...)`
- `EventCfg`:reset/startup 时的随机化项(对应你昨天手写的 `_reset_idx` 随机化)
- `TerminationsCfg`

追一个 `mdp.joint_pos_target` 之类的 func 进去看实现(在 `isaaclab/envs/mdp/` 下),你会发现所谓 manager-based 就是**把 direct 类里的方法拆成了可配置的函数积木**。

**4.2 动手**:给 manager-based Cartpole 的 `RewardsCfg` 加一个现成的惩罚项(比如关节速度惩罚 `mdp.joint_vel_l2`,weight 给个小负数),训一把对比。体会"加奖励项只改配置不写逻辑"的开发效率差异。

**✅ 验收**:能向 Agent 讲清 direct vs manager-based 的取舍(灵活性/透明度 vs 复用性/组合性),以及你的抓取项目打算先用哪种、为什么。

### Session 5 · 上一个真机器人:locomotion 任务(约 1.5 小时,下午)

从倒立摆跨到 12+ 自由度的真实机器人,流程完全一样,只是规模变大:

```bash
# 任务名以 plan/task-list.txt 为准,选 Anymal 或 Go2 的平地行走(flat)任务
./isaaclab.sh train --rl_library rsl_rl --task=<Isaac-Velocity-Flat-Anymal-C-v0 或实际名> \
    --num_envs=4096 --headless physics=newton_mjwarp --video
```

看三样东西:①训练时长和 steps/s(与 Cartpole 对比,感受任务复杂度对吞吐的影响);②它的 cfg 里 `ActuatorCfg` 长什么样(stiffness/damping/effort_limit——抓取任务的手指关节就是同样的配置);③录像里步态从乱蹬到走顺的过程。时间富余的话,play 回放并用 remote desktop 实时看一眼。

### Session 6 · 探索项(自由裁量,约 1 小时)

按兴趣选一个,不求完成:

- **rerun 可视化**:试 `--visualizer rerun`,配合官方文档 *Newton Physics Integration → Visualization* 页,跑通 SSH 下的 rerun 看图路径(大概率需要端口转发或 rerun 的 serve 模式)。跑通的话它会成为你比 remote desktop 顺手得多的日常工具。
- **预读 Allegro**:打开 Repose-Cube-Allegro 任务源码通读一遍(不训练),对照 Session 1 的五个问题做笔记——这是下周灵巧手主线的起点。
- **补装 Isaac Sim 6.0.1**:若你确定下周要开始导入自定义手部资产,按安装指南附录 B 执行(15GB,挂代理),装完启动一次 GUI 确认能开即可,不深入。

### 收尾(20 分钟)

让 Agent 把本周末产出归档:`plan/weekend1-experiments.md`(实验对照表)、`plan/task-list.txt`、遗留问题清单(如 --video 是否可用、rerun 是否跑通)、以及你自己的三句话总结:最反直觉的一个发现 / 最没搞懂的一个概念 / 下周最想做的一件事。

---

## 常用命令速查(本期用到的全集)

```bash
lab3   # 别名:进目录+激活环境(若已按安装指南配置)

# 列任务 / 找代码
./isaaclab.sh -p scripts/environments/list_envs.py
grep -rn "<任务名>" source/isaaclab_tasks/ -l

# 训练(kit-less + Newton)
./isaaclab.sh train --rl_library rsl_rl --task=<T> --num_envs=4096 --headless physics=newton_mjwarp
# 快速肉眼验证(remote desktop)
./isaaclab.sh train --rl_library rsl_rl --task=<T> --num_envs=16 --max_iterations=200 physics=newton_mjwarp --visualizer newton
# 回放
./isaaclab.sh play --rl_library rsl_rl --task=<T> --num_envs=16 physics=newton_mjwarp --visualizer newton
# 曲线
./isaaclab.sh -p -m tensorboard.main --logdir logs/ --port 6006   # 本地 ssh -L 6006:localhost:6006

# 传统入口(新子命令缺参数时的后备)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=<T> ...
```

## 本期明确不碰的内容(防发散)

柔性/VBD(第 3 期)、自定义资产导入与 Isaac Sim GUI 工具链(第 2 期)、PhysX vs Newton 对照实验(第 2 期)、多 GPU/集群、模仿学习 mimic、sim-to-real。周末的敌人是贪多。
