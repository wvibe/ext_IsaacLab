# 周末学习 · 第 1 期下午 —— Go2 四足行走:Manager-based 体系解剖与实验

> 承接 `weekend1-experiments.md`(上午 Cartpole)。任务:`Isaac-Velocity-Flat-Unitree-Go2-v0`(名字以 list_envs 实际输出为准)。
> **主线**:上午你在 direct 环境里手写了风扰动;下午你会发现 manager-based 体系里"加扰动"只是一行配置——亲身对比两种工作流的开发成本,同时学会四个新概念:**CommandManager、按项记账的奖励日志、EventManager 内置域随机化、ActuatorCfg**。
> **时间预算**:约 3.5~4 小时。训练量级(已核对官方配置):Go2 **平地**任务官方默认就是 **300 iter**(`agents/rsl_rl_ppo_cfg.py` 里 flat 覆盖为 300,1500 是崎岖地形版的量级),5090 上单次训练估计 **3~6 分钟**。基线和 E1~E5 全跑完时间绰绰有余,不必刻意压缩实验数量。

---

## ⏸ 上下文快照(2026-07-11 14:00 写入,新会话从这里续接)

> 本节由 Agent 会话结束前写入,目的是让任何新开的对话读完本文档即可无缝接手。
> **当前断点(17:00 更新):A0~A2 主体完成(E1/E3/E5 + rough 系列已跑,E2/E4 未做)→ Agent 已实现"连续 vz 跳跃指令"扩展任务 `go2_jump` 并挂起训练。全部细节见文末"⏱ 下午实际执行记录"。**

### 机器与仓库状态

- 仓库:`~/vibe/sim/IsaacLab3`(3.0.0-beta2),venv 在 `env_isaaclab/`,一切命令在仓库根目录执行。
- git:当前分支 `wei/weekend1-cartpole`(含上午 cartpole 风扰动代码 + plan 文档共 2 个 commit)。remote:`origin`=官方(只 fetch),`fork`=`git@github-vibe:wvibe/ext_IsaacLab.git`(个人小号,SSH 别名 `github-vibe`,分支已推上去并设好跟踪,`git push` 默认进 fork)。`release/3.0.0-beta2` 也推到了 fork 供网页 diff。
- 网络:GitHub 走代理(mihomo,127.0.0.1:7897);**S3/资产下载必须直连**——确认 `echo $no_proxy` 含 `amazonaws.com`,否则资产下载报 `FileNotFoundError`(上午踩过)。
- 注意:`~/vibe/sim/IsaacLab`(无 3)是旧 v2.3 克隆,**只作对照,勿在里面看/改代码**。本文档所有相对路径均相对 `IsaacLab3` 仓库根。

### 这是一个什么任务(给新会话的任务背景)

`Isaac-Velocity-Flat-Unitree-Go2-v0` = **训练一个"万向遥控行走控制器"**:平地上 4096 只并行 Go2,每 10 秒收到随机速度指令(x/y 线速度、转向角速度,±1 范围,2% 环境命令站立),策略以 50 Hz 读观测(48 维:基座线/角速度、重力投影、指令、关节位置/速度、上帧动作)、输出 12 个关节的目标角偏移。目标:**不管什么指令都用稳定、平滑、省力的步态精确跟踪**。产物是可部署的底层运动控制器(上层手柄/导航/大模型发指令即可驱动),这是四足 sim-to-real 的标准范式。与上午对比:Cartpole 目标固定,这里是 goal-conditioned——观测里 3 维 `velocity_commands` 是它的"耳朵"。episode 20 秒,基座触地即死,超时属正常截断。

### 代码地图(A1 精读材料,已由 Agent 核对过,路径相对仓库根)

| 层级 | 文件 | 要点 |
|---|---|---|
| 注册 | `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/__init__.py` | 任务名→配置类。entry_point 是通用类 `ManagerBasedRLEnv`(对比 direct:环境=引擎,配置=任务)。4 个变体:Flat/Rough × 训练/Play |
| PPO 超参 | 同目录 `agents/rsl_rl_ppo_cfg.py` | flat 覆盖 `max_iterations=300`、`experiment_name="unitree_go2_flat"`(1500 是 rough 的量级) |
| 通用模板 | `.../velocity/velocity_env_cfg.py`(384 行) | 五 Manager 全在此,机器人留空 `robot = MISSING`(L110)。详见下方"五 Manager 速记" |
| Go2 特化 | `.../velocity/config/go2/rough_env_cfg.py` + `flat_env_cfg.py` | 合计约 35 行实质差异:换资产、动作 scale 0.25、调 6 个奖励权重、终止 body 名 `"base"`;flat 再:地形换平面、删 height_scan(观测 48 维)、`feet_air_time`→0.25、`flat_orientation_l2`→-2.5。flat 里还定义 `physics=newton_mjwarp` 预设(L17-31) |
| 奖励函数库 | `.../velocity/mdp/rewards.py` | `feet_air_time`(L27-46):只在脚**落地帧**结算,腾空时长减 0.5s 阈值,站立指令不给分(防原地跳) |
| 机器人资产 | `source/isaaclab_assets/isaaclab_assets/robots/unitree.py` | `UNITREE_GO2_CFG`(L142-183):`DCMotorCfg` stiffness=25, damping=0.5, effort_limit=23.5——策略给目标角,内部 PD 算力矩再按电机模型饱和。**E4 目标,下周灵巧手同款结构** |

**五 Manager 速记**(`velocity_env_cfg.py` 内):
- `CommandsCfg`(L140-151):每 10s 重采样速度指令 ±1,`rel_standing_envs=0.02`。E5 改 `ranges`。
- `ObservationsCfg`(L162-192):单 policy 组 8 条款按序拼接,6 项带均匀噪声,无 critic 组(此模板不搞 teacher-student)。
- `RewardsCfg`(L284-316):主账 `track_lin_vel_xy_exp`(exp 核,flat 权重 1.5)+ `track_ang_vel_z_exp`(0.75);税:lin_vel_z -2.0、ang_vel_xy -0.05、力矩 -1e-5、关节加速度、`action_rate_l2` -0.01(E2)、大腿接触 -1.0;美学正项 `feet_air_time`(E1)。
- `EventsCfg`(L196-280):startup 摩擦/质量随机,reset 出生随机,interval `push_robot`(L275-280,每 10~15s 基座速度设为 ±0.5 随机——上午 20 行手写风扰动在这里是 6 行声明,E3 调到 ±1.5)。**注意 L226-236 `base_com` 用 `preset(..., newton_mjwarp=None)` 在 Newton 后端显式禁用**——beta 版后端差异实锤。
- `TerminationsCfg`(L320-327):20s 超时(不罚)+ 基座触地即死。

### 基线结果(2026-07-11 13:57 完成,已核对日志)

- 命令:见下方 A0(带 `--visualizer viser` 版本);**训练仅 104 秒**,29M 步,约 29 万步/秒(5090 实测,比预估还快)。
- 关键指标(iter 299):**success_rate 98.96%**,ep_len 990/1000,`error_vel_xy` 0.10 m/s(≈10%),`error_vel_yaw` 0.17,`track_lin_vel_xy_exp` 1.41/1.5(拿到 94%),base_contact 终止仅 0.6%。
- **有趣发现:`feet_air_time` = -0.008,是负的**——策略选择快步频小碎步(单步腾空 < 0.5s 阈值),宁可此项持续小额扣分也要保速度跟踪的大头。奖励博弈活例子,E1(×4)的前后对照参照点。
- 日志目录:`logs/rsl_rl/unitree_go2_flat/<时间戳>_baseline/`。
- 训练结束时 viser 报 `RuntimeError: cannot schedule new futures after shutdown`——**无害**,是进程退出时浏览器还连着 websocket 的收尾噪音,checkpoint 完好。

### 速查命令(本机已验证)

```bash
# 训练(带网页实时可视化;viser 地址看日志,默认 :8080;关标签页不影响训练)
./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs=4096 --headless physics=newton_mjwarp --visualizer viser --run_name baseline

# 回放最新 checkpoint(Play 变体默认关 push 和观测噪声,见 E3 陷阱)
./isaaclab.sh play --rl_library rsl_rl --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
    --num_envs 16 --headless --real-time physics=newton_mjwarp --visualizer viser

# TensorBoard(重点面板:Episode_Reward/* 按项记账、Metrics/base_velocity/error_*)
./isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl/unitree_go2_flat

# 回放指定 checkpoint(不带 --checkpoint 默认加载"最新 run"——小心加载到废号!)
./isaaclab.sh play --rl_library rsl_rl --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
    --num_envs 8 --headless --real-time physics=newton_mjwarp --visualizer viser \
    --checkpoint logs/rsl_rl/unitree_go2_flat/<run>/model_50.pt
# 注意:checkpoint 绑定观测维度+网络结构,rough 的 checkpoint 必须配 Rough 任务名

# 断点续训(崩溃恢复;地形课程等级会重置但权重继承)
./isaaclab.sh train ... --resume --load_run <run目录名> --checkpoint model_150.pt

# Hydra 覆盖语法(实测):标量直接写;dict 值必须是合法 Python 字面量(键带引号)
'env.rewards.feet_air_time.weight=1.0'
'env.events.push_robot.params.velocity_range={"x": (-1.5, 1.5), "y": (-1.5, 1.5)}'
'env.sim.physics.num_substeps=2'   # Newton rough 防 NaN(速度约减半)
```

---

## Session A0 · 基线 + 顺手清掉两个遗留问题(约 30 分钟)

```bash
# 1. 确认任务名并跑基线(后台挂着,挂完就去读代码)
#    注意:--max_iterations 不传,用官方默认 300;--run_name 给每次 run 打标签,
#    否则 logs/rsl_rl/unitree_go2_flat/ 下全是时间戳,TensorBoard 里分不清谁是谁
./isaaclab.sh -p scripts/environments/list_envs.py | grep -i "go2\|anymal"
./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs=4096 --headless physics=newton_mjwarp --run_name baseline

# 2. 【遗留问题①】顺手测 --video:在上面命令里加
#    --video --video_length 200 --video_interval 500
#    可用→写进速查;报错→记录报错原文,录像需求继续由 viser/play 承担

# 3. 训练挂着的同时,开 TensorBoard(你会看到与上午完全不同的曲线面板,见 A1.4)
```

**【遗留问题②的处置建议】**上午发现的 `[rad]` 注释误导:先去 IsaacLab GitHub issues 搜 `initial_pole_angle_range`,没人报过就把你的复现记录整理成 issue(中文草稿让 Agent 翻译润色)。这是很好的第一次上游互动,但别占用下午超过 15 分钟——挂出去就继续。

## Session A1 · 解剖配置栈(约 1 小时,基线训练挂后台时做)

**A1.1 先找到配置继承链。**manager-based 任务的 cfg 不在一个文件里,是一条继承链。让 Agent 帮你把这条链画出来:

```bash
grep -rn "Isaac-Velocity-Flat-Unitree-Go2" source/isaaclab_tasks/ -l
# 顺着 entry_point 找,典型结构:
#   velocity_env_cfg.py(定义 Obs/Rew/Event/Command 的通用模板,所有腿足共用)
#     └─ unitree_go2 的 flat_env_cfg.py(继承后按机型微调:改权重、砍传感器)
#           └─ isaaclab_assets 里的 UNITREE_GO2_CFG(ArticulationCfg + ActuatorCfg)
```

读的目标是回答:**"Go2 和 Anymal 的行走任务,到底差在哪几行?"**——答案少得惊人(已核对:`go2/rough_env_cfg.py` 相对通用模板的实质差异约 25 行——换机器人资产、调 6 个奖励权重、改终止判定的 body 名;`flat_env_cfg.py` 再叠约 10 行),这就是 manager-based 的复用主张。

**A1.2 带六个问题精读 `velocity_env_cfg.py`:**

1. `CommandsCfg`(上午没有的新东西):速度指令怎么采样?范围多大?多久重采样一次?——RL 的"任务目标"在这里定义,策略学的不是"往前走"而是"跟踪任意指令"。
2. `ObservationsCfg`:policy 组里有哪些项?哪些加了 noise?有没有第二个组(critic/privileged)?——如果有,这就是 teacher-student 的伏笔(critic 能看真实速度,部署时策略看不到)。
3. `RewardsCfg`:数一数有几个 RewTerm,按你上午的"账单占比"框架把它们分成主账(速度跟踪 exp 项)和税(action_rate、joint_accel、feet_air_time…)。
4. `EventCfg`:**找到 `push_robot` 这一项**——官方内置的"风",interval 模式定期给基座一个随机速度。对照你上午手写的 20 行风扰动代码,体会两种工作流的成本差。再看看还有哪些 startup 随机化(质量、摩擦)。
5. `TerminationsCfg`:什么姿态判死?有没有非法接触终止?
6. Go2 的 `ActuatorCfg`:stiffness/damping/effort_limit 是多少?是 Implicit 还是显式执行器模型?——**这组参数就是下周灵巧手手指关节的同款配置,认真看。**

**A1.3 追一个 RewTerm 的实现**。挑 `feet_air_time` 进 `mdp/rewards.py` 看函数体(它要读接触传感器!),搞清 `func + params + weight` 三件套怎么被 RewardManager 调用。这一步之后,manager-based 对你就不再是黑盒。

**A1.4 看 TensorBoard 的新面板——上午痛点的官方解法。**manager-based 会把**每个奖励项单独记账**(`Episode_Reward/track_lin_vel_xy_exp`、`Episode_Reward/action_rate_l2`…)。上午你总结过"改奖励后总 reward 失去横向可比性";在这里,即使改了某项权重,其他项的原始曲线仍然可比。把这个面板用熟,它是下午所有实验的裁判。

**✅ A1 验收**:向 Agent 口述"一个观测值从物理引擎到策略网络输入的完整路径"(引擎状态 → ObsTerm func → noise → clip/scale → 拼接成 policy 组向量),让它挑错。

## Session A2 · 四个实验(约 2 小时,核心环节)

沿用上午格式:改动 → **先写预测** → 训练 → 曲线+viser 验证 → 记录。全部用 Hydra 覆盖,不改文件(manager 项的覆盖路径形如 `'env.rewards.<term名>.weight=...'`,具体项名以 A1 读到的为准;路径写错会报 cfg key 错误,报错信息里会列出合法键,本身就是学习材料)。每个实验用默认 300 iter 与基线对比,并加 `--run_name E1_feet_air` 之类的标签。

**执行次序约定**:E1~E3 与 E5 全部走 Hydra 覆盖,顺序随意;**E4 必须最后做**——它大概率要改 `isaaclab_assets` 里的共享机器人配置文件,会污染其他实验,做完立即 `git checkout` 还原。

**实验 E1 · 步态美学:`feet_air_time` 权重 ×4。**
预测练习:腾空时间奖励加大,步态会______(更大步幅/更高抬腿/还是干脆跳起来?)。速度跟踪项会付出多少代价?
观察:viser 里对比基线与 E1 的步态;TensorBoard 看 `track_lin_vel` 项是否下降。
学习点:美学项与任务项的博弈——这类"步态塑形"手法在灵巧手上的对应物是"手指姿态自然性"奖励。

**实验 E2 · 平稳税重演:`action_rate_l2` 权重 ×5。**
预测:参照上午实验 2 的"平稳税 vs 救援敏捷"教训,这次税加在动作变化率上,推倒恢复能力会怎样?收敛速度呢?
观察:重点看被 push 之后的恢复表现(viser 里等 push 事件发生,或用 play + 手动加大 push 幅度考它)。
学习点:同一课在 12 自由度上的复习——确认上午的直觉可迁移。

**实验 E3 · 官方版"加大风力":push_robot 事件强度 ×3。**
把 `env.events.push_robot` 里的速度范围调大(具体 params 键名以代码为准),等价于上午你的 ±4→±12。
预测:训练前期曲线会___,最终鲁棒性会___;与 E2 组合看:高平稳税 + 强推搡,策略还学得会吗?
**⚠️ 已核实的陷阱**:`-Play-v0` 变体在 `flat_env_cfg.py` 末尾显式设置 `self.events.push_robot = None`(观测噪声也一并关掉)——官方立场是 **play 时默认关闭所有随机化**。所以想在 play 里看"被推后的恢复",要么用非 Play 任务名 + `--num_envs 16`,要么在 play 命令里用 Hydra 把 push 重新打开。
学习点:内置 EventManager 的用法;以及**验收方式**——训练时开扰动 vs 只在 play 时开扰动的对照,上午你已总结过,这里用官方机制重跑一遍;Play cfg 的这个设计本身就是官方给出的答案。

**实验 E4 · 执行器灵魂拷问:stiffness 减半。**
把 Go2 ActuatorCfg 的 stiffness 砍半(这个在 asset cfg 里,Hydra 路径较深,找不到就直接改文件、记得 git diff 留痕)。
预测:关节跟踪变"软",步态会___;策略能不能通过学习补偿硬件变弱?
观察:`Episode_Reward` 各项 + viser 步态松垮程度。
学习点:**这是下周最重要的预习**——灵巧手抓取对指尖刚度极其敏感,stiffness/damping 将是你抓取任务的头号超参。

**(选做)实验 E5 · 指令域扩展**:把 `CommandsCfg` 的速度范围上限提高 50%,看策略是全域变差还是学出"高速小跑/低速慢走"两种模态。学习点:任务分布的宽度 vs 单点性能。

**✅ A2 验收**:实验表(格式沿用上午)写入本文档末尾;每行有"预测 vs 实际"两栏——**预测错的行比预测对的更有价值,单独标注**。

## Session A3 · 收尾与桥接(约 20 分钟)

1. 保留一个最满意的 Go2 checkpoint,play + viser 录一段屏,作为第 1 期的"毕业作品"。
2. 让 Agent 把 direct vs manager-based 的对照表(你现在两边都有实战了)写成 10 行以内的笔记:各自改动成本、透明度、调试体验、你的抓取项目选型倾向。
3. 预览明天:打开 `Isaac-Repose-Cube-Allegro-v0` 的源码目录**只读 20 分钟不跑**,用上午的五问过一遍,把"看不懂的名词"列成清单——那就是明天的学习目标。

---

## ⏱ 下午实际执行记录(17:00 由 Agent 写入)

实际执行与原计划的偏差和额外收获,按时间序:

1. **8 狗小规模训练实验(14:27)**:num_envs=8 训练 300 iter → success 0%、action std 0.95 不降、每局必摔。教训:**并行度是 on-policy RL 的粮食**,8 envs × 24 步 = 192 样本/迭代,梯度噪声大到无法学习。观看训练用 64 envs 起步;正经实验一律 4096。
2. **逐 checkpoint 回放工作流确立**:训练 4096 全速跑,观察用 play + `--checkpoint model_0/50/100/299.pt` + 8 狗回看学习进程——比实时看训练好(可回放、可对比、条件干净)。
3. **E3 踩坑(14:52)**:Hydra 覆盖 dict 参数写成 YAML 风格 `{x:[-1.5,1.5]}` → `ast.literal_eval` 解析失败**静默回退成字符串**,训练到 iter 19(首次 push 事件触发)才炸 `AttributeError: 'str' object has no attribute 'get'`。修正:键带引号的 Python 字面量。**值解析错误 fail-late,路径错误 fail-fast**——查 `logs/.../params/env.yaml` 可验证覆盖是否真正生效。
4. **E1/E3/E5 结果**:见实验记录表。E5 的崩溃模式(放弃整门生意而非全域略差)是今天最有价值的意外发现。
5. **Rough 系列(改进实验 2)**:结果见表。两个平台级发现:**① Newton/MJWarp 在崎岖地形(接触密集)上会低概率 NaN 崩溃**(基线 iter 196 崩,resume 版 753 iter 平安→随机事件;`num_substeps=2` 两个 run 均未再崩,速度代价约一半);**② 本机 kit-less 安装没有 PhysX**(需 Isaac Sim,明日 B4)。对下周灵巧手(接触密集)的后端选型是关键输入。
6. **checkpoint 与任务绑定教训**:rough 的 checkpoint(235 维观测/[512,256,128])加载进 flat Play 环境(48 维/[128,128,128])直接 size mismatch——checkpoint 不是"狗脑",是绑定观测空间和网络结构的矩阵组。
7. **GPU 效率认知**:nvidia-smi 的 92% Util ≠ 榨干(功耗 230W/600W、显存 4.4/32.6GB)。两个杠杆:num_envs 加倍(8192 实测 302 iter ≈ 4096 版 600 iter 水平)、多实验并行(单进程仅 ~3GB 显存,可 3 路并发)。

> **跳跃任务全程复盘已单独成文**:[go2-jump-task-retrospective.md](go2-jump-task-retrospective.md)——含 v1→v3 三版设计演进、峰值停靠零梯度陷阱的诊断全过程、沉淀的十条方法论。以下各节是当日的即时执行记录,细节以复盘文档为准。

### 改进 1:连续 vz 跳跃指令任务 `go2_jump`(Agent 实现,待验收)

**需求**(用户口述):跳跃指令不是 0/1 旗标,而是**连续的 z 轴速度指令**——箭头越高起跳越用力,低箭头做小跳;允许 x+z(前跳)、y+z(侧跳)混合分量;在**全地形(rough)**上训练;原 `lin_vel_z_l2` 惩罚整项去掉,z 轴上"看指令与实际是否一致"(跟踪奖励)。

**设计要点**:
- 指令向量 3→4 维:`(vx, vy, ωz, vz)`——**vz 放第 4 维**,前 3 维顺序不动,所有现有奖励/观测(`[:, :2]`、`[:, 2]`)零改动兼容。
- vz 采样:`rel_jump_envs=0.3` 概率抽 `U(0.3, 1.2)` m/s,否则 0——多数环境正常走路,30% 带跳跃分量;与 vx/vy 独立采样,混合方向自然涌现。
- 奖励:新增 `track_lin_vel_z_exp`(exp 核,权重 1.0)替代删除的 `lin_vel_z_l2`——**vz 指令为 0 时它自动奖励"别颠簸"(软化版旧税),vz>0 时奖励向上冲**,一个项两用,自洽。
- 物理常识:重力下无法持续跟踪 vz>0,预期涌现行为是**反复起跳**(上升相匹配指令拿分,下落相亏分)——指令越大,起跳越猛才划算。这本身就是实验假设,待曲线验证。
- viser 箭头升级:重写 `_debug_vis_callback`,箭头方向含 pitch 分量——**vz 指令大时箭头真的指向天上**。
- 新增 `Metrics/base_velocity/error_vel_z` 按项记账。

**文件清单**(全部新增,零官方文件改动,git 可直接 diff):
```
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2_jump/
├── __init__.py       # 注册 Isaac-Velocity-Rough-Unitree-Go2-Jump-v0 (+Play)
├── jump_mdp.py       # JumpVelocityCommand(Cfg) + track_lin_vel_z_exp
├── jump_env_cfg.py   # UnitreeGo2RoughJumpEnvCfg (+PLAY):换指令项、删 lin_vel_z_l2、加 z 跟踪
└── agents.py         # PPO cfg(experiment_name="unitree_go2_jump")
```

**状态(17:10)**:冒烟测试通过;`jump_v1` 训练 1500 迭代完成(error_vel_z 0.22)。**但 play 目测未见真实跳跃**——v1 复盘出三个缺陷:① 水平分量视觉/信号上淹没垂直分量;② 均匀逐步跟踪 vz 的语义错误(跳跃是脉冲事件不是持续状态,策略用小幅弹跳蹭分);③ 坡地上 vz 与地形纠缠(爬坡被误罚)。

### 改进 1 修订版(v2,19:40 实现):脉冲式窗口峰值跳跃

v1 → v2 的三个设计替换:

- **二分指令(互斥)**:75% 平面行走(vz=0)/ 25% 纯垂直跳跃(vx=vy=ωz=0、heading 关闭,vz∈U(0.5,1.5))。混合方向留作第三阶段微调。指令维度 4→5,第 5 维是**带符号跳跃残差**(vz_cmd − 已达峰值):正=还欠一跳,零=已达成,负=跳过头(马尔可夫性修复);目标箭头的垂直分量画残差正部——**跳完箭头当场消失**。
- **窗口峰值奖励**(替换逐步跟踪):每 3 s 一个窗口,记录**世界系** vz 的历史最大值 peak,奖励 `exp(-(peak-cmd)²/0.25)`、权重 1.5。peak 只增不减 → 早跳早锁分、跳过头持续扣分到窗口结束、不跳持续低分,"一正一负"由同一核函数给出。
- **坡地区分**:跳跃环境是原地跳,与坡度解耦;行走环境(cmd=0)给 0.5 m/s **死区**——步态起伏(~0.3)和爬坡合法 vz(~0.4)免罚,偷跳(≥0.7)被罚;下坡 vz 为负、不抬高 max,天然免罚。峰值用世界系测量,基座俯仰不失真。

**扭矩可行性核算**(实测整机质量 15.69 kg):vz=1.5 m/s 起跳需四腿总蹬地力 242 N,折算单膝 ~13.3 N·m < 23.5 N·m 限幅,蹬伸期关节速度 ~8 rad/s « 30 rad/s——约 40% 余量,可行;**指令上限勿超 1.5**(只用后腿蹬时单腿已贴限)。

新任务名:`Isaac-Velocity-Flat-Unitree-Go2-Jump-v0`(第一阶段,平地,信号纯净)/ `...-Rough-...-Jump-v0`(第二阶段);各带 Play 变体(跳跃占比 50%)。冒烟测试三项全过(flat 观测 50 维、rough 237 维、指令 5 维、纯垂直约束成立、窗口 150 步边界峰值正确清零、无 NaN)。

**训练命令(用户执行)**:
```bash
# 第一阶段:平地(实验名 unitree_go2_jump_flat,默认 1000 迭代)
./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Velocity-Flat-Unitree-Go2-Jump-v0 \
    --num_envs=4096 --headless physics=newton_mjwarp --run_name jump_v2_flat
# 第二阶段:rough(建议加 substeps=2 防 NaN)
./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Velocity-Rough-Unitree-Go2-Jump-v0 \
    --num_envs=4096 --headless physics=newton_mjwarp --run_name jump_v2_rough 'env.sim.physics.num_substeps=2'
```
观察面板:`Episode_Reward/track_jump_peak_exp`(爬升)、`Metrics/base_velocity/error_jump_peak`(下降)、`track_lin_vel_xy`(守住)。

### 改进 1 第三版(v3,20:10 实现):高度指令替代速度指令

**v2 训练结果与复盘**:8192 envs 训练 188 iter 即收敛(track_jump_peak_exp 1.47/1.5,error 0.03),但 play 目测暴露两个问题:① 速度峰值在低指令段(0.5~0.7 m/s)可被"俯身-快速起身"蹭到,四脚不离地——**速度是加速度的积分,高度才是速度的积分、才是对"跳"的直接定义**;② 可视化误导(箭头随身体俯仰倾斜、绿箭头本来就画实际速度、零长箭头压扁成碟片),曾被误判为"指令混合"——冒烟断言证明指令纯粹性从未被破坏。

**理论起跳高度核算**(决定指令动态范围):蹲伸行程 ~0.25 m,DC 电机力矩-转速衰减曲线 τ(ω)=23.5(1-ω/30) 沿行程积分 → 离地速度 ~2.3-2.4 m/s → 弹道 ~0.28 m + 蹬直增高 0.09 m ≈ **顶点理论增量 0.35-0.38 m**(功率校验通过:8 关节各 ~59 W « 176 W 峰值);RL 现实预期 0.25-0.35 m。

**v3 设计**(v2 的窗口/二分/对称核骨架不变,换测量量):
- 指令 = **顶点高度增量 Δh ∈ (0.20, 0.45) m**(相对标称站高)。下限 0.20 > 起身极限 0.10 的两倍 → 物理强制离地;上限 0.45 超理论天花板 ~30% 作余量档——exp 核(σ=0.1 m)在够不着时仍留 ~0.37 分和活梯度,持续把策略往极限推(E5 教训的反面应用)。
- 测量 = 窗口内**世界系基座高度峰值**,目标 = `env_origins.z + ref_height + Δh`。**参考高度用常数 0.27**(冒烟实测沉降站高 0.266,非 URDF 标称 0.33!)——若用"窗口起始实际高度"会被"先蹲低基准再站起"作弊。
- **armed 机制**(实测中发现的新坑):出生瞬间机器人从高处落下、峰值缓冲区被污染;且跨窗口边界的一跳会被记入两个窗口。修复:窗口开始时峰值停在参考高度、**基座回到地面附近(ref+0.05 m 内)才武装峰值追踪**。
- 走路死区 0.10 m;残差观测/箭头显示改为高度残差 [m];可视化三修:蓝箭头仅偏航旋转(身体俯仰不再画歪纯水平指令)、绿箭头改世界系实际速度、零长箭头整体缩隐(不再有"碟片/球")。
- 损失(PPO)仍零改动;新面板 `Metrics/base_velocity/error_jump_height` [m]。

冒烟通过:flat 50 维/rough 237 维、指令纯粹性断言、出生瞬态未泄漏进峰值、窗口 150 步边界重置正确、3 迭代训练无 NaN。**训练命令与 v2 完全相同**(任务名未变)。play 验收:垂直箭头的狗必须**四脚离地**、箭头越长跳得越高(0.2 小跳 vs 0.45 竭力跳)、跳完箭头消失、3 s 后再跳;若高指令段够不着 0.45 属预期(物理极限),看它是否已竭尽全力。

**flat v3 最终结果(21:00 resume 跑完,共 1004 iter)**:`error_jump_height` **0.0123 m**(≈ 完美策略理论底数 0.014,狗比假设更早起跳)、`track_jump_peak_exp` 1.444/1.5(96%)、`error_vel_xy` 0.096(与纯基线持平,走路零损失)。指标层面已证明真跳——0.012 m 的峰值精度靠 0.1 m 上限的"起身"物理上做不到。

**rough 地形相对高度适配(21:30)**:原 rough 配置的参考高度用格子原点绝对 z,在台阶/斜坡格(占比 60-80%)会**惩罚爬楼**(爬升 0.3 m 冲破死区被当偷跳)且**判死中心外的跳跃**(目标够不着)。修复:峰值测量改为**基座相对局部地面高度**——地面取 height scanner 射线命中 z 的**中位数**(187 根射线,对缺失命中鲁棒,缺失回退按标称地面填充);爬楼时基座与地面同升、相对高度不动,跳跃时地面不动、信号干净。flat 无 scanner,回退 env_origins(精确等价旧行为);rough 走路死区放宽到 0.15 m(吸收跨台阶时地面中位数的波纹)。冒烟:机器人撒到 5 级难度地形上,静置相对站高 0.279±0.016(地形无关性成立)、走路环境浮雕格零误差、flat 回归通过、双任务 3 迭代训练含新 success 指标正常。

**success 指标修正(21:15)**:原 `Metrics/success_rate` 只考核平面阈值(xy<0.5, yaw<0.4),跳跃环境平面指令为零反而"白送"成功——指标对跳跃任务有反向粉饰效果。已改:① 新增 `Metrics/jump_success_rate` = 已完成跳跃窗口中按时交付的比例(窗口结束时 |误差|<0.10 m 算交付,**只统计跳跃窗口**,不被走路环境稀释);② `Metrics/success_rate` 改为**平面阈值 AND 全部跳跃窗口交付**的合取。metrics 是纯记账管道,不进 reward/观测/loss,对收敛零影响。注意:曲线定义变了,新旧 run 的 success_rate 不可直接对比;正在跑的 resume run 不含此指标(代码已加载),下一次启动才生效。早期迭代 episode 活不满 3 s 窗口时 jump_success_rate 不上报,属预期。

**✅ 跳跃任务通关(22:19–23:00)**:两修复(停靠归零+核拓宽)叠加后 `jump_v3_rough_fix2` 从头重训一次通关,1500 iter 共 29 分钟:jump_success_rate **0.98**、error_jump_height 0.017 m、error_vel_xy 0.16(行走无损)、摔倒率 2%、地形课程 6 级。终验对照实验(128 狗强制跳跃):指令-实达相关 **0.99**,难地形(0.7-1.0)与平缓地形交付率均 100%、零摔倒——连续可变跳跃高度的原始需求完整达成。交付模型 `logs/rsl_rl/unitree_go2_jump/2026-07-11_22-18-50_jump_v3_rough_fix2/model_1499.pt`。全程复盘与方法论沉淀见复盘文档 §8.6/§9。

**第二阻塞因素:奖励核感知半径(22:10)**:停靠修复后从头重训的 `jump_v3_rough_fix` 到 iter 301 仍零进展;对照实验二显示狗仍蹲伏(62/64),但峰值这次**如实读到 0.178**——测量已无误,问题在奖励数学:σ=0.1 核在 3σ 外死平,蹲(2.9σ)改站(1.9σ)每步只差 0.04 分(平均奖励的 0.09%),在 rough 优势噪声里不可分辨;flat 能自举是因为狗从不蹲、步态抖动峰值距目标仅 1.2-1.5σ。修复:**σ 0.1→0.2、权重 1.5→3.0**——蹲 0.37/站 1.22/小跳 2.09/全跳 3.00 分每步,爬升全程单调可感知;交付阈值不变,验收不软化。再次从头重训(`jump_v3_rough_fix2`)。详见复盘文档 §8.5。

**rough 首训失败诊断与峰值停靠修复(21:43)**:`jump_v3_rough` 8192 envs 跑到 iter 602,`jump_success_rate` 全程贴零、`error_jump_height` 纹丝不动在 0.078≈0.25×平均指令(即"跳跃环境完全没跳"的理论值;flat 版 300 iter 就有明显进展)。用 model_600 做对照实验(全环境强制跳跃指令+最简单地形+64 狗):狗收到跳跃指令后**集体蹲到 0.134 m 并保持不动**,峰值缓冲区 p90 精确锁死在停靠值 0.270,126 窗口 0 交付、零摔倒。**根因是 v3 设计里的真 bug——峰值停靠在 ref_height(0.27) 造出零梯度口袋**:训练早期台阶上跳跃尝试常摔,策略学到"跳跃指令→蹲低避险";蹲到 0.14 后,任何 +0.13 以内的探索性小跳峰值都够不到 0.27 地板,`max(0.27, rel_h)` 使奖励对动作的梯度在一大片邻域内精确为零,PPO 永远爬不出来(连平地格也不跳了,证明不是"斜坡物理跳不起来")。flat 版没踩坑是侥幸:平地无摔倒恐惧、狗一直站着(0.28>0.27),梯度始终活着。修复:**峰值停靠改为 0(地面)**——峰值恒等于窗口内实际最高点,蹲伏从"奖励不可见"变成"主动扣分"(离目标更远),反蹲伏梯度自动出现;walk 死区(站高 0.28 « 0.42)和 armed 防出生瞬态机制不受影响。冒烟:walk 零误差保持、armed 正常、瘫倒姿态峰值 0.16-0.24 可见(不再被地板遮蔽)。**旧 checkpoint 已被蹲伏行为污染,须从头重训**(命令同前,不要 resume)。

**验收方式**(健身回来后):
1. `git status` + 逐文件 diff 审查代码;
2. 训练曲线:`Episode_Reward/track_lin_vel_z_exp` 是否爬升、`error_vel_z` 是否下降、`track_lin_vel_xy` 是否守住(85%+ 环境还在走路);
3. play + viser:找带上扬箭头的狗,看是否真的在跳、箭头越陡跳得越猛:
```bash
./isaaclab.sh play --rl_library rsl_rl --task Isaac-Velocity-Rough-Unitree-Go2-Jump-Play-v0 \
    --num_envs 8 --headless --real-time physics=newton_mjwarp --visualizer viser
```

---

## 明日(周日)预告:Allegro 转方块——进入你的真正领域

主线任务 `Isaac-Repose-Cube-Allegro-v0`(16 自由度 Allegro 手,掌上把方块转到目标朝向,dexterity 经典 benchmark)。计划要点,细案明早根据今天进度再定:

- **B1 基线训练**:量级再上一档,预计 5090 上 30~60 分钟出可看策略——早上第一件事挂上,吃早饭去。
- **B2 精读**:观测里的物体位姿表示(四元数!正好实地检验 xyzw 约定)、旋转距离奖励、success 容差、掉落终止;对照 Go2 找"手和腿的任务定义差在哪"。
- **B3 实验候选**(选 2 个):success 容差收紧/放宽 → 精度与成功率的权衡;物体尺寸/质量随机化幅度 → 泛化 vs 收敛;指尖 stiffness(接续今天 E4)→ 抓握稳定性。
- **B4(如时间富余)**:按安装指南附录 B 补装 Isaac Sim 6.0.1,跑 `scripts/demos/hands.py` 看官方手部资产陈列,为第 2 期(自定义手 + 物体资产)选型。

## 实验记录区(执行时填写)

| # | run | 改动 | 预测 | 实际(ep_len / 各项reward / 行为) | 预测对了吗 |
|---|---|---|---|---|---|
| 基线 | `baseline`(13:57) | 原版 | — | ep_len 990;success 98.96%;err_vel_xy 0.10;track_lin 1.41/1.5;**feet_air_time -0.008(负!小碎步)**;训练 104s | — |
| E1 | `E1_feet_air`(14:48) | feet_air_time ×4(0.25→1.0) | 步幅变大,track 付代价 | **步态几乎没变**:feet_air_time -0.029 ≈ 基线值×4(行为不变只是税翻倍);track_lin 1.43 反而微升;success 100% | ❌ ×4 仍不足以撬动小碎步策略——策略宁可交 4 倍罚金 |
| E2 | 未做 | action_rate ×5 | | | |
| E3 | `E3_push_robot`(14:55) | push ±0.5→±1.5 | 前期更难,最终更鲁棒 | track_lin 1.39 基本保住;action_rate 支出 -0.074→-0.113(+53%);摔倒率 0.7%→2.9%;success 100% | ✅ 用更多纠正动作买鲁棒性,账单清晰可见 |
| E4 | 未做 | stiffness ×0.5 | | | |
| E5 | `E5_cmd_range`(15:54) | 指令域 ±1.0→±1.5 | 全域略差 or 双模态 | **崩溃**:track_lin 0.15/1.5;err_xy 1.13 ≈ ±1.5 均匀指令模长期望(1.15)→**完全放弃线速度跟踪**,只做转向(err_yaw 0.19 正常)+站稳(97% 超时);action std 0.59 未收敛 | ❌ 出现预测外的第三种结局:exp 核在宽指令域下前期梯度贴零,bootstrap 失败滑入"躺平收税后净收入"局部最优 |

**Rough 地形系列(改进实验 2,16:04~16:45)**:

| run | 配置 | 跑到 | success / err_xy / 地形等级 | 结局 |
|---|---|---|---|---|
| `rough_baseline` | newton, substeps=1, 4096 | iter 196 | 64% / 0.41 / 1.3 | **NaN 崩溃**(物理求解器发散,随机事件) |
| `rough_resume` | 从 model_150 续,同上 | iter 753(手动停) | **99.3%** / 0.18 / **5.8** | 平安,说明 NaN 是低概率随机 |
| `rough_substeps2` | substeps=2, 4096 | iter 618(手动停) | 96.6% / 0.21 / 6.0 | 平安越过 196 坎 |
| `rough_substeps2` | substeps=2, **8192** | iter 302(手动停) | 98.1% / 0.17 / 3.6 | **302 iter ≈ 4096 版 600 iter 的水平**,大 batch 增益实测 |

## 遗留问题(滚动)

- [x] 基线训练完成(13:57,结果见上下文快照;`--video` 未随基线测试,仍待验证)
- [ ] `--video` kit-less 可用性(A0 验证)
- [ ] cartpole `[rad]` 注释 issue 是否已提交上游
- [x] Hydra 覆盖 manager 项的实际路径语法 → **已实测写进速查:标量直接写;dict/tuple 值必须是合法 Python 字面量(键带引号),否则静默变字符串、事件首次触发时才炸(E3 踩坑实录见执行记录)**
- [ ] `newton_kamino` 对比(接触密集任务已到:rough 地形上 newton_mjwarp 会低概率 NaN 崩溃,substeps=2 可缓解;kamino 对比值得做)
- [ ] **PhysX 后端本机不可用**:kit-less 安装没有 Isaac Sim,`physics=physx` 直接报错——装 Isaac Sim 6.0.1(明日 B4)后补 Newton vs PhysX 接触密集对比
- [ ] E2(action_rate ×5)、E4(stiffness ×0.5)未做——E4 仍是下周灵巧手最重要预习
- [ ] E5 抢救实验:std 放宽 / 指令课程 / 拉长迭代,验证 bootstrap 失败假设
- [x] `go2_jump` 跳跃任务验收 → **通关**(jump_success 0.98,指令-实达相关 0.99,难易地形一致;详见 [go2-jump-task-retrospective.md](go2-jump-task-retrospective.md));play 目测验收留给周日早上
