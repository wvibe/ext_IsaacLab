# 周末学习 · 第 1 期周日 —— Allegro 转方块：手内操作三要素入门

> 承接 `weekend1-afternoon-go2-plan.md`（周六 Go2）。主线任务：`Isaac-Repose-Cube-Allegro-v0`（名字以 list_envs 实际输出为准）。
> **任务本质**：16 自由度 Allegro 手掌心朝上托一个方块，把它转到随机目标朝向；转到即刷新目标继续转，方块掉落则重置。DeXtreme sim-to-real 工作的直系任务，dexterity 经典 benchmark。
> **今天只学三件新事**：① 物体位姿进观测（四元数实战）② 旋转距离奖励 ③ 掉落终止与 success 容差。手臂、视觉、ADR、LSTM 全都不在今天范围。
> **与安装线的咬合**：本文档 Session B0 的冒烟测试决定 Allegro 走 Newton 还是 PhysX。Isaac Sim 下载（安装指南 Step 1）现在就挂上，与 B0 并行。
> **时间预算**：约 5~6 小时。基线训练 30~60 分钟出可看策略（5090 + 4096 envs），节奏延续"后台挂训练、前台读代码"。

---

## Session B0 · 冒烟测试与后端定案（约 20 分钟，决定今天走哪条路）

```bash
cd ~/vibe/sim/IsaacLab3
source .venv/bin/activate

# 0. 安装指南 Step 0 + Step 1 先挂上（另开终端，纯下载不占脑子）

# 1. 确认任务名，顺手把五档 manipulation 候选的可用性一次查清，存档
./isaaclab.sh -p scripts/environments/list_envs.py | grep -i "allegro\|shadow\|dexsuite\|factory\|lift" | tee -a plan/task-list.txt

# 2. Newton 后端冒烟测试（小规模，2 分钟内见分晓）
./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Repose-Cube-Allegro-v0 \
    --num_envs=256 --max_iterations=10 --headless physics=newton_mjwarp
```

**✅ 验收与分叉**：
- **能跑** → 今天全程 Newton，Isaac Sim 装完只做 Cartpole 对照（安装指南 Step 5），不打断主线。继续 B1。
- **报错"后端不支持"或行为异常**（方块穿模、手指抖散等）→ 记录报错/现象原文，Allegro 改走 PhysX：先按安装指南走完 Step 1-4，再回到 B1 并把命令里的 `physics=newton_mjwarp` 换成实测可用的 PhysX 语法（安装指南 Step 5 的待验证点，在这里顺手定案）。
- 两种结果都不亏：后者等于把"求解器对照"提前变成刚需，实测数据照记。

**故障排查**：
- 任务名对不上 → 以 list_envs 输出为准，beta2 重构过注册名；`grep -rn "Repose" source/isaaclab_tasks/ -l` 直接定位源码。
- 冒烟测试 OOM → 256 envs 不该 OOM，检查是否有僵尸训练进程占显存（`nvidia-smi`）。

---

## Session B1 · 基线训练挂后台（5 分钟操作，30~60 分钟等待）

```bash
# 正式基线（后端按 B0 结论；iter 数先给 2000，曲线平了可提前杀）
./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Repose-Cube-Allegro-v0 \
    --num_envs=4096 --max_iterations=2000 --headless physics=newton_mjwarp

# TensorBoard 照旧（IDE 自动转发 6006）
./isaaclab.sh -p -m tensorboard.main --logdir logs/ --port 6006
```

挂上就去 B2，别盯曲线。

**✅ 验收**：训练启动无报错，TensorBoard 出现新 run。**顺手记录 steps/s**——与 Cartpole（~106 万）和 Go2 的吞吐并排写进实验记录，三个任务复杂度的量化对比自己会说话。

**预测练习（开训前写下）**：你预计 Allegro 的 steps/s 比 Go2 高还是低？大概差几倍？（提示：自由度 16 vs 12，但接触对数完全不是一个量级。）

---

## Session B2 · 精读源码：四个问题（约 1.5~2 小时，基线挂着时做，今天含金量最高的环节）

先定位：

```bash
grep -rn "Repose-Cube-Allegro" source/isaaclab_tasks/ -l
# 顺 entry_point 打开，direct 任务，典型位置 source/isaaclab_tasks/isaaclab_tasks/direct/allegro_hand/
# 大概率与 shadow_hand 共享 in-hand 基类，把基类也打开
```

**问题 1 · 非对称 actor-critic 的证据（昨天 RL 回顾的思考题，今天兑现）**
在观测定义里找：critic（或 states/privileged 组）是否比 policy 组多看了东西？典型的特权信息：物体真实线速度/角速度、指尖接触力——真机上拿不到或很贵的量。
- 若有 → 记下两组各自的维度和内容清单，想一句话：为什么这几个量给 critic 无害、给 actor 就毁 sim-to-real？
- 若这个任务没用非对称 → 也记下来，去 Shadow OpenAI 变体的 cfg 里找对照（它是满配版），这就是下次学 Shadow 的钩子。

**问题 2 · 四元数实战：旋转距离奖励怎么算**
- 找到物体朝向进观测的地方，确认 xyzw 顺序（3.0 新约定，第一次真刀真枪出场）。
- 追旋转奖励的实现：目标朝向与当前朝向的"距离"用的是什么？（典型：四元数差取角度 `2*acos(|q_diff.w|)`，或 rot_dist 形式。）注意 w 分量在 xyzw 排布里的索引位置——这是约定切换最容易埋 bug 的地方。
- 想一步：这套"目标朝向差"的奖励设计，换成柔性物体为什么会失效？（柔性体没有单一刚体位姿——这个问题的答案就是你第 3 期要解决的核心问题之一，今天只需要意识到它存在。）

**问题 3 · 账本框架上手：success 容差与掉落终止**
- success 阈值多少弧度？达成后是重置还是原地刷新目标（consecutive successes）？
- 掉落怎么判？（典型：方块与手的距离超阈值。）掉落罚多大？
- 按你 Cartpole 时期的账本框架分类：主账（旋转距离 shaping）、奖金（success bonus）、罚金（掉落/超时）各是谁、量级各多少。写成三行笔记。

**问题 4 · 头号超参露真容：手指的 ActuatorCfg**
- Allegro 手指关节的 stiffness/damping/effort_limit 是多少？与昨天 Go2 腿部的数值差几个数量级？
- Implicit 还是显式执行器模型？
- 把两组数并排记下——**这张对照表就是你研究目标头号超参的第一份手感数据**，E4（Go2 腿）与今天 C2（Allegro 指）两个实验共用它做参照系。

**✅ B2 验收**：向 IDE Agent 口述"方块当前朝向从物理引擎到旋转奖励标量的完整数据路径"（引擎四元数 → 观测拼接 → 奖励函数里的距离计算），让它挑错。四个问题各留下 3 行以内的笔记。

---

## Session B3 · 基线验收与观赏（约 30 分钟）

```bash
# 曲线：重点看 consecutive_successes（或等价指标）是否爬升、episode 长度是否变长
# viser 观赏（沿用你验证过的工作流）
./isaaclab.sh play --rl_library rsl_rl --task=Isaac-Repose-Cube-Allegro-v0 \
    --num_envs=16 physics=newton_mjwarp --visualizer viser
```

看三样东西：手指是"协同滚动"方块还是"抛接投机"；接近目标朝向时是减速微调还是冲过头再折返；掉落前有没有可辨认的前兆姿态。

**✅ 验收**：基线策略能连续完成多次重定向（偶尔掉落正常）。留基线 checkpoint，logs 目录改名 `allegro_baseline`。

**故障排查**：
- 训了 2000 iter 还是乱抖不成功 → 先怀疑后端（回 B0 分叉重审），再怀疑 iter 不够（这任务比 Go2 慢热，可再给 2000）。
- viser 里方块渲染不出 → 记录现象，改用 `--visualizer newton`（remote desktop）确认是可视化问题还是物理问题。

---

## Session B4 · 两个实验（约 2 小时，从三个候选里选二）

沿用格式：改动 → **先写预测** → 训练（与基线同 iter）→ 曲线 + viser 验证 → 记录。direct 任务的 Hydra 覆盖路径形如 `env.<字段名>=...`（比 manager-based 浅，具体字段名以 B2 读到的 cfg 为准；报错列出的合法键照旧是免费文档）。

**实验 C1 · success 容差收紧/放宽（±一倍）**
预测练习：容差收紧一半，成功率降多少？策略会变得更"谨慎"（末段减速）还是不变？放宽一倍呢——会不会学出更潦草的转法？
学习点：精度与成功率的权衡曲线。抓取任务里"抓稳的判定阈值"是同一个设计决策。

**实验 C2 · 指尖 stiffness 减半（接 Go2 E4 的剧情）**
预测练习：昨天 Go2 腿 stiffness 减半的结果是___（抄你的实验记录）；同样的改动放到手指上，你预计方块掉落率变化___、成功间隔变化___。手指比腿更敏感还是更不敏感？为什么？
学习点：头号超参在真正目标肢体上的第一次实测。**做完把 Go2-E4 与 Allegro-C2 写成一张跨肢体对照小表**——这张表是本周末"腿→手迁移"设计原则的第一份验证数据。

**实验 C3 ·（备选）物体尺寸/质量随机化幅度加大**
预测练习：随机化范围扩大 50%，收敛慢多少？最终策略在标准方块上会不会反而变差（泛化税）？
学习点：域随机化的宽度 vs 单点性能——第 3 期"不同软硬物体"域随机化的刚体预演。

**选择建议**：C2 必做（它是主线剧情），C1 和 C3 二选一；若 B0 走了 PhysX 分叉导致时间紧张，只做 C2。

**✅ B4 验收**：实验表写入本文档末尾，"预测 vs 实际"两栏齐全，预测错的行单独标注。

---

## Session B5 · 收尾与第 1 期毕业（约 30 分钟)

1. 最满意的 Allegro checkpoint 用 viser 录屏，与周六的 Go2 录屏并列——第 1 期两件毕业作品。
2. 让 Agent 汇总一张**第 1 期总表**：Cartpole / Go2 / Allegro 三个任务的 steps/s、收敛 iter、观测维度、动作维度、你踩的最大的坑各一行。
3. 三句话总结照旧：最反直觉的发现 / 最没搞懂的概念 / 第 2 期最想做的事。
4. 预览第 2 期：只读不跑，看 DexSuite（若 list_envs 里有）或 Shadow-OpenAI-LSTM 的 cfg 20 分钟，列"看不懂的名词清单"。
5. 若 Isaac Sim 已装完：remote desktop 跑 `./isaaclab.sh -p scripts/demos/hands.py`，给第 2 期资产选型攒第一印象（安装指南 Step 6）。

---

## 实验记录区（执行时填写）

> 进度总档见 [weekend1-sunday-allegro-progress.md](weekend1-sunday-allegro-progress.md)（2026-07-12 停笔：B0–B1 + B2 问题 1–3 已完成；B4/B5 未做）。

| # | run | 改动 | 预测 | 实际（succ率 / 掉落率 / 各项reward / 行为） | 预测对了吗 |
|---|---|---|---|---|---|
| 基线 | `..._allegro_baseline` + `..._resume` | 原版 Direct+Newton | steps/s 低于 Go2 | ~105k steps/s；iter~2736 时 succ~93%、consec~5.3 | ✅ |
| C1 | | success 容差 ___ | | | |
| C2 | | 指 stiffness ×0.5 | | | |
| C3 | | 物体随机化 +50% | | | |

**跨肢体对照表（C2 做完填）**：

| | Go2 腿（E4） | Allegro 指（C2） |
|---|---|---|
| 默认 stiffness / damping | 25.0 / 0.5（DCMotor） | 3.0 / 0.1（Implicit）——已抄 cfg，实验未做 |
| 减半后主要行为变化 | | |
| 任务指标损失 | | |
| 结论：手比腿更敏感吗 | | |

## 遗留问题（滚动，继承周六清单并追加）

- [x] Allegro 在 beta2 的 Newton 支持状态（B0 定案）→ Direct + newton_mjwarp
- [x] beta2 下 PhysX 的正确 CLI 语法 → `physics=physx`
- [x] 本任务是否用了非对称 AC → 否；Shadow OpenAI 变体对照（B2 问题 1）
- [x] B2 问题 2–3 旋转距离 + 账本（见 progress §7）
- [ ] B2 问题 4 Actuator 热身 + B4-C2；C1/C3；B3 精看；B5
- [ ] 柔性物体的"朝向/位姿"如何定义——刚体四元数奖励的失效点（第 3 期核心问题，今天只登记不展开)
- [ ] --video 在 kit-less 下的可用性（继承自周六，若周六已定案则划掉）
- [ ] L72 `goal_rot[:,0]=1.0` 疑似 wxyz 遗留（低优先级）

## 本期明确不碰的内容（防发散）

Shadow Hand 全系（下次专场）、DexSuite（第 2 期主线）、手臂 IK / 05_controllers（第 2 期）、视觉观测、LSTM/记忆策略、ADR/PBT、柔性体。今天的敌人依然是贪多。
