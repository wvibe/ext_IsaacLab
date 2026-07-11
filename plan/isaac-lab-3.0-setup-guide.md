# Isaac Lab 3.0 (Beta 2) Ubuntu 环境配置指南(修订版 v2)

> **目标机器**:Ubuntu 24.04.4 + RTX 5090 32GB + Ryzen 9 9900X + 32GB RAM(主机名 Labr24)
> **目标版本**:Isaac Lab `release/3.0.0-beta2` + Newton `1.2.1`(随 Lab 自动安装)+ PyTorch 2.10.0 (cu128)
> **安装位置**:`~/vibe/sim/IsaacLab3`(源码已 clone),venv 为仓库内 `env_isaaclab`
> **明确不装**:Isaac Sim(周日视情况按附录 B 补装 6.0.1)、Docker、CUDA Toolkit(torch wheel 自带运行时)
> **修订说明**:v2 由执行 Agent 根据 2026-07-11 上午的实际执行情况修订。原 v1 的阶段 0~3 已完成,阶段 4 完成过 90%(后因目录搬迁失效需返工);修正了 Newton 版本号、任务名验证方式、国内网络策略。

---

## 给执行 Agent 的总体指令

1. **严格按阶段顺序执行**,每阶段结束运行验收命令确认通过后再继续;失败先查「⚠️ 排障」,不要盲目重试。
2. **网络策略(本机为墙内网络,这是 v1 最大的缺口)**:
   - 本机代理为 clash-verge,mixed-port `7897`,shell 环境变量 `http_proxy`/`https_proxy` 默认已指向它。
   - **所有安装命令保持代理环境变量,不要 unset**。GitHub 直连时通时断,必须走代理。
   - 每次开始联网操作前先验证代理存活:`curl -s --max-time 5 -x http://127.0.0.1:7897 -o /dev/null -w "%{http_code}" https://www.google.com` 应返回 2xx/3xx。
   - **已知问题**:2026-07-11 开机时 mihomo 报 `bind: address already in use`,代理假死(环境变量在、端口没监听)。若复发:请用户在 clash-verge 界面重启内核,或经用户批准后通过控制接口 `curl --unix-socket /tmp/verge/verge-mihomo.sock -X PATCH http://localhost/configs -d '{"mixed-port": 7897}'` 重绑端口。
3. **不要复用任何旧环境**:本机存在 Isaac Lab 2.3 旧克隆(`~/vibe/sim/IsaacLab`)、conda `isa` 环境(Isaac Sim 5.1 + isaaclab 0.53,关联 `~/rbt/IsaacLab`)、`~/docker` 下的旧启动脚本。全部保持原样不动。
4. **环境管理器用 uv venv,不用 conda**(用户已确认)。miniconda 环境留给旧项目。
5. **不要升级或改动系统级组件**。驱动已是 595.71.05(Isaac Sim 6.0.1 的验证分支),**不要再动驱动**。
6. 下载量大、首次运行要 JIT 编译 warp kernel 并从 AWS S3 拉资产,长时间无输出属正常,耐心等待。
7. 遇到本文档未覆盖的错误:保留完整报错,优先在 https://github.com/isaac-sim/IsaacLab/issues 检索,不要自行修改 Lab 源码。

---

## 当前状态快照(2026-07-11 10:10,执行 Agent 填写)

| 项目 | 状态 |
|---|---|
| NVIDIA 驱动 | ✅ 595.71.05(open 内核模块,Ubuntu 官方源),CUDA 13.2,重启后验证正常 |
| 源码 | ✅ `~/vibe/sim/IsaacLab3`,分支 `release/3.0.0-beta2`,commit `c5b042d657`(2026-07-10) |
| uv | ✅ 已安装(`~/.local/bin/uv`) |
| venv `env_isaaclab` | ❌ **失效**——创建于搬迁前的 `~/vibe/IsaacLab3`,脚本与 editable 安装内嵌的绝对路径已断,需删除重建 |
| torch 2.10.0+cu128 | 曾装入旧 venv 并验证 CUDA 可用;wheel 已在 uv 缓存,重装很快 |
| 15 个 isaaclab 子包 | 曾装完 14 个(editable,搬迁后失效);`isaaclab_mimic` 因 robomimic 走 GitHub 直连被墙失败 |
| extras(newton/rl/visualizer) | ❌ 未安装(v1 执行时因 mimic 失败中断) |
| 代理 | ✅ 7897 正常监听并可出站(10:00 修复) |
| 磁盘 | ✅ 713GB 可用 |

**结论:从阶段 2 重新开始,阶段 0/1 无需重复。**

---

## 阶段 0:系统前置检查 —— ✅ 已完成(2026-07-11)

记录:Ubuntu 24.04.4 / 内核 6.17.0-35 / GLIBC 2.39 / 驱动 595.71.05 / RTX 5090 32GB / 磁盘 713GB 可用 / git 正常。全部通过,不再重复执行。

---

## 阶段 1:安装 uv 并克隆 Isaac Lab 3.0 —— ✅ 已完成(2026-07-11)

记录:uv 已在 PATH;源码位于 `~/vibe/sim/IsaacLab3`,分支 `release/3.0.0-beta2`,commit `c5b042d657`。不再重复执行。

---

## 阶段 2:重建 Python 3.12 虚拟环境

Python 版本**必须是 3.12**(Isaac Sim 6.X / Lab 3.0 硬性绑定)。因目录搬迁,旧 venv 必须删除重建:

```bash
cd ~/vibe/sim/IsaacLab3

# 2.1 删除失效的旧 venv
rm -rf env_isaaclab

# 2.2 重建(--seed 保证环境内有 pip,Lab 安装器依赖它;uv 已缓存 cpython 3.12.11,秒级完成)
uv venv --python 3.12 --seed env_isaaclab

# 2.3 激活并确认
source env_isaaclab/bin/activate
python --version        # 期望 Python 3.12.x
which python            # 期望 ~/vibe/sim/IsaacLab3/env_isaaclab/bin/python
uv pip install --upgrade pip
```

**✅ 验收标准**

- [ ] `python --version` 为 3.12.x 且路径位于 `~/vibe/sim/IsaacLab3/env_isaaclab` 内

---

## 阶段 3:安装 Isaac Lab 全部扩展(含 torch 与 Newton 1.2.1)

> v1 把 torch 单列为一个阶段;实测 `./isaaclab.sh -i` 会自动安装官方指定的 `torch==2.10.0 + torchvision==0.25.0 (cu128)`,无需手动预装,故合并。
> Newton 以 `newton[sim]==1.2.1` 作为 `isaaclab_newton` 的依赖自动装入(v1 写的 "Newton 1.0" 是错的),**不要单独 pip install newton**。

```bash
cd ~/vibe/sim/IsaacLab3
source env_isaaclab/bin/activate

# 3.1 确认代理存活(见总体指令 2;不通则先修代理,不要 unset 直连硬跑)
curl -s --max-time 5 -x http://127.0.0.1:7897 -o /dev/null -w "%{http_code}\n" https://www.google.com

# 3.2 一键安装:核心包 + mimic/teleop 子模块 + newton/rl/visualizer extras
# 大部分 wheel 已在 uv 缓存,预计 5~15 分钟;robomimic 需经代理从 GitHub clone
./isaaclab.sh -i

# 3.3 核对关键包
uv pip list | grep -iE "^(isaaclab|newton|mujoco|warp-lang|rsl|torch)"
```

**✅ 验收标准**

- [ ] `./isaaclab.sh -i` 以退出码 0 结束
- [ ] 包列表含 `isaaclab`、`isaaclab-newton`、`newton`(1.2.1)、`warp-lang`、`rsl-rl-lib`、`torch`(2.10.0+cu128)
- [ ] GPU 验证通过:

```bash
python -c "
import torch, warp, newton
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device='cuda') @ torch.randn(1024, 1024, device='cuda')
print('matmul OK:', x.shape, '| warp:', warp.__version__, '| newton:', newton.__version__)
"
```

**⚠️ 排障**

- **robomimic 下载失败(`isaaclab_mimic` 步骤)**:v1 执行时的实际卡点。确认代理存活后重跑 `./isaaclab.sh -i`(幂等)。仍失败可单独重试:`uv pip install --editable source/isaaclab_mimic`。本机 `~/vibe/robo/robomimic` 有一份本地 clone,可作最后兜底(需核对 tag v0.4.0)。
- **pypi.nvidia.com / download.pytorch.org 连接被拒**:代理又假死了,按总体指令 2 修复,不要切直连。
- **`pip check` 报依赖冲突**(coverage/packaging/numpy 等):beta2 已知现象(上游 issue #6200),不阻塞运行。**不要为消除警告手动改包版本。**
- 安装中途网络超时:直接重跑 `./isaaclab.sh -i`。

---

## 阶段 4:冒烟测试

```bash
cd ~/vibe/sim/IsaacLab3
source env_isaaclab/bin/activate

# 4.1 先列出实际注册的任务名(v1 引用的 *-Warp-v0 命名已过时;beta2 通过
#     physics= 参数选后端,任务名需以本机实际输出为准)
./isaaclab.sh -p scripts/environments/list_envs.py 2>/dev/null | grep -i cartpole \
  || ./isaaclab.sh -p -c "import isaaclab_tasks, gymnasium as gym; [print(k) for k in gym.registry.keys() if 'Cartpole' in k]"

# 4.2 官方冒烟命令:kit-less + Newton MJWarp 后端 + Newton 可视化器,小规模快速验证
#     (来自 beta2 官方 kitless 安装文档;首次运行 JIT 编译 warp kernel + 拉取资产,慢属正常)
./isaaclab.sh train --rl_library rsl_rl \
    --task=Isaac-Cartpole-Direct-v0 \
    --num_envs=16 --max_iterations=10 \
    physics=newton_mjwarp --visualizer newton
```

> 说明:机器在脚边,4.2 带 `--visualizer newton` 可在显示器/remote desktop 上看到实时画面;若想纯 headless,去掉 `--visualizer` 参数。

**✅ 验收标准**

- [ ] 任务列表能正常打印(记录 Cartpole/Anymal 相关的实际任务名,写入安装记录)
- [ ] 4.2 训练正常步进 10 个 iteration 退出,无未捕获异常

**⚠️ 排障**

- 资产下载慢/卡住:资产在 AWS S3,确认代理存活;长期方案见附录 A 资产缓存。
- `ModuleNotFoundError: No module named 'isaaclab'`:venv 未激活。
- `./isaaclab.sh train` 子命令不存在:退回 `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py ...` 传统方式,参数不变。

---

## 阶段 5:真实训练验证(GPU 吞吐测试 + Newton 路径验证)

```bash
cd ~/vibe/sim/IsaacLab3
source env_isaaclab/bin/activate

# 5.1 Cartpole 4096 并行环境 headless 训练(任务名以阶段 4.1 实际输出为准)
./isaaclab.sh train --rl_library rsl_rl \
    --task=Isaac-Cartpole-Direct-v0 \
    --num_envs=4096 --headless \
    physics=newton_mjwarp

# 5.2 (另开终端)确认 GPU 在干活
watch -n 2 nvidia-smi

# 5.3 换一个 locomotion 任务(如 Anymal/Go2 平地行走,任务名以 4.1 输出为准),训练 5 分钟即可停
# 5.4 TensorBoard 看曲线(日志默认在 logs/)
./isaaclab.sh -p -m tensorboard.main --logdir logs/   # 浏览器打开 http://localhost:6006
```

**✅ 验收标准**

- [ ] iteration 持续推进,mean reward 上升
- [ ] GPU 利用率显著(通常 >50%)
- [ ] TensorBoard 正常显示曲线

**⚠️ 排障**

- 首个 iteration 前长时间停顿:warp kernel JIT 编译,一次性成本。
- 吞吐低且 GPU 利用率低:确认 `--headless`;确认无其他进程占 GPU。

---

## 阶段 6:环境固化与记录

```bash
cd ~/vibe/sim/IsaacLab3
source env_isaaclab/bin/activate
uv pip freeze > plan/env-snapshot-$(date +%Y%m%d).txt
git log --oneline -1; python --version
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

填写文档末尾「安装记录」表。**(可选)**向 `~/.bashrc` 追加别名:

```bash
alias lab3='cd ~/vibe/sim/IsaacLab3 && source env_isaaclab/bin/activate'
```

---

## 附录 A:资产本地缓存(强烈建议,墙内网络必做)

Isaac Lab 资产托管在 AWS S3,按需拉取。开启本地缓存后落盘复用,配置方式以官方文档为准:
`https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/asset_caching.html`

**⚠️ 实测坑(2026-07-11)**:S3(`omniverse-content-production.s3-us-west-2.amazonaws.com`)**走本机代理会失败**(连接超时),直连反而正常——与 GitHub 恰好相反。运行任何需要拉资产的命令前,把 AWS 域名加入 no_proxy:

```bash
export no_proxy="$no_proxy,amazonaws.com" NO_PROXY="$NO_PROXY,amazonaws.com"
```

建议将其固化到 `~/.bashrc`(或写进 lab3 别名),规则:**装依赖挂代理,拉资产直连**。

## 附录 A2:SSH / 远程可视化(实测可用)

三个可视化器的远程能力:

| 可视化器 | 原理 | 远程可看? |
|---|---|---|
| `--visualizer newton` | 本地 OpenGL 窗口 | ❌ 需要显示器 / remote desktop |
| `--visualizer viser` | 网页服务,绑定 `0.0.0.0:8080` | ✅ 浏览器打开 `http://192.168.1.220:8080` |
| `--visualizer rerun` | 网页 + gRPC 流(9090/9876) | ✅ 同样支持浏览器远程 |

回放训练好的策略并远程观看(play 自动加载 `logs/rsl_rl/<task>/` 下最新 checkpoint):

```bash
./isaaclab.sh play --rl_library rsl_rl --task Isaac-Cartpole-Direct-v0 \
    --num_envs 16 --headless physics=newton_mjwarp --visualizer viser
# Mac 浏览器打开 http://192.168.1.220:8080;结束:pkill -f "play.py --rl_library"
```

## 附录 B:周日补装 Isaac Sim 6.0.1(现在不要执行)

进入「GUI 场景编辑 / URDF 导入 / Gains Tuner / PhysX 后端对比」学习阶段时,在**同一 venv**中执行(约 15GB 下载,挂代理):

```bash
source ~/vibe/sim/IsaacLab3/env_isaaclab/bin/activate
uv pip install "isaacsim[all,extscache]==6.0.1" --extra-index-url https://pypi.nvidia.com
isaacsim   # 首次启动编译 shader 较慢;需在显示器或 remote desktop 上操作
```

> 驱动已是 595.71.05(6.0.1 的验证分支),无需再动系统。

## 附录 C:「不要做」清单(护栏)

1. 不要在旧的 Isaac Lab 2.3 目录(`~/vibe/sim/IsaacLab`)里执行任何本文档命令。
2. 不要动系统 Python、不要 `sudo pip install`、不要再动 NVIDIA 驱动。
3. 不要手动安装 CUDA Toolkit / cuDNN。
4. 不要单独 `pip install newton` 或 `mujoco`——版本由 `./isaaclab.sh -i` 统一管理。
5. 不要为消除 `pip check` 警告手动改包版本。
6. 不要 `git pull` 到 develop 最新——学习期锁定 beta2。
7. 不要用 Docker 方式安装(仅作 pip 路线彻底失败的兜底,需用户批准)。
8. 不要在安装命令里 unset 代理环境变量(v1 执行中吃过的亏:直连 GitHub 被墙)。
9. 再次搬迁仓库目录前先告知——venv 内嵌绝对路径,搬迁即失效。

---

## 安装记录(由执行 Agent 填写)

| 项目 | 值 |
|---|---|
| 安装日期 | 2026-07-11(10:30 完成) |
| Lab 分支 / commit | `release/3.0.0-beta2` / `c5b042d657` |
| Python 版本 | 3.12.11(uv 托管) |
| torch 版本 | 2.10.0+cu128 ✅ |
| 关键包版本 | newton 1.2.1 / mujoco-warp 3.8.1 / warp-lang 1.13.0 / rsl-rl-lib 5.0.1 / skrl 2.1.0 / rerun-sdk 0.34.1 |
| NVIDIA 驱动版本 | 595.71.05(open),warp 识别 5090 为 sm_120 |
| 阶段 4 冒烟测试 | ✅ 通过(Cartpole 16 envs × 10 iter,headless + newton_mjwarp,无异常) |
| 阶段 5 训练验证 | ✅ 通过(Cartpole 4096 envs × 150 iter,**~106 万 steps/s**,reward 5.3→296 收敛,训练仅 9.7 秒) |
| 可视化验证 | ✅ 通过(用户在 remote desktop 以 `--visualizer newton` 跑 16 envs × 200 iter,2756 steps/s,reward 255、success_rate 1.0,倒立摆稳定站立) |
| 依赖快照 | `plan/env-snapshot-20260711.txt`(227 个包) |
| 训练日志 | `logs/rsl_rl/cartpole_direct/` |
| 遇到的问题与解决方式 | ① 开机 mihomo 端口绑定失败致代理假死→控制接口重绑 7897;② robomimic 直连 GitHub 被墙→改为全程挂代理;③ 仓库搬迁至 `~/vibe/sim/` 致 venv 失效→重建(阶段 2) |

---

## 附录 D:实际执行时间线(2026-07-11 上午,完整动作序列)

| 时间 | 动作 | 结果 |
|---|---|---|
| 09:27 | 系统盘点:Ubuntu 24.04.4 / 内核 6.17 / 驱动 580.159.03 / RTX 5090 / 32GB RAM / 713GB 磁盘;发现旧环境(Lab 2.3 源码、conda `isa` = Isaac Sim 5.1、docker 镜像已删) | 基线确认 |
| 09:30 | 调研发布状态:Isaac Lab 3.0.0-beta2(2026-06)+ Newton;Isaac Sim 6.0.1 GA;6.0.1 验证驱动为 595 分支;kit-less 模式可完全不装 Isaac Sim | 确定技术路线 |
| 09:37 | 用户执行 `sudo apt install nvidia-driver-595-open`(Ubuntu 官方源 595.71.05,顺带清除残留的 575/580) | DKMS 对 6.17 内核编译成功 |
| 09:41 | 重启前检查:`dkms status` installed、Secure Boot 关闭 | 确认可安全重启 |
| 09:46 | 重启,验证驱动 595.71.05 / CUDA 13.2 生效,旧 conda 环境 torch 仍正常(驱动向下兼容) | ✅ |
| 09:47 | clone `release/3.0.0-beta2` 至 `~/vibe/IsaacLab3`(当时的位置) | commit `c5b042d657` |
| 09:50 | 建 venv,第一次 `./isaaclab.sh --install` | ❌ 失败:代理假死(开机时 mihomo 报 `bind: address already in use`,7897 无监听而环境变量仍指向它) |
| 09:55 | unset 代理重试安装 | 部分成功:torch 2.10.0+cu128 与 14 个子包装入;`isaaclab_mimic` 因 robomimic 需直连 GitHub 被墙失败,extras 未装 |
| 10:00 | 诊断代理:定位 mihomo 端口绑定失败,经控制接口 PATCH `mixed-port` 重绑 7897 | ✅ 代理恢复 |
| 10:05 | 用户整理目录:仓库迁至 `~/vibe/sim/IsaacLab3` | venv 内嵌绝对路径全部失效 |
| 10:12 | 依据实际情况将本计划修订为 v2,用户审阅批准 | — |
| 10:20 | 阶段 2:删除旧 venv 重建(uv,Python 3.12.11) | ✅ 秒级完成 |
| 10:21 | 阶段 3:`./isaaclab.sh -i` 挂代理完整安装 | ✅ 一次通过,约 5 分钟,退出码 0 |
| 10:27 | 验收:15 个子包 + newton 1.2.1 + mujoco-warp 3.8.1 + warp 1.13.0(识别 sm_120)+ RL 框架四件套 + 三个可视化器;GPU matmul 正常;确认任务名 `Isaac-Cartpole-Direct-v0` 存在(`-Warp-v0` 旧命名已废弃) | ✅ |
| 10:28 | 阶段 4 冒烟:Cartpole 16 envs × 10 iter(headless,newton_mjwarp),首跑含 warp JIT 约 1 分钟 | ✅ |
| 10:29 | 阶段 5:Cartpole 4096 envs × 150 iter | ✅ ~106 万 steps/s,reward 5.3→296,训练 9.7 秒 |
| 10:30 | 用户在 remote desktop 跑 `--visualizer newton` 版本(16 envs × 200 iter) | ✅ 2756 steps/s,success_rate 1.0,肉眼确认倒立摆站稳 |
| 10:32 | 阶段 6:依赖快照 `plan/env-snapshot-20260711.txt`,填写安装记录与本时间线 | 收官 |
