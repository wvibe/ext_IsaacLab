# Isaac Sim 6.0 + PhysX 后端补装指南（Isaac Lab 3.0 beta2 · kit-less 环境升级）

> **状态：✅ 已于 2026-07-12 执行完成**（安装 isaacsim 6.0.1.0，PhysX 训练验证通过）。执行中与原稿不一致处已就地修订，完整执行记录见文末「实际执行记录」。
> 目标：在现有 kit-less + Newton 环境（`~/vibe/sim/IsaacLab3`，uv venv，Python 3.12，torch 2.10.0 cu128）上**原地补装** Isaac Sim 6.0 pip 包与 `isaaclab_physx` 后端，全程不破坏已验证的 Newton 工作流。
> 为什么装：① PhysX FEM 柔性体（第 3 期核心，Newton VBD 之外的对照路线）② URDF/MJCF 导入器与 GUI 资产工具链（第 2 期自定义手/物体资产）③ PhysX vs Newton 求解器对照实验 ④ `scripts/demos/hands.py` 官方手部资产陈列。
> 预算：下载 ~10-15 GB（isaacsim[all,extscache]），首次启动拉扩展约 10 分钟。建议磁盘预留 ≥ 30 GB。
> 官方依据：Isaac Lab v3.0.0-beta 文档 "Installation using Isaac Sim Pip Package"（2026-06-18 更新）。

---

## Step 0 · 前置确认与回滚快照（必做，5 分钟）

```bash
cd ~/vibe/sim/IsaacLab3
source env_isaaclab/bin/activate   # 修订：实际 venv 是仓库内 env_isaaclab，不是 .venv

python -V                          # 必须 3.12.x（Isaac Sim 6.X 硬性要求）
ldd --version | head -1            # 需 GLIBC >= 2.35；Ubuntu 24.04 自带 2.39，应通过
df -h ~ | tail -1                  # 可用空间 >= 30G
git rev-parse --abbrev-ref HEAD    # 应为 release/3.0.0-beta2
git status --short                 # 确认工作区干净（本地实验改动先 stash 或 commit）

# 回滚快照：出问题时对照恢复
uv pip freeze > ~/pre-isaacsim-freeze-$(date +%m%d).txt
```

**✅ 验收**：四项检查全部通过，freeze 文件已生成。任何一项不过 → 停下解决，不要带病安装。

**⚠️ 风险预案**：uv 重建 venv 极快。最坏情况（依赖被搅乱且难以修复）的兜底方案是删掉 venv 按原安装指南重建 kit-less 环境（半小时内可恢复），所以放心操作；但快照仍是第一道防线——先 diff 再重建。

---

## Step 1 · 安装 Isaac Sim 6.0 pip 包（下载大头在这步）

**修订（实测）**：版本应为 **6.0.1.0**（`source/isaaclab/setup.py` 钉的就是 `isaacsim[all,extscache]==6.0.1.0`；索引上可用版本 6.0.0.0 / 6.0.0.1 / 6.0.1.0）。且这一步**必须绕过代理直连**：`pypi.nvidia.com` 会 301 到国内 CDN `pypi.nvidia.cn`，直连实测 ~100 MB/s；反而走代理时 8GB 级的 `isaacsim-extscache-kit` 反复超时/TLS 断连（30s 和 600s 超时都失败过两轮）。

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  UV_HTTP_TIMEOUT=600 \
  uv pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com
```

- `extscache` 会把大部分扩展缓存打进包里，换取首次启动时少联网拉取。
- 网络中断可直接重跑，uv 有缓存（部分下载的大 wheel 不缓存，会整体重下，所以直连+长超时很关键）。
- **副作用清单（实测）**：isaacsim 依赖解析会动一批既有包——`torch 2.10.0+cu128 → 2.11.0`（Step 2 恢复）、`mujoco 3.8.1 → 3.8.0`、`mujoco-warp 3.8.1 → 3.8.0.3`、scipy/requests/urllib3/websockets 等若干降级。mujoco 系的降级是 isaacsim 的钉版本，实测 Newton 训练回归不受影响，无需干预。

**✅ 验收**：`python -c "import isaacsim; print('ok')"` 不报错。

---

## Step 2 · 重新锁定 PyTorch（关键，勿跳过）

isaacsim 的依赖解析可能改动 torch 版本或换成 CPU/其他 CUDA 轮子（实测：被换成了 2.11.0 默认轮，并新增 torchaudio 2.11.0）。**RTX 5090（sm_120）必须 cu128 轮子**，官方文档也是"先装 isaacsim、后钉 torch"的顺序。**修订**：torchaudio 一并钉住，避免三件套版本错配：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  UV_HTTP_TIMEOUT=600 \
  uv pip install -U torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**✅ 验收**：输出 `2.10.0+cu128 12.8 True NVIDIA GeForce RTX 5090`（字样相近即可）。

---

## Step 3 · 接受 EULA（修订：SSH 终端即可，GUI 首启改为可选项）

**修订（实测）**：EULA 不需要 GUI——首次 `import isaacsim` 就会在终端提示，接受一次后**持久生效**：

```bash
source ~/vibe/sim/IsaacLab3/env_isaaclab/bin/activate
printf 'Yes\n' | python -c "import isaacsim"     # 接受 EULA，持久记录
python -c "import isaacsim; print('ok')"          # 复核：不再弹提示
```

（临时绕过也可用环境变量 `OMNI_KIT_ACCEPT_EULA=YES`，但不持久。）

**GUI 首启（可选，remote desktop 会话里做）**：`isaacsim` 命令首启拉取/加载扩展约 10 分钟，弹出主窗口即验收。headless 训练（Step 5）不依赖这一步——PhysX 训练已在纯 SSH 下验证通过。

**故障排查**：
- `ModuleNotFoundError: No module named 'isaacsim'` → venv 没激活，或激活了别的环境。
- 卡在扩展拉取 → 检查代理是否对该会话生效（`echo $https_proxy`）。
- 窗口黑屏/崩溃 → 先跑 `nvidia-smi` 确认驱动正常，再查 Isaac Sim Linux troubleshooting 文档。

---

## Step 4 · 给 Isaac Lab 补装 physx 后端子模块

**修订（实测）**：当初 `./isaaclab.sh -i` 其实已经把 `isaaclab_physx`、`isaaclab_ovphysx` 等胶水包全部 editable 装上了（见 env-snapshot-20260711），缺的只是 isaacsim 运行时本体。Step 1 装完后 import 直接通过，**本步无需任何操作**，仅验收：

```bash
cd ~/vibe/sim/IsaacLab3
python -c "import isaaclab_physx; print('physx backend ok')"
```

- 备忘：beta2 的 `--install` 接受子模块列表：`assets, contrib, mimic, newton, ov, physx, rl, tasks, teleop, visualizers`。
- 可选彩蛋：OVPhysX（kit-less PhysX，运行时钉在 ovphysx==0.4.13）胶水包同样已在，属远期探索项。

**✅ 验收**：`python -c "import isaaclab_physx"` 静默通过。（2026-07-12 实测通过）

---

## Step 5 · PhysX 后端训练验证（回到 SSH 即可）

**先确认 Newton 老路没被搅坏**（回归测试）：

```bash
./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Cartpole-Direct-v0 \
    --num_envs=4096 --max_iterations=50 --headless physics=newton_mjwarp
```

应与你之前的基线量级一致（~百万 steps/s 级）。

**再跑 PhysX 后端**。**修订（定案）**：正确语法是 `physics=physx`（候选 1，与 Newton 惯例一致），无需再试 `presets=physx`：

```bash
./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Cartpole-Direct-v0 \
    --num_envs=4096 --max_iterations=50 --headless physics=physx
```

**✅ 验收（2026-07-12 实测结果）**：
1. ✅ 日志出现 `isaaclab_physx.physics.physx_manager` 加载信息（TGS solver 警告即是标志）。
2. ⚠️ **重要发现**：`Isaac-Cartpole-Direct-v0` 在 PhysX 下**不收敛**（200 iters 后 reward 仍 ~2，success 0），但同任务 Newton 下正常收敛（reward ~285，success 1.0）。原因：该任务被本地 commit `4f41279`（wind + reward scale 调参）改过，参数是在 Newton 动力学下调出来的，对 PhysX 动力学不适配——**不是后端故障**。改用未改动的 manager-based `Isaac-Cartpole-v0` 验证：PhysX 下正常收敛（episode length 300，success 1.0）。这本身就是第一份"同任务跨求解器不迁移"的对照案例。
3. ✅ 第一笔吞吐对比（4096 envs，Cartpole Direct，RTX 5090）：**Newton MJWarp ~950k steps/s vs PhysX ~655k steps/s**（PhysX 为 Newton 的 ~69%）。

**故障排查**：
- PhysX 启动明显变慢属正常：它要拉起 Kit 运行时，与 kit-less Newton 的秒级启动不同（实测整体 wall time 差别不大：50 iters 均 ~15s）。

---

## Step 6 ·（收尾福利）官方手部资产陈列室

remote desktop 会话里：

```bash
./isaaclab.sh -p scripts/demos/hands.py
```

看官方自带哪几款灵巧手（Allegro / Shadow 等），为第 2 期资产选型攒第一印象。

---

## 完成后的环境能力清单（更新速查用）

| 能力 | 装前 | 装后 |
|---|---|---|
| Newton MJWarp 训练（kit-less） | ✅ | ✅（Step 5 已回归验证，~950k steps/s） |
| PhysX 训练 | ❌ | ✅（已验证，~655k steps/s） |
| PhysX FEM 柔性体 / surface gripper | ❌ | ✅（第 3 期启用） |
| URDF/MJCF 导入器、GUI 资产工具 | ❌ | ✅（第 2 期启用） |
| viser / rerun / newton 可视化 | ✅ | ✅ 不受影响 |
| kit 可视化（Isaac Sim 视口） | ❌ | ✅ 新增选项（GUI 首启待做） |

## 实际执行记录（2026-07-12）

- **版本**：isaacsim **6.0.1.0**（跟随 `source/isaaclab/setup.py` 的钉版本，而非原稿的 6.0.0）。
- **网络**：走代理下载大包两轮均失败（`isaacsim-extscache-kit` 超时/TLS 断连）；绕过代理直连 `pypi.nvidia.com`→`pypi.nvidia.cn` CDN 后 ~100 MB/s，全程 55 秒完成。**结论：NVIDIA 源国内直连远优于代理，pip/uv 装 NVIDIA 源时应加 `env -u http_proxy -u https_proxy ...`。**
- **EULA**：终端内 `printf 'Yes\n' | python -c "import isaacsim"` 一次性持久接受，无需 GUI。
- **torch**：isaacsim 把 torch 换成 2.11.0 默认轮并新增 torchaudio，按 Step 2 重钉三件套 2.10.0+cu128 后验证 `True NVIDIA GeForce RTX 5090`。
- **其他包变动**：mujoco 3.8.1→3.8.0、mujoco-warp 3.8.1→3.8.0.3（isaacsim 钉版本），Newton 回归不受影响；另新增 usd 转换器若干（mujoco-usd-converter、urdf-usd-converter 等，正好服务第 2 期资产工作流）。
- **训练验证**：Newton 回归 ✅（reward ~285 / success 1.0 / ~950k steps/s）；PhysX 在未改动的 `Isaac-Cartpole-v0` ✅（success 1.0 / ~655k steps/s）；本地调参过的 `Isaac-Cartpole-Direct-v0` 在 PhysX 下不收敛（Newton 下收敛）→ 记为跨求解器参数不迁移案例。
- **快照**：装前 `~/pre-isaacsim-freeze-0712.txt`，装后 `plan/env-snapshot-20260712.txt`（330 包）。
- **待做（需 remote desktop）**：GUI 首启 `isaacsim` 开一次主窗口；`./isaaclab.sh -p scripts/demos/hands.py` 看手部资产。

## 遗留问题（滚动清单，追加）

- [x] beta2 下 PhysX 的正确 CLI 语法 → **定案：`physics=physx`**（2026-07-12 实测）
- [ ] GUI 首启 + `scripts/demos/hands.py` 手部资产陈列（需 remote desktop 会话）
- [ ] `Isaac-Cartpole-Direct-v0` 本地调参（commit 4f41279）在 PhysX 下不收敛——若做求解器对照实验需先回退调参或按 PhysX 重调
- [ ] OVPhysX（kit-less PhysX）能否覆盖 FEM 柔性体需求（远期探索项）
- [ ] kit 可视化与现有 viser 工作流的取舍（有 GUI 需求时再评估）
