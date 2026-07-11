# 周末学习 · 第 1 期实验记录 —— 上午:Cartpole 快速上手

> 日期:2026-07-11 上午(约 11:30–12:15)
> 环境:Isaac Lab 3.0.0-beta2 kit-less + Newton MJWarp,RTX 5090,详见 `isaac-lab-3.0-setup-guide.md`
> 训练脚手架:rsl-rl PPO,`Isaac-Cartpole-Direct-v0`,除特殊注明外 4096 envs × 150 iters(约 10 秒/次)

## 一、实验总览

| # | run(logs/rsl_rl/…) | 改动(相对上一行) | reward* | ep_len* | success* |
|---|---|---|---|---|---|
| 0 | `cartpole_baseline/11-42-49` | 原版参数 | 295.8 | 299.0 | 1.000 |
| 1 | `cartpole_direct/11-49-13` | `rew_scale_pole_pos` -1 → **-5** | 285.5 | 293.3 | 0.984 |
| 2 | `cartpole_direct/11-57-50` | 终止罚 -2→**-5**,速度罚 ×3,初始角 ±0.25→**±0.35**(实为 ±63°) | 276.9 | 290.1 | 0.955 |
| 3 | `cartpole_direct/12-14-53` | **加风 ±4 N·m**(白噪声力矩,自写代码),参数回调(±0.3 / cart_vel -0.02 / pole_vel -0.01);64 envs × 300 iters + viser 实时观看 | 287.9 | 295.7 | **1.000** |

\* 最后 10 个迭代的均值。**注意:不同行的 reward 定义不同,数值不可横向比较**;可比的是 ep_len 和 success。

## 二、每个实验学到了什么

**实验 1(角度惩罚 ×5):曲线几乎不动。** 原因:①任务已饱和(两版都接近满时长);②二次惩罚在最优点附近趋近于零,放大一个≈0 的项还是≈0——权重只在"杆歪得厉害"的训练早期起作用。副作用倒是可测:success 从 1.000 掉到 0.984(修正更激进,偶尔用力过猛)。
**教训:改奖励后,曲线数值失去横向可比性;行为(回放)才是裁判;要让权重"有感",得改在账单占比大的项上。**

**实验 2(求平稳 + 任务变难):success 不再顶头(0.955)。** 三个原因:
1. `initial_pole_angle_range` 在代码里**乘了 π**(cfg 注释写 `[rad]` 是误导,±0.35 实为 ±63°,离 90° 判死线仅 27°)——任务比预想难得多;
2. 显性终止罚(-2→-5)是"零钱":摔倒的真实代价是损失后续所有 +1 存活奖励(≈-280),显性罚金占比 <2%,调它几乎无效;
3. 小车速度罚 ×3 与"极端开局需要猛冲救杆"直接冲突,策略理性地放弃了最难的开局。
**教训:①有存活奖励的任务里终止惩罚天然是配角;②平稳税和救援敏捷性此消彼长;③改参数前先读实现,注释可能骗人。**

**实验 3(随机风,域随机化):在 ±4 N·m 持续乱流下 success=1.000。** 自己动手给环境加了第一个功能(见下),viser 网页里全程观看了"被吹倒 → 学会顶风微调 → 稳定"的演化。
**教训:域随机化以少量 reward(风的修正成本)换取鲁棒性;训练时开扰动=练抗性,只在 play 时开扰动=考鲁棒。**

## 三、代码改动(本地 commit,未推送)

`source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/`:

- **`cartpole_env_cfg.py`**:奖励系数调整(terminated -5 / cart_vel -0.02 / pole_vel -0.01,初始角 ±0.3);新增 `wind_torque_max = 0.0`(0=关闭,Hydra 可命令行覆盖)。
- **`cartpole_env.py`**:`_apply_action()` 中当 `wind_torque_max > 0` 时,对 `cart_to_pole` 关节每个物理步(120Hz)施加 `±wind_torque_max` 均匀采样的白噪声力矩。进阶方向(未做):把风存成状态、每 N 步重采样,模拟"阵风"。

## 四、固化下来的工作流(速查)

```bash
# 正式训练(headless,~10s/150iter)
./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Cartpole-Direct-v0 \
    --num_envs=4096 --max_iterations=150 --headless physics=newton_mjwarp

# 边训边看(观赏用,环境数小、迭代多;浏览器 http://192.168.1.220:8080)
./isaaclab.sh train ... --num_envs=64 --max_iterations=400 --visualizer viser 'env.wind_torque_max=4.0'

# 回放(自动加载最新 checkpoint;--checkpoint 指定文件;--real-time 实时速度)
./isaaclab.sh play --rl_library rsl_rl --task=Isaac-Cartpole-Direct-v0 --num_envs 4 \
    --headless --real-time physics=newton_mjwarp --visualizer viser \
    'env.episode_length_s=30' 'env.initial_pole_angle_range=[-0.05,0.05]'

# 曲线(实时刷新;Cursor 自动转发端口)
./isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl/ --port 6006

# Hydra 覆盖:任何 env cfg 字段皆可命令行改,如 'env.wind_torque_max=8.0'
```

观察平稳性/行为差异的要点:少量环境 + `--real-time` + 长 episode + 小初始角,盯"站稳后的状态";TensorBoard 只看 ep_len/success 趋势(smoothing 拉高,success_rate 按 reset 批次统计,天然抖动)。

## 五、遗留问题

- [ ] `--video` 录像在 kit-less 下是否可用,未验证(下午测)
- [ ] `newton_kamino` 引擎对比,未做(留到接触密集任务再横评)
- [ ] cfg 中 `initial_pole_angle_range` 的 `[rad]` 注释误导(实际乘 π)——可作为第一个 upstream issue/PR 素材
- [ ] 上午 10:29 的最早基线 run 未保留(被后续流程覆盖),以 11-42-49 为准

## 下午计划

四足平地行走 `Isaac-Velocity-Flat-Unitree-Go2-v0`(manager-based 风格):ActuatorCfg、RewTerm/ObsTerm/EventTerm 配置体系,与 direct 风格对照。
