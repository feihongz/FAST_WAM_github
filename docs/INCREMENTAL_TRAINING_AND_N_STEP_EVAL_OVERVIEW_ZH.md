# FastWAM Incremental 训练与 N-step Eval：逻辑和代码结构

> 本文面向需要继续开发、复现实验或审查实现的工程人员。内容按当前工作区代码整理，基准提交为 `bf9262d`；prefix eval 等实现仍包含未提交修改，因此应以本文列出的具体文件和函数为准。

## 1. 先说结论

这里的 **incremental** 不是“不断加入新数据的增量学习”，也不是从旧 checkpoint 做普通续训。它指在基础 FastWAM 上新增的两条 unified 训练链路：同一个训练样本同时学习两种 action 去噪方式。

- `wo-video`（简称 `wo`）：action token 只能看到真实首帧对应的 video token，不能看到 noisy future-video token。
- `w-video`（简称 `w`）：action token 可以看到完整 video token，与 future video 做 joint denoising。

仓库实现了两个参数共享方案：

| 变体 | Video DiT | ActionDiT | 每个 batch 的 action 分支 |
| --- | --- | --- | --- |
| `FastWAMUnifiedShared` / UniShare | 1 个，共享 | 1 个，共享 | 同一个 ActionDiT 分别跑 `wo`、`w` 两次 |
| `FastWAMUnifiedTwoAction` / TwoAction | 1 个，共享 | 2 个，分开 | `action_expert_wo` 跑 `wo`，`action_expert_w` 跑 `w` |

后续 eval 的 **N-step** 只在 UniShare 上实现。它的准确含义是：

> 一次 action chunk 的总共 `T` 次 flow-matching solver 调用中，前 `N` 次使用 `w` 的 joint video-action vector field，后 `T-N` 次切换到 `wo` 的 action-only vector field。

因此 N 不是环境 timestep、不是执行的 action 数、不是视频帧数，也不是“先把视频去噪 N 步，再把该视频交给 action 模型”。更准确的名字是 `joint_denoise_prefix_steps`；当前配置名仍是 `video_prefix_steps`。

## 2. 基础 FastWAM 与 incremental 改了什么

### 2.1 基础组件

基础模型位于 `src/fastwam/models/wan22/fastwam.py`，核心组件是：

- `video_expert`：Wan2.2 video DiT，处理 VAE video latent。
- `action_expert`：ActionDiT，处理连续 action chunk。
- `MoT`：Mixture-of-Transformers 容器。每层分别用 video/action expert 产生 Q/K/V，再拼接后做带 mask 的 mixed attention。
- video/action 各自的 `WanContinuousFlowMatchScheduler`。
- 文本条件，以及可选的 proprio 编码后追加到 text context 的条件。

在当前 unified 配置中，`video_dit_config.action_conditioned=false`。也就是说，video query 不读取 action token；信息流主要由 mixed-attention mask 决定 action 能读哪些 video token。

### 2.2 两种 attention mask

两个模式的共同部分是：

- video query 按 `first_frame_causal` video mask 读取 video token；
- action query 可以读取全部 action token；
- video query 不读取 action token。

区别只在 action 到 video 的可见范围：

| Query → Key/Value | `wo` | `w` |
| --- | --- | --- |
| video → video | video 自身 mask | video 自身 mask |
| video → action | 不可见 | 不可见 |
| action → action | 全可见 | 全可见 |
| action → video | 只看 first-frame tokens | 看全部 video tokens |

对应实现：

- 基础 `wo` mask：`FastWAM._build_mot_attention_mask()`。
- unified `wo` 包装：`FastWAMUnifiedShared._build_wo_video_mask()`。
- unified `w` mask：`FastWAMUnifiedShared._build_w_video_mask()`。
- 实际 mixed attention：`MoT.forward()`。

这里的 `w` 是 **Joint-style 同步 video/action 去噪**，不是 `fastwam_idm.py` 中“先生成视频、再固定视频做 action”的 IDM teacher-forcing/two-stage 路径。

## 3. 训练输入和公共链路

### 3.1 数据样本

`RobotVideoDataset` 先从 LeRobot 数据中取连续 33 个 observation 和 32 个 action，再按 `action_video_freq_ratio=4` 对 observation 抽帧：

```text
原 observation index: 0, 1, 2, ... 32
送入 video 的 index : 0, 4, 8, ... 32  => 9 frames
送入 action          : 0 ... 31         => 32 actions
```

处理后的训练样本主要包含：

```text
video         [B, 3, 9, H, W]
action        [B, 32, action_dim]
proprio       [B, ..., proprio_dim]
context       [B, L, text_dim]
context_mask  [B, L]
image_is_pad / action_is_pad
```

LIBERO 将 2 路相机水平拼接为 `224×448`，action/state 维度分别为 7/8；RoboTwin 将 3 路相机拼成 `384×320`，action/state 都是 14 维。文本 embedding 通常预计算到 cache，正式训练时 `load_text_encoder=false`。

### 3.2 `build_inputs()`

`FastWAM.build_inputs()` 完成公共预处理：

1. video 经 VAE 编码成 video latent；
2. 当 `fuse_vae_embedding_in_latents=true` 时，保存干净的 first-frame latent；
3. text embedding、mask 和 proprio 移到模型设备；
4. proprio 只取当前时刻并编码，然后追加到 context；
5. action、padding mask 转换到训练 dtype/device。

训练加噪时，future-video latent 和 action 分别采样 timestep、噪声和 flow-matching target；首帧 latent 随后被干净首帧覆盖。因此训练和推理都把真实首帧当作 observation anchor。

### 3.3 公共 loss 形式

两种 incremental 变体最终都使用：

```text
L_action_mix = alpha_wo * L_action_wo + alpha_w * L_action_w
L_total      = lambda_video * L_video + lambda_action * L_action_mix
```

`set_action_mix_weights()` 会把 `alpha_wo`、`alpha_w` 归一化；当前 YAML 默认均为 `0.5`，`lambda_video=lambda_action=1.0`。

action/video loss 都会：

- 对预测的 flow vector 与 `noise - clean_sample` 做 MSE；
- 排除 padding token/frame；
- 乘 scheduler 的 timestep training weight。

一个容易忽略的细节是：`L_video` 只取 `wo` forward 的 video 输出。`w` forward 的 video 输出没有单独再算一次 video loss；但 `w` action loss 仍可沿 action 对 video K/V 的依赖更新共享 video 表征。

## 4. UniShare 训练逻辑

入口类：`src/fastwam/models/wan22/fastwam_unified_shared.py::FastWAMUnifiedShared`。

每个 batch 的 `training_loss()` 只生成一份 noisy video、一份 noisy action，以及各自的一组 timestep/target。随后复用同一组 pre-DiT token，分别跑两次 MoT：

```text
                         ┌─ mask_wo ─ shared MoT ─ pred_action_wo
noisy video/action ──────┤
same timestep / target   └─ mask_w  ─ shared MoT ─ pred_action_w
                              │
                              └─ 同一个 video_expert + 同一个 action_expert
```

伪代码对应如下：

```python
inputs = build_inputs(sample)
noisy_video, target_video = noise_video(inputs.video)
noisy_action, target_action = noise_action(inputs.action)

video_pre = video_expert.pre_dit(noisy_video, ...)
action_pre = action_expert.pre_dit(noisy_action, ...)

out_wo = mot(video_pre, action_pre, attention_mask=mask_wo)
out_w  = mot(video_pre, action_pre, attention_mask=mask_w)

loss_video     = video_loss(video_expert.post_dit(out_wo.video), target_video)
loss_action_wo = action_loss(action_expert.post_dit(out_wo.action), target_action)
loss_action_w  = action_loss(action_expert.post_dit(out_w.action), target_action)
```

这条链路研究的是：同一 ActionDiT 是否能同时学习 first-frame-only 和 full-video-conditioned 两个局部 vector field。代价是每个 batch 对共享 MoT 做两次完整 forward，两个 action 目标也会直接竞争同一组 ActionDiT 参数。

## 5. TwoAction 训练逻辑

入口类：`src/fastwam/models/wan22/fastwam_unified_two_action.py::FastWAMUnifiedTwoAction`。

初始化时创建：

- 一个 `video_expert`；
- 两个从同一 ActionDiT backbone 初始化的 `action_expert_wo`、`action_expert_w`；
- 两个 MoT 容器 `mot_wo`、`mot_w`。

两个 MoT 引用同一个 `video_expert`，但引用不同的 ActionDiT：

```text
                     ┌─ mot_wo = shared video + action_expert_wo ─ mask_wo
noisy video/action ──┤
                     └─ mot_w  = shared video + action_expert_w  ─ mask_w
```

它与 UniShare 使用相同 noisy sample、timestep、target 和混合 loss。差别是 `L_action_wo` 与 `L_action_w` 不再直接更新同一个 ActionDiT；两者仍通过共享 video expert 耦合。

checkpoint 中 `dit` 保存 `ModuleDict({"mot_wo", "mot_w"})`。加载代码也兼容早期的 `mot_wo/mot_w` 或单 `mot` payload。静态推理 `w` 时，会临时把 `self.action_expert/self.mot` 切换成 w 分支，结束后恢复。

当前 TwoAction 明确不支持 `inference_mode=prefix`，因为还没有定义在第 N 步应该怎样从 `action_expert_w` 切换到 `action_expert_wo`，以及如何处理两个 head 之间的 action latent trajectory 分布偏移。

## 6. 从启动脚本到 optimizer step

训练调用链为：

```text
scripts/jihe/train_*_8xh100.sh
  -> scripts/train_zero1.sh
  -> accelerate / DeepSpeed ZeRO-1
  -> scripts/train.py
  -> fastwam.runtime.run_training()
       -> Hydra instantiate(cfg.model)
       -> Hydra instantiate(cfg.data.train / val)
       -> Wan22Trainer.train()
            -> model.training_loss(sample)
            -> backward / clip grad / optimizer / LR scheduler
            -> eval / checkpoint / resume state
```

配置组合关系是：

```text
configs/train.yaml
  + configs/task/<benchmark>_unified_<variant>_*.yaml
      + configs/data/<benchmark>.yaml
      + configs/model/fastwam_unified_<variant>.yaml
```

`runtime.create_fastwam_unified_shared/two_action()` 把 Hydra 的 model、scheduler、loss 配置转换成模型构造参数。`Wan22Trainer` 只依赖统一的 `training_loss()` 接口，因此 trainer 本身没有为 incremental 单独写分支。

正式 JiHe launcher 还负责解析运行环境、检查 Wan2.2/ActionDiT/data/text cache/stats、设置输出盘、W&B、GPU 数和诊断日志。逻辑调试时应先看 Python；复现实验环境时再看这些 shell launcher。

## 7. Eval 的三种推理模式

UniShare 的统一入口是 `infer_action_mode(..., inference_mode=...)`：

| 模式 | 实际入口 | 含义 |
| --- | --- | --- |
| `wo` | `infer_action_without_video()` → `FastWAM.infer_action()` | 只编码真实首帧，缓存 video K/V，迭代去噪 action |
| `w` | `infer_action_with_video()` → `FastWAMJoint.infer_action()` | video/action 从噪声开始，在全部 solver iteration 中同步更新 |
| `prefix` | `infer_action_video_prefix()` | 前 N 次 `w`，剩余次数 `wo` |

TwoAction 只实现静态 `wo/w`；prefix 会抛 `NotImplementedError`。当前没有根据状态或置信度动态切换的 gate。

## 8. N-step prefix 的精确状态机

设总 inference steps 为 `T`，配置 `video_prefix_steps=N`，要求 `0 <= N <= T`。以 `T=10` 为例：

```text
N=0 : wo wo wo wo wo wo wo wo wo wo
N=1 : w  wo wo wo wo wo wo wo wo wo
N=5 : w  w  w  w  w  wo wo wo wo wo
N=9 : w  w  w  w  w  w  w  w  w  wo
N=10: w  w  w  w  w  w  w  w  w  w
```

### 8.1 初始化

`infer_action_video_prefix()` 会：

1. 生成 Gaussian future-video latent；
2. 生成 Gaussian action latent；
3. 编码当前 observation，使用干净 first-frame latent 覆盖 video latent 的第一帧；
4. 构造 video/action 各自的 inference schedule。

video 与 action 分别创建 generator，但在指定 seed 时二者都用同一个 seed 初始化。这提供确定性，却不等于两个独立随机试验。

### 8.2 前 N 次 joint iteration

当 `step_idx < N`：

```text
(video_t, action_t)
        -> w-mask joint forward
        -> (pred_video, pred_action)
        -> 分别 scheduler.step
        -> (video_next, action_next)
        -> 再次覆盖干净 first-frame latent
```

video 和 action 在同一次 forward 中读取的是 update 之前的当前 latent，然后同时预测、随后同时更新。它不是“先更新视频，再让 action 看更新后的视频”。

因此 N=1 时，action 唯一一次 joint prediction 看到的 future-video 仍是初始 Gaussian noise；这一步更新出来的 video 在下一轮切到 `wo` 后不会再被 action 使用。

### 8.3 后 T-N 次 action-only suffix

第一次进入 `step_idx >= N` 时：

1. 丢弃 prefix 阶段的 partial future-video latent；
2. 只用原始 first-frame latent 重新做一次 video prefill；
3. 在 `MoT.prefill_video_cache()` 中缓存每层 video K/V；
4. 后续 action step 通过 `MoT.forward_action_with_video_cache()` 只重算 action Q/K/V，并读取缓存的首帧 video K/V；
5. action latent 继续继承前 N 次 joint update 后的状态。

所以，prefix 对 action 的影响保存在 action latent trajectory 中；partial future video 本身不会被 suffix 继续消费。

### 8.4 两个端点

- `N=0`：走 `_infer_action_without_video_custom()`，从第一步起使用 first-frame video KV cache。
- `N=T`：所有 iteration 都走 joint 分支，不创建 suffix cache。

当前函数内部无条件把 `force_custom_prefix=True`，LIBERO evaluator 也会硬编码传 true；因此配置里的同名开关实际上不能关闭 custom endpoint 路径。

### 8.5 N 与去噪量不是线性关系

inference schedule 先对均匀的 `u` 使用：

```text
sigma = shift * u / (1 + (shift - 1) * u)
```

当前常用的 `T=10, shift=5` 下，前 N 步累计覆盖的 `|delta sigma|` 为：

| N | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 累计路径 | 0% | 2.17% | 4.76% | 7.89% | 11.76% | 16.67% | 23.08% | 31.82% | 44.44% | 64.29% | 100% |

最后一步单独占 35.71%。因此不能把 N=5 解释成 50% video contribution，也不应把 N=9 到 N=10 当作普通的“再多 10%”。做结果分析时，使用 sigma cutoff 或累计 `|delta sigma|` 会比单纯的 N 更准确。

## 9. N-step 与训练目标的关系

训练分别监督了 `wo` 和 `w` 两个局部 vector field，但没有显式采样或监督如下完整轨迹：

```text
w -> w -> ... -> wo -> wo
```

prefix eval 是在推理时硬组合两个已经学到的 vector field，属于 inference ablation。它可以回答“action sampling 的前多少次调用启用 joint field 时表现如何”，不能直接回答“逐渐变清晰的预测视频是否提升 action”，也不能等价为连续 gate 强度。

若要回答 partial-video quality/alignment，应实现另一条实验：video 独立去噪 N 步 → 冻结该 partial video → action 显式读取它。当前 prefix 链路不是这一设计。

## 10. LIBERO eval 调用链

```text
scripts/jihe/eval_libero_*prefix*.sh
  -> experiments/libero/run_libero_manager.py
       -> 为 suite/task 分配 worker/GPU
  -> experiments/libero/eval_libero_single.py
       -> eval_single_process()
       -> run_single_task()
       -> run_single_episode()
       -> _predict_action_chunk()
       -> model.infer_action_mode(...)
```

`_predict_action_chunk()` 负责：

- 读取主视角/腕部相机并按训练方式拼接；
- 归一化 proprio，构造 prompt；
- 传递 `inference_mode`、`num_inference_steps`、`video_prefix_steps`；
- 反归一化 action 并恢复 LIBERO gripper 约定。

episode 是 receding-horizon 控制：每次预测长度通常为 32 的 action chunk，只执行前 `replan_steps` 个 action，再基于新 observation 重新规划。默认 `replan_steps=10`，它与内部 diffusion 的 `video_prefix_steps` 是两个完全不同的轴。

需要注意当前分发优先级：若 `visualize_future_video=true`，evaluator 会优先调用完整 `infer_joint()`，即使同时配置了 `inference_mode=prefix` 也会绕过 prefix。正式 prefix sweep 应保持 `visualize_future_video=false`。

## 11. RoboTwin eval 调用链

```text
scripts/jihe/eval_robotwin_unified_shared_prefix_8xh100.sh
  -> experiments/robotwin/run_robotwin_manager.py
       -> 50 tasks × clean/random phase 的多进程/GPU 调度
  -> experiments/robotwin/eval_robotwin_single.py
       -> 调用 third_party/RoboTwin/script/eval_policy.py
       -> 将 inference 参数传到 policy
  -> experiments/robotwin/fastwam_policy/deploy_policy.py
       -> WorldActionRobotWinPolicy._infer_action_chunk()
       -> model.infer_action_mode(...)
```

RoboTwin policy 将 head camera 与两个 wrist camera 拼成 `384×320`，归一化 14 维 state，生成 action chunk 后将前 `replan_steps` 个 action 放入 `pending_actions` 队列。队列耗尽时才请求新 observation 并重新推理；默认 `replan_steps=24`。

manager 可通过 `ROBOTWIN_EVAL_GPU_IDS` 选择物理 GPU，并用 `ROBOTWIN_EVAL_LAUNCH_DELAY_SECONDS` 错峰启动 SAPIEN/Vulkan worker。它最终汇总每个 task 的 clean/random success rate 到 `summary.csv` 和 `summary.json`。

## 12. 关键配置和常用入口

模型配置：

- `configs/model/fastwam_unified_shared.yaml`
- `configs/model/fastwam_unified_two_action.yaml`
- `loss.alpha_wo / alpha_w`：两个 action loss 的权重。
- `video_scheduler.* / action_scheduler.*`：训练与推理 scheduler。
- `video_attention_mask_mode=first_frame_causal`：prefix action-only suffix 的前提。

训练任务配置：

- `configs/task/libero_unified_shared_2cam224_1e-4.yaml`
- `configs/task/libero_unified_two_action_2cam224_1e-4.yaml`
- `configs/task/robotwin_unified_shared_3cam_384_1e-4.yaml`
- `configs/task/robotwin_unified_two_action_3cam_384_1e-4.yaml`

正式训练入口：

```bash
bash scripts/jihe/train_libero_unified_shared_8xh100.sh
bash scripts/jihe/train_libero_unified_two_action_8xh100.sh
bash scripts/jihe/train_robotwin_unified_shared_8xh100.sh
bash scripts/jihe/train_robotwin_unified_two_action_8xh100.sh
```

静态 `wo/w` 对照入口：

```bash
bash scripts/jihe/eval_libero_incremental_4x2gpu.sh
bash scripts/jihe/eval_robotwin_incremental_8xh100.sh
```

RoboTwin UniShare N-sweep：

```bash
N_START=0 N_END=10 NUM_INFERENCE_STEPS=10 \
  bash scripts/jihe/eval_robotwin_unified_shared_prefix_8xh100.sh
```

核心 eval 配置：

```yaml
EVALUATION:
  inference_mode: prefix
  video_prefix_steps: 5
  num_inference_steps: 10
  replan_steps: 10  # LIBERO；RoboTwin 默认 24
```

## 13. 推荐阅读顺序

如果要快速接手代码，建议按以下顺序：

1. `configs/model/fastwam_unified_shared.yaml`：先看组件和 loss 配置。
2. `src/fastwam/models/wan22/fastwam.py`：理解输入、基础 mask、loss 和 canonical inference。
3. `src/fastwam/models/wan22/mot.py`：理解 mixed attention 与 KV cache。
4. `src/fastwam/models/wan22/fastwam_unified_shared.py`：看双分支训练和 prefix 状态机。
5. `src/fastwam/models/wan22/fastwam_unified_two_action.py`：看双 ActionDiT 的共享边界。
6. `src/fastwam/runtime.py`、`src/fastwam/trainer.py`：看 Hydra 构造和训练闭环。
7. `experiments/libero/eval_libero_single.py` 或 `experiments/robotwin/fastwam_policy/deploy_policy.py`：看 benchmark 输入输出适配。
8. 对应 manager 和 `scripts/jihe/*.sh`：最后看多 GPU 调度、实验矩阵和产物汇总。

## 14. 当前实现边界与维护提醒

- prefix 只支持 UniShare；TwoAction 的 head-switch 语义尚未定义。
- `force_custom_prefix` 当前是事实上的硬编码，不是有效开关。
- LIBERO 的 `visualize_future_video=true` 会优先走完整 joint inference，应避免和 prefix 同时启用。
- prefix suffix 丢弃 partial future-video，只保留它对 action latent 前缀轨迹的影响。
- N 不是线性去噪比例，跨 scheduler shift 比较时尤其不能只看整数 N。
- 训练没有显式覆盖 `w -> wo` 切换轨迹，因此中间 N 是组合式推理消融，不是训练内分布。
- video loss 只来自 `wo` forward；修改分支或 loss 时要确认是否有意保持这一非对称设计。
- 固定 diffusion seed 有利于不同 N 的 paired comparison，但不能替代多 sampling seed 评测。
- 正式结果应同时保存 Git SHA、dirty diff、完整 Hydra config、checkpoint/stats 路径与 seed；当前 prefix 相关代码仍在未提交工作区中。

相关专项文档：

- `docs/INCREMENTAL_TRAINING.md`：原始 incremental 训练使用说明。
- `docs/LIBERO_N_STEP_EVAL_AUDIT.md`：LIBERO prefix 语义和实验审计。
- `docs/CURRENT_INCREMENTAL_TRAINING_AND_DENOISING_EVAL.md`：训练/评测进展快照与已有结果。
