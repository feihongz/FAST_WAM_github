# LIBERO N-step Prefix Evaluation 代码逻辑审计

> 审计日期：2026-08-02  
> 审计范围：LIBERO Unified Shared 模型的 `video_prefix_steps=N` 推理、评估链路、已保存 sweep 结果及其代码版本一致性。  
> 本文仅记录只读审计结论，未对实现做修改。

## 1. 核心结论

当前代码实现的是：

> 在每次 action diffusion sampling 的总共 `T` 次 model call 中，前 `N` 次使用 joint video-action 分支，后 `T-N` 次切换为 first-frame-only 的 action-only 分支。

在这个定义下，当前 `step_idx < video_prefix_steps` 的判断、w/wo attention mask 和 suffix KV cache 链路基本正确，没有发现明确的 off-by-one 或 cache 接错。

但是，如果实验原本想表达的是：

> 先将 future video 去噪 N 步，再将这个 partial video 用于 action alignment 或 action generation。

那么当前实现与该目标不一致。当前代码不会在切换后继续使用已经部分去噪的 future-video latent；它测量的更准确地说是 **joint denoising vector field 在 action 轨迹前缀中启用多少次**。

因此，推荐将当前变量的含义表述为：

```text
joint_denoise_prefix_steps
```

而不是直接称为 `N-step video quality` 或 `N-step video-action alignment`。

## 2. LIBERO evaluation 调用链

主要入口：

- `experiments/libero/eval_libero_single.py:425`：`_predict_action_chunk`
- `experiments/libero/eval_libero_single.py:484-488`：传递 prefix 参数并调用 `infer_action_mode`
- `src/fastwam/models/wan22/fastwam_unified_shared.py:656-663`：根据 `inference_mode` dispatch
- `src/fastwam/models/wan22/fastwam_unified_shared.py:440-640`：prefix inference 实现

正式 sweep 的主要设置为：

```text
num_inference_steps = 10
action_horizon       = 32
replan_steps         = 10
num_steps_wait       = 30
num_video_frames     = 9
seed                 = 42
rand_device          = cpu
visualize_future_video = false
```

每次 replanning 的执行过程为：

1. 根据当前 observation 预测一个长度为 32 的 action chunk。
2. action sampler 内部执行 10 次 diffusion/flow-matching solver iteration。
3. 环境只执行 action chunk 的前 10 个 action。
4. 执行完 10 个 action 后，使用新 observation 重新规划。

因此，`video_prefix_steps=N`：

- 不是环境中的 N 个 timestep；
- 不是执行 N 个 action；
- 不是 N 个视频帧；
- 而是每次 action chunk sampling 内部前 N 次 joint model call。

## 3. N-step 的精确执行语义

当 `T=10` 时，各个 N 对应的分支序列是：

```text
N=0 : wo wo wo wo wo wo wo wo wo wo
N=1 : w  wo wo wo wo wo wo wo wo wo
N=5 : w  w  w  w  w  wo wo wo wo wo
N=9 : w  w  w  w  w  w  w  w  w  wo
N=10: w  w  w  w  w  w  w  w  w  w
```

其中：

- `w`：action token 可以 attend 全部 video token；
- `wo`：action token 只能 attend 真实 observation 对应的 first-frame video token，以及其他 action token。

mask 定义位于：

- `src/fastwam/models/wan22/fastwam_unified_shared.py:34-67`
- `src/fastwam/models/wan22/fastwam.py:385-407`

### 3.1 初始化

Prefix sampler 会初始化：

- 一个 Gaussian future-video latent；
- 一个 Gaussian action latent；
- 将 video latent 的第一帧覆盖为真实 input image 编码得到的 first-frame latent。

相关代码位于 `fastwam_unified_shared.py:520-542`。

### 3.2 前 N 次 joint iteration

对于 `step_idx < video_prefix_steps`：

```text
(video_t, action_t)
        ↓ joint forward
(pred_video, pred_action)
        ↓ scheduler update
(video_next, action_next)
```

代码位于 `fastwam_unified_shared.py:595-610`。

关键更新顺序是：

1. 使用当前的 `latents_video` 和 `latents_action` 做 joint prediction；
2. 得到 `pred_video` 和 `pred_action`；
3. 随后才分别更新 video latent 和 action latent。

因此 action 在第 i 次 joint forward 中看到的是 **本次 update 之前** 的 video latent。

### 3.3 第 N 次之后的 action-only suffix

对于 `step_idx >= video_prefix_steps`：

1. 将 inference mode 切换为 `wo`；
2. 不再使用当前 partial future-video latent；
3. 重新从原始 first-frame latent 构建 video KV cache；
4. 后续 iteration 只预测并更新 action latent。

代码位于 `fastwam_unified_shared.py:613-631`。

action latent 会继承前 N 次 joint 更新的结果，因此前缀中 video 对 action 产生的影响不会被清空；但是已经部分去噪的 future-video latent 会被丢弃。

## 4. 一个容易误解的更新顺序问题

当前代码并不是：

```text
先更新 video → 再让 action 读取更新后的 video
```

而是：

```text
action 与 video 同时读取当前 latent → 同时预测 → 再同时更新
```

由此产生两个重要结果：

### N=1

`N=1` 时，唯一一次 joint action prediction 读取的是：

- 干净的真实 first frame；
- 尚未经过任何 scheduler update 的 Gaussian future-video noise。

随后 video 会被更新一步，但下一次 iteration 已切换到 `wo`，更新后的视频不会被 action 再次读取。

所以 N=1 不能解释为“action 使用了一步去噪后的视频”。

### 一般的 N<T

第 N 次 joint forward 完成后，代码仍然执行第 N 次 video scheduler update；但下一次 iteration 会切换到 `wo`，因此这次更新后的 future-video latent不会被 action 消费。

这不是以“joint model call 数量”为定义时的 off-by-one；但如果 N 的定义是“action 实际使用了 N-step-denoised video”，那么存在语义上的错位。

## 5. Scheduler 导致 N 与实际去噪量不线性

Scheduler 位于：

```text
src/fastwam/models/wan22/schedulers/scheduler_continuous.py:63-87
```

正式设置为：

```text
T = 10
shift = 5
```

对应的 sigma 边界为：

| Boundary | Sigma |
|---:|---:|
| 0 | 1.000000 |
| 1 | 0.978261 |
| 2 | 0.952381 |
| 3 | 0.921053 |
| 4 | 0.882353 |
| 5 | 0.833333 |
| 6 | 0.769231 |
| 7 | 0.681818 |
| 8 | 0.555556 |
| 9 | 0.357143 |
| 10 | 0.000000 |

前 N 次 joint iteration 所覆盖的累计 `|Δsigma|` 为：

| N | 累计去噪路径 | 剩余路径 |
|---:|---:|---:|
| 0 | 0.00% | 100.00% |
| 1 | 2.17% | 97.83% |
| 2 | 4.76% | 95.24% |
| 3 | 7.89% | 92.11% |
| 4 | 11.76% | 88.24% |
| 5 | 16.67% | 83.33% |
| 6 | 23.08% | 76.92% |
| 7 | 31.82% | 68.18% |
| 8 | 44.44% | 55.56% |
| 9 | 64.29% | 35.71% |
| 10 | 100.00% | 0.00% |

最后一次 update `sigma: 0.357143 → 0` 单独占总去噪路径的 35.71%。

这意味着：

- N=5 不代表使用了 50% 的 video contribution；
- N=9 到 N=10 也不是只多了最后 10%；
- N=9 的 joint action prediction 最低只读取到约 `sigma=0.556` 的 video latent；
- video 更新到约 `sigma=0.357` 后立即被丢弃，最后一个 action iteration 使用的是 `wo`；
- N=10 才让权重最大的最后一次 action update 使用低噪 video。

因此，N=10 相比 N=9 的性能变化不能简单解释为“多使用了一步视频”。更合理的横轴应当是 `sigma cutoff` 或累计 `|Δsigma|`。

## 6. 训练与 prefix inference 的关系

Shared 模型训练时，在同一 noisy sample 上分别计算：

- `loss_action_wo`
- `loss_action_w`

相关代码位于 `fastwam_unified_shared.py:149-270`。

这意味着模型分别学习了 w 和 wo 两种局部 vector field，但训练目标没有显式构造：

```text
w → w → ... → wo → wo
```

这样的完整 hard-switch sampling trajectory。

因此 prefix inference 是对两个已学习 vector field 的组合。它可以作为有效的推理消融，但不能直接等价为连续的 video gate 强度或 partial-video 质量。

## 7. 当前代码中确认存在的问题

### 7.1 `force_custom_prefix` 配置实际无效

Evaluator 在 prefix 模式下直接硬编码：

```python
infer_kwargs["force_custom_prefix"] = True
```

位置：`experiments/libero/eval_libero_single.py:485-487`。

模型内部又无条件执行：

```python
force_custom_prefix = True
```

位置：`src/fastwam/models/wan22/fastwam_unified_shared.py:471`。

因此 `configs/sim_libero.yaml` 中的 `EVALUATION.force_custom_prefix` 当前没有实际控制作用，属于死配置/误导性 API。

### 7.2 `visualize_future_video=true` 会绕过 prefix

Evaluator 的调用优先级为：

```python
if visualize_future_video:
    pred = model.infer_joint(...)
elif hasattr(model, "infer_action_mode"):
    ...
```

位置：`experiments/libero/eval_libero_single.py:480-488`。

因此同时设置：

```yaml
inference_mode: prefix
visualize_future_video: true
```

实际执行的是完整 `infer_joint`，而不是 prefix N-step。外部元数据仍可能显示 prefix 配置，从而静默污染实验。

正式 LIBERO N-sweep 中 `visualize_future_video=false`，所以已保存结果没有受到这个问题影响。

### 7.3 Prefix 代码和实验结果没有绑定到冻结版本

截至审计时，相关文件仍是未提交工作区修改，包括：

```text
configs/sim_libero.yaml
experiments/libero/eval_libero_single.py
src/fastwam/models/wan22/fastwam_unified_shared.py
```

结果目录没有保存完整 Git SHA、源码 diff 或源码快照。当前代码与正式 sweep 运行时的代码不是逐字一致，导致结果复现依赖运行时间、历史会话记录和结果元数据恢复。

### 7.4 固定 seed 的含义

正式 evaluation 每次 replanning 都重新使用固定 `seed=42`：

- 有利于不同 N 之间进行配对比较；
- 但 50 个 episodes 主要覆盖环境 initial-state 差异；
- 不覆盖 diffusion sampling seed 的不确定性；
- video/action 分别使用同一 seed 初始化 generator，二者的初始噪声并非独立随机实验。

这一点不会单独破坏当前 N 间比较，但在讨论 video-action alignment 和随机性鲁棒性时需要明确说明。

## 8. 已保存 LIBERO N-sweep 的核验结果

结果目录：

```text
/root/feihong/FastWAM/evaluate_results/libero_prefix_shared_T10/
prefix_shared_T10_20260710_2218
```

完整性检查：

- N=0 至 N=10 均已完成；
- 每个 N 包含 40 个 tasks；
- 每个 task 包含 50 个有效 episodes；
- 每个 N 共计 2,000 episodes；
- 所有 N 的 invalid episode 数均为 0；
- 各 N 的任务列表、seed、replan、action horizon 等主要配置一致；
- 归一化移除输出目录和 `video_prefix_steps` 后，11 份 manager config 一致。

正式结果为：

| N | Successes | Episodes | Success Rate |
|---:|---:|---:|---:|
| 0 | 1957 | 2000 | 97.85% |
| 1 | 1952 | 2000 | 97.60% |
| 2 | 1960 | 2000 | 98.00% |
| 3 | 1950 | 2000 | 97.50% |
| 4 | 1956 | 2000 | 97.80% |
| 5 | 1956 | 2000 | 97.80% |
| 6 | 1955 | 2000 | 97.75% |
| 7 | 1952 | 2000 | 97.60% |
| 8 | 1959 | 2000 | 97.95% |
| 9 | 1953 | 2000 | 97.65% |
| 10 | 1970 | 2000 | 98.50% |

这些结果可以作为“joint denoising prefix length”的正式结果使用，但暂时不应表述为“不同 N-step partial-video 质量对应的 action 表现”。

## 9. 正式 sweep 与当前代码的版本差异

### 正式 N=0 和 N=10

正式 sweep 运行时：

- N=0 直接 shortcut 到原始 Shared `wo` inference；
- N=10 直接 shortcut 到原始 Shared `w` inference；
- N=1 至 N=9 才进入自定义 prefix loop。

因此正式曲线的两个端点本质上是 baseline anchor，而不是由当前 custom endpoint 代码独立产生。

逐 task 核对结果：

- 正式 N=0 与原始 Shared wo 的 40/40 个 task success count 完全一致；
- 正式 N=10 与原始 Shared w 的 40/40 个 task success count 完全一致。

### N=9 补跑

N=9 在代码修改期间补跑完成，结果中存在三代元数据：

| 元数据版本 | Tasks 数量 | 时间范围 |
|---|---:|---|
| 无 `eval_force_custom_prefix` 字段 | 23 | 2026-07-15 09:52:52 至 23:06:07 |
| `eval_force_custom_prefix=false` | 2 | 2026-07-15 23:17:54 至 23:28:57 |
| `eval_force_custom_prefix=true` | 15 | 2026-07-15 23:46:21 至 2026-07-16 03:00:52 |

历史补丁核对表明，这几版只改变了 N=0/N=T 的 endpoint shortcut 和结果元数据，没有改变 N=1…9 的核心循环。因此 N=9 的数学路径应当一致，但从严格实验复现角度，仍建议冻结代码后重跑 N=9。

## 10. Endpoint sanity 的证据强度

后续保存了 custom endpoint sanity：

```text
/root/feihong/FastWAM/evaluate_results/libero_endpoint_sanity_T10/
endpoint_sanity_long_only_20260716
```

LIBERO-10 上：

- `wo_orig` 与 custom N=0 均为 `477/500`；
- `w_orig` 与 custom N=10 均为 `490/500`；
- 逐 task、逐 episode success/failure 列表一致。

这提供了较强的 rollout-level endpoint 等价证据，但仍有两个缺口：

1. 只覆盖 LIBERO-10，没有对全部 40 tasks 重跑 custom endpoint；
2. 没有保存固定 observation/seed 下 action tensor 的 `max_abs_diff` 或 `torch.allclose` 结果。

## 11. 建议的最小修复与验证

### 11.1 首先修复确定的实现问题

1. 让 `force_custom_prefix` 真正由配置控制，删除 evaluator 和模型内部的无条件覆盖。
2. 禁止 `visualize_future_video=true` 静默绕过 prefix，或者为 prefix sampler 显式返回其实际使用的 partial video。
3. 为每次正式实验保存 Git SHA、dirty diff、完整 Hydra config 和结果 schema version。

### 11.2 增加最小单元测试

1. Route-spy 测试：断言 `T=10, N=k` 时恰好调用 k 次 joint 和 `10-k` 次 action-only。
2. 固定 observation/seed：检查 custom N=0 与 canonical wo 的 action tensor `allclose`。
3. 固定 observation/seed：检查 custom N=10 与 canonical w 的 action tensor `allclose`。
4. 固定 tensor：检查 cached wo forward 与完整 wo attention forward 的输出 `allclose`。
5. 记录每次 action prediction 实际读取的 video sigma、inference mode 和 `|Δsigma|`。

### 11.3 更匹配 alignment 研究问题的实验

建议至少补充以下对照：

1. 当前实现：`w early → wo late`。
2. 反向实现：`wo early → w late`，用于判断低噪阶段 video guidance 是否更重要。
3. Single-step ablation：每次只让一个 solver step 使用 w，定位具体 timestep 的贡献。
4. Partial-video alignment：video 先独立去噪 N 步，随后 freeze partial video，并让 action 在后续或全部 solver steps 中 attend 该视频。
5. 使用 `sigma cutoff` 或累计 `|Δsigma|` 作为横轴，而不只使用离散 N。
6. 在固定环境初始状态的基础上增加多个 diffusion seeds，报告均值和方差。

## 12. 最终判断

当前正式结果能够支持的结论是：

> 在 action diffusion 的不同前缀区间启用 joint video-action vector field，会对最终 action success rate 产生怎样的影响。

当前结果尚不能直接支持的结论是：

> 随着 future video 被逐步生成得更完整，video-action alignment 如何连续改善 action performance。

当前实现本身没有发现明确的 prefix 计数 off-by-one 或 suffix KV cache 功能错误；真正需要优先解决的是实验语义、非线性 scheduler、代码版本冻结以及两个确定的 evaluator/config 问题。
