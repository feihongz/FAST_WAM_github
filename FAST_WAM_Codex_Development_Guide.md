# FAST-WAM Stage 2 / Stage 3：给 Codex 代码工程师的开发建议

> 目标仓库：`feihongz/FAST_WAM_github`  
> 核对版本：`main@bf9262d90eaf08ecee9e72d79b92d59f7c2259b0`  
> 文档性质：实施规格与验收清单，不是论文表述

## 1. 任务一句话定义

在现有 UnifiedShared WAM 上增加两个彼此独立的轻量模块：

- **Stage 3 Action Alignment Adapter**：只修正 w-video 分支，让模型自己预测的未来视频对 action 产生正确、无害的影响；原 wo-video 分支必须保持不变。
- **Stage 2 Binary Gate**：在每次 action chunk 推理前，仅根据当前观测、指令和 proprio，选择 `N=0`（wo-video）或 `N=10`（w-video）。

最终系统只保留两个运行端点，不实现 `N=1...9` 的路由。

## 2. 开工前必须统一的语义

### 2.1 `N=0` 和 `N=10` 的真实含义

当前代码中：

- `N=0`：`inference_mode="wo"`，不做 future-video denoising；**action solver 仍运行 10 步**。
- `N=10`：`inference_mode="w"`，video 和 action 在同一个 10-step solver 循环中联合更新。

因此，`N` 表示 **video NFE**，不是 total NFE，也不是视频帧数。

当前典型配置为：

```text
num_frames = 33
action_video_freq_ratio = 4
num_video_frames = 9
num_inference_steps = 10
```

也就是说，`N=10` 通常预测 9 帧视频，但执行 10 次 video denoising。代码变量中必须区分：

```text
num_video_frames      # 视频帧数，例如 9
num_inference_steps   # solver 次数，固定为 10
selected_video_nfe    # Gate 选择带来的 video NFE，0 或 10
```

### 2.2 N=10 不是“先生成完整视频，再预测动作”

`FastWAMJoint.infer_action()` 每个 solver step 都会用当前 video latent 更新 action latent。因此 Adapter 必须在这 10 次 joint update 中每一步都生效；训练也要覆盖内部 `k=0...9` 的 video 状态。

Gate 仍然只选择完整的 `N=0` 或 `N=10`。训练 Adapter 时采样内部 `k`，不等于增加中间 N 的推理模式。

### 2.3 最终开发顺序

```text
现有 UnifiedShared checkpoint
        ↓
先训练 Stage 3 Adapter
        ↓
用最终 Adapter 重新计算 E0 / E10 标签
        ↓
再训练最终 Stage 2 Gate
        ↓
固定所有权重，扫描 Gate threshold 做 closed-loop Pareto eval
```

Stage 3 一旦重训，旧的 Stage 2 标签和 Gate 都视为过期。

## 3. 当前仓库已有能力与工程缺口

关键现有文件：

| 文件 | 当前能力 | 本任务相关限制 |
|---|---|---|
| `src/fastwam/models/wan22/fastwam_unified_shared.py` | 已有 wo/w 两种 mask 和 `infer_action_mode()` | w/wo 共享同一 `mot` 和 `action_expert`，不能直接微调 w 骨干 |
| `src/fastwam/models/wan22/fastwam_joint.py` | 实现 10-step video/action joint rollout | Adapter 必须进入每一步 `_predict_joint_noise()` |
| `src/fastwam/trainer.py` | 通用 WAM 训练 | 当前会强制解冻整个 `model.dit`，不能直接用于 Adapter/Gate |
| `src/fastwam/models/wan22/fastwam.py` | 基础推理与 checkpoint | checkpoint 只显式保存 `mot`、`proprio_encoder` |
| `src/fastwam/datasets/lerobot/robot_video_dataset.py` | 已提供 GT video/action/context/proprio | 标签生成需要稳定 sample/episode identity；当前异常回退随机样本不适合制标签 |
| `experiments/libero/eval_libero_single.py` | LIBERO 静态 wo/w 推理 | 需增加 per-chunk route、日志和计时 |
| `experiments/robotwin/fastwam_policy/deploy_policy.py` | RoboTwin action queue 推理 | 需在 action queue 重新填充时路由一次 |
| `configs/sim_libero.yaml`、`configs/sim_robotwin.yaml` | 只有静态 `inference_mode` | 需增加 gate/random/static 配置 |

必须避免两个错误：

1. 把 Stage 3 loss 塞进现有 `Wan22Trainer`，结果整个共享 DiT 被解冻，wo-video 性能漂移。
2. 只把 Adapter/Gate 注册到模型，却没有显式保存和加载，训练完成后 eval 实际仍在跑原模型。

## 4. 建议的代码组织

优先新增独立模块，尽量不破坏旧类和旧 checkpoint：

```text
src/fastwam/models/wan22/
├── fastwam_unified_aligned.py       # 新模型，继承 UnifiedShared
└── video_action_alignment.py        # Stage 3 Adapter

src/fastwam/models/
└── video_gate.py                    # Stage 2 Binary Gate

src/fastwam/alignment/
├── losses.py                        # masked per-sample loss
├── rollout.py                       # frozen self-video rollout
└── cache_dataset.py                 # 可选的 rollout cache

src/fastwam/evaluation/
├── routing.py                       # static/gate/random router
└── metrics.py                       # route、NFE、latency 聚合

scripts/
├── train_stage3_alignment.py
├── generate_gate_labels.py
└── train_stage2_gate.py

experiments/
├── plot_pareto.py
├── libero/run_pareto_sweep.py
└── robotwin/run_pareto_sweep.py

tests/
├── test_alignment_adapter.py
├── test_alignment_losses.py
├── test_freeze_contract.py
├── test_video_gate.py
├── test_routing.py
├── test_routing_metrics.py
└── test_pareto.py
```

不要把 Stage 2、Stage 3 继续堆进 `fastwam_unified_shared.py` 的一个大 `training_loss()` 中。旧 UnifiedShared 应作为冻结基线保留。

## 5. Stage 3：Action Alignment Adapter

### 5.1 网络结构

建议新建：

```python
class VideoActionResidualAdapter(nn.Module):
    ...
```

输入与输出：

```text
action hidden:       [B, T_action, 1024]
future video hidden: [B, T_video_tokens, 3072]
adapter correction:  [B, T_action, action_dim]，即 action flow velocity residual
```

建议配置：

```yaml
alignment:
  bottleneck_dim: 256
  num_heads: 8
  ffn_multiplier: 2
  drop_first_video_frame: true
  zero_init_output: true
```

实现方式：

1. 按 `tokens_per_frame` 整理 video tokens。
2. 去掉第一帧，只让 Adapter 读取 future-video tokens。
3. 对每帧空间 token 做 pooling，控制显存。
4. action hidden 投影到 256 维 query；video hidden 投影到 256 维 key/value。
5. 做 8-head cross-attention 和一个小 FFN。
6. `256 → action_dim` 的输出 projection 零初始化，以 residual 形式加到冻结的 w-video action velocity。

参数量目标约 2–5M。零初始化必须保证：加载原 checkpoint、尚未训练 Adapter 时，w-video 输出与旧模型数值等价。

### 5.2 模型接入点

新建：

```python
class FastWAMUnifiedAligned(FastWAMUnifiedShared):
    ...
```

在 w-video 路径中：

```python
base_action_tokens = tokens_out["action"]
pred_action_w_base = action_expert.post_dit(base_action_tokens, action_pre)
delta_velocity = alignment_adapter(
    action_tokens=base_action_tokens.detach(),
    video_tokens=tokens_out["video"].detach(),
    video_meta=video_pre["meta"],
)
pred_action_w = pred_action_w_base + delta_velocity
```

wo-video 路径不得调用 Adapter。

推理时需要让 Adapter 进入 `FastWAMJoint.infer_action()` 的每次 `_predict_joint_noise()`。推荐在基础 joint forward 中抽出一个 action-velocity hook；它同时接收 post-DiT 的基础 velocity 和 MoT hidden，旧模型默认原样返回，Aligned 子类只在 `_unified_inference_mode == "w"` 时加 residual。若不愿重构基础类，可以在 Aligned 子类 override `_predict_joint_noise()`，但不要让训练和推理各维护一套不同的 Adapter 逻辑。

推理时每个 query 只能执行一个分支：

```text
选择 wo → 只运行冻结 wo
选择 w  → 只运行冻结 w base + Adapter
```

绝不能先计算 wo action，再计算 w action 后做融合。`u0` 只在训练时作为 detached anchor 使用。

### 5.3 Stage 3 的一条训练样本

现有 demonstration 已提供：

```text
当前帧、GT future video、指令、proprio、GT action、padding mask
```

首版每个 microbatch 均匀采一个内部 step：

```python
k = randint(0, 10)
```

然后构造三条 action-velocity 路径。三条路径必须共享完全相同的：

```text
noisy action latent
action noise
action timestep
action flow target
padding mask
```

三路分别为：

```text
v0       = 冻结 wo-video action velocity
v_gt     = 冻结 w-video + GT future state 的 teacher velocity
v_self   = 冻结 w-video + self-rollout future state + 可训练 Adapter
v_target = 当前 flow-matching 的 GT action velocity target
```

这里使用 `v_*`，不要在 Stage 3 内部把 `v_self` 命名为 `E10`。Stage 2 的 `E10` 是完整推理后最终 action chunk 的误差，两者层级不同。

### 5.4 self-rollout future 的构造

N=10 推理中 action 会看到每一个内部 video state，因此训练必须覆盖这些状态。

建议实现：

1. 用冻结 video generator，从相同的 video noise 开始。
2. 固定 first-frame latent。
3. 在 `torch.no_grad()` 下运行前 `k` 个 video solver step，得到 `z_self_k`。
4. 用同一 video noise 和同一 `sigma_k` 将 GT future latent 加噪，得到 `z_gt_k`。
5. `v_gt` 使用 `z_gt_k`，`v_self` 使用 `z_self_k`。

当前模型配置 `action_conditioned: false`，理论上可以只 rollout video rows，避免无用的 action forward；实现时必须加配置断言，并用测试验证 video-only rollout 与 joint MoT 中的 video velocity 等价。

不要使用 `torch.inference_mode()` 生成后续要被 Adapter autograd 保存的 tensor；使用 `torch.no_grad()` 并显式 `.detach()`，否则可能触发 inference-tensor autograd 错误。

若在线 rollout 太慢，增加离线 cache，但不要缓存解码后的 RGB：

- 缓存 fp16/bf16 video latent 和 `k/sigma/seed/sample_id`。
- 每个样本只缓存一个由 hash 决定的 `k`，让全数据均匀覆盖 0–9；多 seed 时再覆盖其他 `k`。
- shard 写入、支持断点续跑，并在 manifest 记录 base checkpoint hash、代码 commit 和 scheduler 配置。

### 5.5 Loss

所有误差必须先按 `action_is_pad` 做 masked、per-sample MSE：

```text
e0_i    = MSE(v0_i,     v_target_i)
egt_i   = MSE(v_gt_i,   v_target_i)
eself_i = MSE(v_self_i, v_target_i)
```

判断 GT future 是否有帮助：

```text
g_i = 1[egt_i < 0.95 × e0_i]
```

正确修正目标：

```text
target_delta_i = g_i × stopgrad(v_gt_i - v0_i)
```

损失：

```text
L_action = weighted_mean(eself)

L_align = masked_MSE(
    v_self - v0,
    target_delta
)

L_safe = mean(relu(eself - stopgrad(e0)))

L_stage3 = L_action + 1.0 × L_align + 0.5 × L_safe
```

当 GT future 有帮助时，Adapter 学习 GT future 造成的有益 action correction；无帮助时，`target_delta=0`，w-video action 被拉回 wo-video anchor。

首版不加入 video reconstruction loss、不训练 Gate、不加入中间 N 的单调性 loss。

### 5.6 冻结与优化器合同

Stage 3 只允许以下参数可训练：

```text
alignment_adapter.*
```

建议新建专用 `AlignmentTrainer`，不要复用当前 Trainer 的 dit-only 冻结逻辑。初始化优化器前执行：

```python
model.eval()
model.requires_grad_(False)
model.alignment_adapter.train()
model.alignment_adapter.requires_grad_(True)

named_trainable = {
    name for name, p in model.named_parameters() if p.requires_grad
}
assert named_trainable
assert all("alignment_adapter" in name for name in named_trainable)
```

每次 optimizer step 后，debug 模式抽查：

```text
所有非 Adapter 参数 grad is None
wo 分支关键参数 checksum 未变化
固定 seed 的 wo 输出与训练前一致
```

起始训练配置建议：

```yaml
learning_rate: 1.0e-4
weight_decay: 1.0e-4
mixed_precision: bf16
max_grad_norm: 1.0
lambda_action: 1.0
lambda_align: 1.0
lambda_safe: 0.5
helpful_relative_margin: 0.05
num_solver_steps: 10
```

由于每个 iteration 包含 self-rollout 和三路骨干 forward，先从每 GPU batch size 1 开始，以 gradient accumulation 补有效 batch。

### 5.7 Stage 3 checkpoint

推荐保存独立轻量文件，不重复保存冻结的 5B base：

```python
{
    "schema_version": 1,
    "adapter": adapter.state_dict(),
    "optimizer": ...,              # 仅训练 state checkpoint 需要
    "base_checkpoint": "...",
    "base_checkpoint_sha256": "...",
    "git_commit": "bf9262d...",
    "alignment_config": {...},
    "global_step": ...,
}
```

推理加载顺序固定为：base WAM → alignment Adapter。hash 不匹配时默认报错，只有显式 override 才允许继续。

## 6. Stage 2：Binary Video Gate

### 6.1 Gate 网络

建议约 0.65M 参数：

```text
当前拼接图像 → tiny CNN → 128
T5 context masked mean，4096 → 128
normalized proprio → MLP → 32
concat 288 → MLP 128 → 1 logit
```

接口：

```python
logit = gate(
    input_image,   # [B, 3, H, W]，只含当前观测
    context,       # [B, L, 4096]
    context_mask,  # [B, L]
    proprio,       # [B, D]
)
```

禁止输入：预测 future、GT future、video quality、E0、E10 或任何生成后特征。

若 eval 输入为 prompt 字符串，先调用一次 `model.encode_prompt()`；Gate 和被选中的 action 分支复用同一份 `context/context_mask`，避免重复 T5。不要先把 proprio append 到 context，否则 action 路径可能重复追加。

### 6.2 生成 Gate 标签

标签必须在 Stage 3 最终 checkpoint 后生成。对每条 demonstration，用相同 action seed 分别完整推理：

```python
a0 = aligned_model.infer_action_mode(
    ...,
    inference_mode="wo",
    num_inference_steps=10,
)

a10 = aligned_model.infer_action_mode(
    ...,
    inference_mode="w",
    num_video_frames=sample["video"].shape[1],
    num_inference_steps=10,
)
```

注意：wo 分支不能收到 `num_video_frames`。

在模型归一化 action 空间计算 padding-aware MSE：

```text
E0  = masked_MSE(a0,  a*)
E10 = masked_MSE(a10, a*)
y   = 1[E10 < 0.95 × E0]
```

建议每条样本使用 2–4 个相同 seed pair 后取平均，降低随机标签噪声。接近边界的样本可以保留但降低 sample weight；第一版也可以直接按上述 hard label 实现。

label manifest 至少保存：

```text
sample_id / dataset_id / episode_id / frame_id
E0、E10、relative_gain、label、sample_weight
seeds、margin、num_inference_steps、num_video_frames
base checkpoint hash、Adapter hash、Git commit、数据配置
```

标签脚本必须支持多进程分片、断点续跑和 shard merge。

数据层还需做两点：

1. 在 processor/dataset 中保留稳定的 `idx`、episode 和 frame identity，训练/验证按 episode 划分，不能随机拆相邻帧。
2. label/cache 生成启用 strict data mode；不要沿用 `RobotVideoDataset.__getitem__()` 出错后随机换样本的行为，否则 label 会绑定到错误 sample id。

### 6.3 Gate 训练

Gate 训练不需要加载 5B WAM，只读取原始轻量输入和 label manifest。

```python
loss = BCEWithLogitsLoss(
    pos_weight=num_negative / num_positive,
)
```

建议配置：

```yaml
learning_rate: 1.0e-4
weight_decay: 1.0e-4
num_epochs: 5
early_stop_patience: 2
batch_size: 64
```

记录：train/val BCE、AUROC、AUPRC、positive rate、calibration error。类别不平衡时使用 `pos_weight`，不要简单复制正样本。

Gate 使用独立 checkpoint，并记录其 label manifest hash 与 Adapter hash。

### 6.4 推理 Router

建议新增共享的 `BinaryVideoRouter`，支持：

```text
routing_mode = static | gate | random
```

- `static`：复现原始 wo/w 两个端点。
- `gate`：用于最终方法和 threshold sweep。
- `random`：按相同 w-video 使用率构造必要 baseline。

Gate 规则统一为：

```python
selected_mode = "w" if sigmoid(logit) >= threshold else "wo"
```

一次 policy query 只能 route 一次，并且只执行被选分支。返回或记录：

```text
gate_score
selected_mode
selected_video_nfe = 0 or 10
gate latency
model latency
total latency
```

## 7. LIBERO 与 RoboTwin 接入

### 7.1 LIBERO

修改：

- `experiments/libero/eval_libero_single.py`
- `experiments/libero/summarize_results.py`
- `configs/sim_libero.yaml`

精确流程：

1. 在 `eval_single_process()` 中加载一次 Gate/Router。
2. 在 `_predict_action_chunk()` 中，每次 replan 前 route 一次。
3. 只有 w 分支传 `num_video_frames`。
4. `run_single_episode()` 返回 episode routing metrics。
5. task/result JSON 保存 route count、video NFE 和 latency。
6. `summarize_results()` 用总计数加权聚合，不能平均各 task 的 mean。

Gate/Pareto eval 时禁止 `visualize_future_video=true`；当前该选项会直接走 `infer_joint()`，从而绕开 Router。

### 7.2 RoboTwin

修改：

- `experiments/robotwin/fastwam_policy/deploy_policy.py`
- `experiments/robotwin/fastwam_policy/deploy_policy.yml`
- `experiments/robotwin/eval_robotwin_single.py`
- `experiments/robotwin/run_robotwin_manager.py`
- 必要时修改 vendored RoboTwin 的 episode metrics 输出

正确路由位置是 `WorldActionRobotWinPolicy._infer_action_chunk()`，即 action queue 需要重新填充时。不能每执行一个 environment action 都重复计 Gate query。

每个 episode 结束后写 JSON metrics；现有 timing 如果只使用 `perf_counter()` 而没有 CUDA synchronize，会低估 GPU 推理时间。

## 8. Evaluation 配置建议

同时在 LIBERO、RoboTwin 的 YAML 中声明字段，避免 Hydra override 不存在的 key：

```yaml
EVALUATION:
  routing_mode: static       # static | gate | random
  inference_mode: wo         # static 模式使用
  gate_checkpoint: null
  alignment_checkpoint: null
  gate_threshold: 0.5
  random_video_probability: 0.5

  num_inference_steps: 10
  timing_enabled: true
  timing_warmup_queries: 3
  save_query_metrics: true
```

## 9. Pareto 曲线怎么生成

训练结束后固定 base、Adapter 和 Gate，只改变 eval 的 `gate_threshold`。

先在 validation gate score 上按分位数选择 threshold，使预期 w-video 使用率约为：

```text
0%、10%、20%、40%、60%、80%、100%
```

然后锁定这些 threshold，每个 threshold 都完整重跑 closed-loop evaluation。不能在同一批 rollout 结束后离线换 threshold，因为不同路由会改变后续状态和轨迹。

两个端点应显式运行：

```text
static wo
static w
```

不要用 threshold 0/1 冒充静态端点。

主横轴：

```text
avg_video_nfe_per_query
    = total_selected_video_nfe / num_policy_queries
    = 10 × n_w / (n_wo + n_w)
```

主纵轴：success rate。

同时报告：

```text
video_rate
total_video_nfe_per_episode
latency mean / p50 / p95 per query
episode success
```

因为不同 threshold 可能改变 episode 长度，只报每 query NFE 不够，还需报告每 episode 总 video NFE。

计时必须在前后调用 `torch.cuda.synchronize()`；真实 latency sweep 时一张 GPU 只跑一个 eval 进程。每个 threshold 建议至少 3 个 eval seeds，并在 episode/seed 层 bootstrap 95% CI。

绘图包含：

- Static N=0、Static N=10。
- Random routing baseline。
- Gate only（如需要做 ablation）。
- Gate + Stage 3 Adapter（最终方法）。
- 所有 raw points 与置信区间。
- 删除 dominated points 后的 Pareto frontier。

输出 `pareto_points.csv`、`pareto.png` 和 `pareto.pdf`。

## 10. 必做测试

### 10.1 Stage 3

1. Adapter zero-init 输出严格为 0；新 Aligned 模型初始 w 输出等价旧模型。
2. 一个 optimizer step 后，所有非 Adapter 参数 `grad is None` 且 checksum 不变。
3. 固定 seed 的 wo 输出在 Stage 3 训练前后保持一致。
4. helpful 样本的 alignment target 为 `v_gt-v0`；unhelpful 样本 target 为 0。
5. action padding token 不进入任何 error/loss。
6. safe hinge 在 `eself<=e0` 时为 0，在更差时为正。
7. Adapter 在 N=10 的 10 次 joint step 中均被调用，N=0 调用次数为 0。
8. Stage 3 checkpoint roundtrip 后 Adapter 权重完全恢复。
9. Accelerate/ZeRO 单 batch smoke test：loss finite、仅 Adapter 变化、resume 成功。

### 10.2 Stage 2 与路由

1. Gate 输出 shape、参数量和输入限制正确。
2. label rule、padding mask、相同 seed pair 正确。
3. wo 分支不会收到 `num_video_frames`，w 分支必须收到。
4. 一次 query 只调用一个 action 分支。
5. `routes=[0,10,10]` 时 `avg_video_nfe=20/3`。
6. metrics merge 按 query count 加权。
7. manifest 中 Adapter/checkpoint hash 不匹配时拒绝训练。
8. static wo/w 固定 seed 与修改前行为一致。
9. LIBERO dummy integration 与 RoboTwin action-queue integration。
10. Pareto dominance、同 compute 取更高 success、CSV/JSON roundtrip。

仓库当前没有自有 `tests/`，需要补 pytest dev dependency；GPU/simulator smoke tests使用单独 marker，不进入普通 CPU 单测。

## 11. 建议拆成四个 PR

### PR 1：接口与无回归重构

- 新增 trainable-parameter hook 或专用 Trainer 基类。
- 抽出 w-path action-velocity Adapter hook。
- 补 wo/w dispatch、checkpoint 和固定 seed 回归测试。
- 此 PR 不改变任何已有模型输出。

### PR 2：Stage 3

- Adapter、Aligned 模型、self-rollout、loss、专用 Trainer、配置和轻量 checkpoint。
- 完成 gradient isolation 与全 10-step 调用测试。
- 离线比较训练前后最终 action `E0/E10`。

### PR 3：Stage 2

- 稳定 sample identity、标签生成、manifest、Gate dataset、Gate trainer 和 checkpoint。
- 标签必须引用 PR 2 的最终 Adapter hash。

### PR 4：Router、评测与 Pareto

- LIBERO/RoboTwin per-chunk routing。
- static/gate/random 三种模式。
- query/episode metrics、threshold sweep、plot 和 CI。

每个 PR 都应单独可运行、可回滚，不能用一个巨大 mode flag 同时改变训练、推理和评测。

## 12. 最终验收标准

只有以下条件全部满足才算完成：

1. Stage 3 optimizer 中只有 Adapter 参数。
2. Stage 3 后固定 seed 的 wo action 与 base checkpoint 一致。
3. N=10 每个 joint step 都启用 Adapter，N=0 完全绕过 Adapter。
4. Gate 输入中不存在任何 future 信息。
5. Gate label 来自最终 Stage 3 checkpoint，且 manifest hash 可追溯。
6. 每个 policy query 恰好执行 wo 或 w 中的一个分支。
7. `avg_video_nfe = 10 × n_w / num_queries`，统计从 query log 到 summary 再到图完全一致。
8. static wo/w 能复现原始两个端点。
9. 每个 threshold 都完成独立 closed-loop evaluation。
10. 最终同时交付 success–video-NFE Pareto 和 success–latency 曲线。

## 13. 建议 Codex 工程师每次提交时附带的结果

```text
1. 修改文件列表和设计说明
2. trainable parameter names 与参数量
3. 单元测试命令和结果
4. 一个小数据 batch 的 loss/log 示例
5. wo-invariance 数值对比
6. Stage 3 / Gate checkpoint schema 示例
7. 一个 dummy Pareto summary 与绘图 smoke test
8. 尚未完成项和已知风险
```

最高优先级不是先跑大规模训练，而是先证明三个合同：**wo 不变、只有 Adapter 有梯度、一次推理只执行一个分支**。这三个合同通过后，再投入完整 Stage 3 rollout 和 Gate label 生成。
