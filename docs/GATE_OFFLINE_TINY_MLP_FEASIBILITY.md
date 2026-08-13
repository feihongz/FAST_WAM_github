# Gate Phase 3：Offline Tiny-MLP Feasibility

## 1. 决策目标

Phase 2.6 已证明，seeds 42–46 聚合得到的 Target V2 能在独立 seeds
47–50 上复现。Phase 3 回答下一个、完全不同的问题：

> 只看当前 observation、instruction 和 proprio，Tiny MLP 能否在未见 task 上预测
> multi-seed video utility？

本协议在任何 Gate feature extraction、模型训练或 OOF 预测产生前，于
`2026-08-13T15:33:59Z` 经独立终审后最终冻结。Phase 3 不修改 UniShare
training/inference，不接
LIBERO closed-loop，也不使用 future observation 或 GT action 作为 Gate 输入。

## 2. Train label 与独立考试严格隔离

```text
Target5 bundle                         Validation4 bundle
seeds 42–46                           seeds 47–50
     │                                      │
     ├─ feature-free labels                 └─ analyzer 才能读取
     ├─ fold-train / inner-val                    ↑
     └─ 封存 task-held-out OOF prediction ───────┘
```

- trainer 只能读取 seeds 42–46 的 `utility_mean/std/SEM/t95`；
- trainer、early stopping、feature normalization、模型选择和 checkpoint 选择都不能
  读取 seeds 47–50；
- feature collector/trainer 的 CLI、config 和 imports 都不得接受 Validation4 path 或
  loader；trainer 必须先原子封存 `fold_plan.json`、`oof_predictions.jsonl` 和
  `run_manifest.json`，三者 SHA 互绑后退出；
- predictions bundle 完整封存后，独立 analyzer 才加载 seeds 47–50；
- 主科学指标比较 OOF prediction 与 Validation4 mean；
- 与 Target5 mean 的指标只作为拟合诊断，不能替代独立考试结果；
- 任一 source/target/feature/prediction hash 或 fold membership 不匹配即 fail closed。

## 3. Gate 输入与泄漏边界

每个 state 只使用 action chunk inference 开始时已经可见的三个模态：

1. **visual**：严格取 dataset 的 `video[:, 0]`，使用与 UniShare 相同的 frozen VAE
   编码。VAE latent 做 `2×4` adaptive average pooling，并拼接逐 channel spatial
   standard deviation；随后使用协议固定的 Rademacher random projection 降到
   64 维；
2. **instruction**：使用现有 cached context。根据已清零的 padding row 推导有效
   token，计算 token mean 与 RMS；二者分别用固定 Rademacher projection 降到
   32 维，再拼一个有效 token fraction，共 65 维；
3. **proprio**：当前 normalized `proprio[0]`，8 维。

最终 full feature 为 `64 + 65 + 8 = 137` 维。random projection 的算法、seed、
matrix bytes SHA-256 和 extractor source SHA 都写入 manifest。模型训练中的 z-score
只在当前 fold 的 inner-train states 上拟合；常数列标准化后固定为0。

三份 projection 均由 CPU `torch.Generator` 生成 row-major `float32` Rademacher
矩阵，元素为 `{-1, +1} / sqrt(input_dim)`；不拟合数据，也不参与模型选择。固定 seed：

```text
visual projection:           20260815
instruction mean projection: 20260816
instruction RMS projection:  20260817
```

visual latent 保留 `[channel, time, height, width]` 顺序：只允许 `time=1`，先对每个
channel 做 `adaptive_avg_pool2d(2, 4)` 并按 channel-major row-major flatten，再拼每个
channel 在 `(time, height, width)` 上的 population standard deviation
（`correction=0`）。instruction active-token mask 定义为该 row 任一元素非零；mean 和
RMS 都只在 active rows 上计算，若 active count 为0则 fail closed。所有 pooling 与
projection 都以 `float32` 执行并保存 exact output SHA。

明确禁止作为 feature：

```text
future video
GT action / action_is_pad / valid_length
E0 / E10 / U / seed-level utilities
std / SEM / t interval / direction / high_confidence
selection_bin / Pilot U
suite / task / episode / frame / source index
inference latency / route result
```

其中 suite/task 只允许出现在具名 shortcut baseline 和分组评估中，不能进入 full
Gate feature。

## 4. Feature cache 与 provenance

Feature collector 对 Target V2 的 exact 100 states 逐条 rehydrate，并重新验证
`input_image/context/context_mask/proprio/valid_target_action/action_is_pad` hashes；实际
Gate feature 只从允许字段构造。

输出至少包含：

```text
manifest.json
feature_index.jsonl
features.safetensors
completion.json
```

- join key 使用 `sample_id` 与 global `requested_sample_idx/source_index`；禁止使用会在
  suite 间碰撞的 dataset-local `source_metadata.source_index`；
- feature tensors、ordered rows、target bundle、VAE、checkpoint、stats、dataset/cache、
  projection matrices、resolved config 和 scientific source files 全部内容寻址；
- atomic write，resume 时重新验证 manifest、row hash、tensor hash 和 live input hash；
- Target-100 当前每个 state 来自不同 episode。扩展 Pilot-500 时必须额外按
  `(dataset_name, episode_index)` 分组，避免同 episode 的不同 frame 跨 split。

## 5. 不使用 label 的确定性 split

### 主验证：5-fold task-held-out

group key 为 `(suite, task_index, task)`，绝不使用 bare `task_index`。每个 suite 有
10 个 task，其中5个贡献2 states、5个贡献3 states。按 identity-only stable SHA
排序，将一个2-state task与一个3-state task配成一组，再分给5个 folds。stable hash
namespace 固定为 `libero_gate_mlp_v1`；inner validation namespace 固定为
`libero_gate_mlp_inner_v1`。完整 fold membership 必须在训练前写入 manifest 并封存。

具体分配不使用任何 label：每个 suite 内分别按 stable SHA 排序2-state与3-state
tasks，同 rank 配对；再按 pair identity SHA 排序并依次赋给 fold `0..4`。对某个 outer
fold，inner validation 从其余 tasks 中按 inner namespace 排序，每个 suite 取首个
task；其余为 inner-train。

每折严格包含：

```text
8 held-out tasks
4 suites × 5 states
= 20 test states
```

同一个 task 永不同时出现在 train/test。inner validation 同样只按 task identity
选取，不看 utility、direction、HC 或 validation labels。

### 次验证：leave-one-suite-out

四折各留出一个完整 suite（25 states），只作为 distribution-shift guardrail。所有
preprocessing、target scaling 和 early stopping 仍只能在该折 train/inner-val 上完成。

## 6. 模型、loss 与固定训练策略

Primary model：

```text
137 → 32 → SiLU → Dropout(0.1)
    → 16 → SiLU → scalar utility
```

训练 target 为 Target5 `utility_mean`。每 fold 使用 inner-train median 与
`max(1.4826 × MAD, 1e-6)` 做 robust target scaling。

Primary loss：

```text
uncertainty-weighted Huber(delta=1)
    + 0.25 × pairwise logistic ranking
```

uncertainty weight 在 fold train 内计算：

```text
r_i = 1 / (1 + (SEM_i / robust_target_scale)^2)
```

随后归一化为 train mean 1 并 clip 到 `[0.25, 2.0]`。Ranking pair 只使用 Target5
95% t intervals 不重叠的 train pairs；同 task pair 权重乘2，以强调 state-level
差异。任何 weight/scale 都不能使用外折或 Validation4。

优化器、learning rate、最大 epoch、early-stop patience 和五个 initialization seeds
在 config 中固定。五 seed prediction 取 ensemble mean；不允许根据 outer result
挑选 initialization。

V1 固定训练参数：

```text
optimizer:           AdamW
learning_rate:       1e-3
weight_decay:        1e-3
batching:            full batch
gradient_clip_norm:  1.0
max_epochs:          1000
min_epochs:          100
early_stop_patience: 100
initialization_seeds: [101, 202, 303, 404, 505]
```

early stopping 只看 inner-val 上未加权 Target5 Huber；outer fold、Validation4 和最终
GO 指标均不得参与停止或 checkpoint 选择。uncertainty weights 在 clip 后再次归一化为
train mean 1，避免不同 fold 的有效 loss scale 漂移。

预注册 ablation：

- full multimodal Huber-only；
- full multimodal primary hybrid；
- visual + proprio；
- instruction + proprio（best nonvisual）；
- instruction-only。

基础 baseline：train mean constant、suite mean、unseen-task fallback lookup、以及固定
SHA salts 的 random scores。random namespace 固定为 `libero_gate_random_v1`，使用
1,000 个 salts；Baseline 使用与模型完全相同的 outer folds。

## 7. 独立 Validation4 指标

主指标全部在 sealed task-held-out OOF predictions 上计算：

- Spearman rho、Kendall tau 与 suite-stratified task-cluster bootstrap 95% interval；
- task-label permutation test；
- Validation4 actionable / Target5-HC state 的 sign balanced accuracy 与正负 recall；
- Top/Bottom 20% recall 与 Jaccard；
- 同 task、`|ΔU_validation| > 1e-4` pair 的 ordering accuracy；
- top-20% route budget 下被选 states 的 mean Validation4 utility，以及相对 matched-budget
  random selector 的 bootstrap gain；
- full model 与 best nonvisual baseline 的 paired delta；
- MAE/RMSE 与 constant baseline；
- 四个 suite cuts、leave-one-suite-out、五个 initialization 的 prediction agreement。

Bootstrap 固定2,000次，seed `20260813`，按 suite 内 task group重采样。Permutation
固定5,000次，seed `20260814`。Primary deadband 固定 `1e-4`。

所有 GO 中的相关性、tail、selected utility、within-task pair 和误差 outcome 都取
Validation4 mean。sign cohort eligibility 由训练前冻结的 Target5-HC 决定，但预测是否
正确及正负真值取 Validation4 mean 相对 `±1e-4` 的方向；Target5 direction 版本仅作
secondary diagnostic。within-task pair 预先枚举同 task 的无序 state pairs，并排除
`|ΔU_validation| <= 1e-4`；必须报告 evaluable pair 数和40-task cluster bootstrap，少于
30个 evaluable pairs 时不得判 GO。

## 8. 预注册判定

### GO：允许扩展到 Pilot-500 offline Gate training

以下全部满足：

1. task-held-out OOF vs Validation4 Spearman `>= .30`，task-bootstrap 95% 下界 `> 0`，
   permutation `p <= .05`；
2. within-task pair accuracy `>= .60`，bootstrap下界 `>= .52`；
3. Target5-HC subset 上 sign balanced accuracy `>= .70`，positive/negative recall 各
   `>= .60`；
4. Validation4 Top/Bottom 20% recall 各 `>= .40`；
5. full model 的 Spearman 比 best nonvisual 高至少 `.05`，paired task-bootstrap delta
   下界 `> 0`；
6. top-20% selected mean Validation4 utility `> 0`，且相对 matched-budget random 的
   bootstrap gain 下界 `> 0`；
7. leave-one-suite-out pooled rho `>= .20`，至少3/4 suite fold rho `> 0`，且无 fold
   `< -.20`；
8. 五个 initialization 之间 prediction Spearman 中位数 `>= .80`；
9. full model MAE 相对 train-mean constant 改善至少5%。

GO 只授权用同一 Target V2 schema 扩展后的 Pilot-500 做正式 offline Gate training；
它仍不直接授权 closed-loop。

### CONDITIONAL

若未达到 GO，但同时满足 task-held-out rho `>= .20`、HC balanced accuracy `>= .65`、
Top/Bottom recall 各 `>= .35`、within-task pair accuracy `>= .55`，且显著优于 matched
random，则只允许继续扩标签和 HC/uncertainty-weighted 小规模离线研究，不进入
closed-loop。

### NO-GO

硬完整性失败，或未满足 CONDITIONAL 任一核心下限，即停止当前 Tiny-MLP feature/
target 组合，优先研究更强视觉表征、ranking-only target 或 rollout-level utility。

## 9. 结果解释边界

即使 Phase 3 GO，也只说明 state features 在这个100-state独立 seed考试上具有离线
可学习性。它不证明：

- threshold 已校准；
- N=0/N=10 mixture 能改善 success；
- Gate latency 小于节省的 video compute；
- learned routing 优于 matched-compute random routing；
- Demo action-MSE 与 rollout utility 等价。

这些问题必须在 Pilot-500 offline复验后，再通过小规模 closed-loop smoke 与正式
threshold sweep逐级回答。

另一个明确的部署工程边界是：当前 evaluator 在调用模型之前执行 Router，而本协议的
visual feature 来自模型内部已计算的 VAE latent。即使 offline GO，也不能在 evaluator
外额外重复一次 VAE encode；后续必须新增 prepared-conditioning/feature hook，让 Gate
与 N=0/N=10 两条 prefix endpoint 复用同一份 first-frame latent。
