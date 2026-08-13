# Gate Phase 2.6：Multi-seed Utility Target V2 与独立验证

## 1. 决策目标

Phase 2.5 发现，单次 diffusion seed 的 `U = E0 - E10` 不足以作为 Gate 的直接
监督标签。Phase 2.6 不训练 MLP，也不运行 closed-loop Gate；它回答一个更具体的
问题：

> seeds 42–46 聚合得到的 state-level utility target，能否在完全独立的
> seeds 47–50 上复现？

本协议在任何 seeds 47–50 推理之前锁定。验证结果只能按照这里预注册的定义和门槛
解释，不能看到结果后修改考试标准。

## 2. Target 与 validation 的严格隔离

```text
同一组 100 个已固定 LIBERO states

seeds 42–46                       seeds 47–50
    │                                  │
    ├─ 构造 Target V2                  ├─ 独立 paired N=0/N=10 推理
    │  mean / median / uncertainty      │
    │                                  │
    └─────────── target ───────────────┘
                         ↓
                  独立复现性判定
```

- target 侧只使用 Phase 2.5 已有的 seeds 42–46；
- validation 侧固定使用 seeds 47、48、49、50，全部是新推理；
- target 构造、置信度规则和判定门槛在 validation 数据产生前固定；
- 两侧使用相同 state、GT action、checkpoint、stats、VAE、instruction、proprio、
  route 和 endpoint 参数；
- 每个 seed 内 N=0/N=10 仍共享该 replicate 的 paired inference seed；
- validation 不允许复用 seed42，也不允许根据失败或结果重新抽样 state。

## 3. Target V2 定义

对一个 state 的五个 utility：

```text
U_i = E0_i - E10_i,  i ∈ {42, 43, 44, 45, 46}
```

保存：

```text
target_mean
target_median
sample_std (ddof=1)
SEM
95% t interval (df=4)
positive / negative / deadband counts
sign agreement
direction
high_confidence
uncertain
```

主 deadband 固定为 `epsilon = 1e-4`。方向定义：

- `target_mean > +epsilon`：positive；
- `target_mean < -epsilon`：negative；
- 其余：uncertain/near-zero。

### High-confidence target

一个 state 只有同时满足以下三项才属于 high-confidence：

1. `|target_mean| > 1e-4`；
2. 五个 seed 中至少四个在 deadband 后与 target mean 同方向；
3. target mean 的双侧 95% t interval 完全高于 `+1e-4` 或低于 `-1e-4`。

该规则在已有 target 数据中覆盖 24/100 states，其中 positive 12、negative 12；
按 Pilot strata prevalence 加权后的覆盖率约为 22.7%。这些数字只证明规则有足够的
可评估样本，不参与 validation 门槛调整。

任何训练权重都只作为候选字段保存。只有独立验证达到准入标准后，它才可以成为
Gate 训练输入；本阶段不会自动启动训练。

## 4. Validation 数据与完整性要求

validation long table 固定为：

```text
100 states × 4 independent seeds = 400 rows
base seeds = [47, 48, 49, 50]
```

分析开始前必须全部满足：

- target 100 states、validation 400 rows 全齐；
- `(source_index, validation_replicate_index)` 无重复、无计划外记录；
- 每个 state 恰好 seeds 47–50，derived seed 校验正确；
- errors 为 0；
- source identity、input hashes、GT shape、valid length 在九个 seed 间一致；
- N=0/N=10 route 分别为 prefix 0/10，且共享同一个 replicate seed；
- `U = E0 - E10` 对每条记录精确成立，数值有限；
- checkpoint、dataset stats、VAE、dataset/cache 和代码 provenance 与 target/source
  bundle 一致；
- immutable manifest、selection plan、source records 和 target bundle 的 SHA-256
  均匹配。

任一完整性检查失败都 fail closed，不输出科学结论。

## 5. 预注册主比较

主分析固定比较：

```text
target = mean(U_seed42..46)
validation = mean(U_seed47..50)
```

同时报告 median/median 作为稳健性护栏。主指标：

- Spearman rho 与 Kendall tau，以及 state-bootstrap 95% interval；
- actionable state 和 high-confidence state 的方向复现率；
- positive/negative high-confidence 分层复现率；
- Top/Bottom 20% recall 与 Jaccard；
- Lin concordance correlation coefficient（CCC）；
- target mean 与 validation mean 的 absolute-agreement ICC；
- 全九个单 seed utility 的 `ICC(1,1)` 与聚合 `ICC(1,9)`；
- high-confidence raw/Pilot-weighted coverage；
- median target 的 rank 与 sign retention。

LOSO、4×5 cross-seed matrix、Bland–Altman、mean/median 交叉组合以及
`epsilon = 1e-5 / 1e-4 / 1e-3` 敏感性只作为诊断，不用于事后改变主结论。

Bootstrap 固定 2,000 次，seed 为 20260813，并按既有 Pilot selection stratum 对
state 分层重采样。

## 6. 决策门槛

### GO：允许进入常规 offline Tiny-MLP feasibility

以下条件必须全部满足：

1. mean/mean Spearman `>= .50`，95% CI 下界 `>= .30`；Kendall `>= .35`，
   CI 下界 `>= .20`；
2. actionable target sign retention `>= .75`；high-confidence retention `>= .80`；
   high-confidence positive/negative 各 `>= .70`，且各自 `n >= 10`；
3. Top/Bottom 20% recall 各 `>= .40`，Jaccard 各 `>= .25`；
4. Lin CCC `>= .50`，absolute-agreement `ICC(A,1) >= .50`；
5. 全九 seed 的 `ICC(1,9) >= .75`，并同步披露 `ICC(1,1)`；
6. high-confidence raw coverage 至少 20 states，Pilot-weighted coverage `>= .20`；
7. median/median Spearman `>= .40`，actionable median sign retention `>= .70`。

GO 只授权开始 offline Gate feasibility，不证明 closed-loop Gate 有效。

### CONDITIONAL：只允许高置信小规模 feasibility

若未达到 GO，但以下全部满足：

- mean/mean Spearman `>= .30`；Kendall `>= .20`；
- CCC `>= .30`；`ICC(A,1) >= .35`；
- actionable sign retention `>= .65`；high-confidence retention `>= .70`；
- Top/Bottom recall 各 `>= .35`；
- 全九 seed `ICC(1,9) >= .65`；
- high-confidence 至少 10 states、Pilot-weighted coverage `>= .10`；
- median/median Spearman `>= .30`。

则只允许在 high-confidence/uncertainty-weighted 子集上做小规模 offline feasibility，
不得直接训练最终 Gate、不得进入 closed-loop。

### NO-GO

硬完整性失败，或未满足 CONDITIONAL 的任一核心下限，即为 NO-GO。此时停止当前
demo-action-MSE target 路线，优先研究 shared-random-path、更多重复、ranking-only
目标或 rollout-level utility。

## 7. 若通过后怎么训练

只有 GO（或明确受限的 CONDITIONAL）后才创建训练 PR。训练阶段至少包含：

- task-held-out validation；
- task-only、constant、random-score baseline；
- Huber regression 与 ranking objective 对比；
- uncertainty weighting / high-confidence filtering 消融；
- 先 offline ranking/calibration，再 closed-loop learned-vs-random router。

即使 Target V2 达到 GO，也不代表 Gate 一定能从 observation/text/proprio 中学到它，
更不代表 closed-loop success–compute Pareto 一定改善。
