# Gate Phase 2.5：Demo Utility 多 Seed 稳定性审计

## 1. 为什么 Phase 2 后不能马上训练 MLP

Phase 2 的 Pilot-500 已经证明 demonstration state 中存在正、负两类
`U = E0 - E10`，而且同一个 task 内也有明显差异。但每个 state 当时只运行了一个
inference seed。即使 N=0 与 N=10 在一次 paired inference 内共享相同 action noise，
也不能据此证明 `U` 跨 noise seed 稳定。

Phase 2.5 的目标是回答一个更窄、但决定后续标签是否可信的问题：

> 同一个 state 换不同 diffusion seed 后，“video 有用还是有害”的符号、相对排序和
> 强信号集合是否仍然稳定？

这一阶段仍然不训练 MLP，不修改 UniShare、`training_loss()`、现有 Gate Router 或
LIBERO evaluator。

## 2. 审计单位和计算量

固定选择 100 个 Pilot state，每个 state 使用 5 个 replicate base seed：

```text
replicate 0: base_seed=42  （精确复用 Pilot-500 record）
replicate 1: base_seed=43  （新 paired inference）
replicate 2: base_seed=44  （新 paired inference）
replicate 3: base_seed=45  （新 paired inference）
replicate 4: base_seed=46  （新 paired inference）
```

每个 replicate 内仍通过 `stable_sample_seed(base_seed, source_identity)` 得到实际
inference seed，并保证 N=0 与 N=10 共享完全相同的 seed。最终 long table 为
`100 states × 5 replicates = 500 rows`，但 GPU 只新增 400 次 paired inference。

## 3. 固定选择方案

选择只使用 Pilot 的 seed=42 utility，不查看未来 4 个 seed 的结果。100 个 state
满足以下硬约束：

- 4 个 LIBERO suite 各 25 个；
- 40 个 task 全覆盖，每个 task 2～3 个；
- 每个 episode 最多 1 个 state；
- `valid_length < 32` 恰好 16 个，与 Pilot-500 的 16% 比例一致；
- 五个 utility stratum 的固定配额：
  - `SP`: `U > 1e-3`，25 个；
  - `SN`: `U < -1e-3`，25 个；
  - `MP`: `1e-4 < U <= 1e-3`，13 个；
  - `MN`: `-1e-3 <= U < -1e-4`，12 个；
  - `NZ`: `|U| <= 1e-4`，25 个。

suite × stratum 配额、100 个有序 source index、sample identity、输入 hash 和选择计划
SHA-256 全部写入 immutable manifest。运行中不允许因读取失败而重抽样。

这是刻意提高 tail coverage 的诊断样本，不是 Pilot population 的简单随机样本。
因此 unweighted overall 只用于稳定性诊断；需要总体比例时，按 Pilot-500 五层占比
`SP=.152, SN=.202, MP=.262, MN=.184, NZ=.200` 加权。

## 4. 输出和断点续跑约束

多 seed 使用独立 Collector 和独立 manifest，不复用 single-seed Collector 的
`sample_id` 唯一键。每行唯一键为：

```text
(requested_source_idx, replicate_index)
```

每条记录除 Phase 2 原有的 `E0/E10/U`、route、seed、input hashes 和 provenance
之外，还保存：

```text
replicate_index
replicate_base_seed
replicate_id
reused_from_pilot
source_pilot_manifest_fingerprint
source_pilot_record_sha256
inference_origin
```

恢复前会验证 Pilot manifest/records 内容 hash、当前 checkpoint/stats/VAE/dataset/text
cache、100-state plan 和已有复合键。对同一个 state，5 行的 source identity、当前
输入 hash 和 normalized proprio 必须完全一致。任意缺行、重复、plan 外行、route 错误、
seed 错误或 provenance 漂移都 fail closed。

## 5. 稳定性指标

每个 state 计算 5 个 `U` 的 mean、median、standard deviation、SEM、95% t interval、
正负计数和 sign agreement。主 deadband 为 `|U| <= 1e-4`，并同步检查 `1e-5` 与
`1e-3` 敏感性。

主要验收指标：

- strong positive/negative state 的 5-seed majority 是否保持 Pilot 符号；
- seed42 `U` 与其余 4 seed mean 的 Spearman/Kendall rank correlation；
- leave-one-seed-out rank correlation；
-每个 seed 与其余 seed mean 的 Top-20 / Bottom-20 recall 和 Jaccard；
-随机截距方差分解及 `ICC(1,1)`、`ICC(1,5)`；
-按 Pilot stratum prevalence 加权的 strong-signal 稳定比例。

## 6. 预先固定的决策门槛

### GO：可以进入 offline Tiny MLP feasibility

以下条件必须同时满足：

1. SP+SN majority 与 seed42 符号一致率 ≥ 80%，且 SP/SN 各 ≥ 75%；
2. strong state 中至少 70% 有 ≥4/5 seed 同方向；
3. seed42 vs other-four mean 的 Spearman `rho >= .50`，state-bootstrap 95% CI
   下界 ≥ .30；
4. LOSO Spearman 中位数 ≥ .40，且 5 个中至少 4 个为正；
5. Top-20 与 Bottom-20 recall 均 ≥ .40，Jaccard 均 ≥ .25；
6. `ICC(1,5) >= .75` 且 `ICC(1,1) >= .35`；
7. Pilot-weighted strong-mean states 中，至少 60% 达到 ≥4/5 同符号。

GO 只授权下一步做 offline Tiny MLP/ranking feasibility，不代表 Gate 已在 closed-loop
有效，更不授权跳过 random-router 与 threshold sweep。

### CONDITIONAL：先扩到 9 seeds

若未达到 GO，但 strong majority ≥65%、`rho >= .30`、`ICC(1,5) >= .50`，则不把
单 seed label 直接用于最终训练；先扩到 9 seeds，并用 multi-seed mean/median 与更大
deadband 重新定义标签。

### NO-GO：停止单 seed MLP

strong majority <65%、`rho < .30` 或 `ICC(1,5) < .50` 任一成立，即停止当前标签方案，
优先重定义 utility 或引入 rollout-level supervision。

NZ 不要求稳定。若一个 NZ state 的多数 seed 仍处于 deadband，或 mean interval 跨 0，
它应标为 ambiguous，而不是强行作为正/负分类标签。

## 7. 运行顺序

实现完成后严格按以下顺序：

1. CPU unit/integration tests；
2. 1 state × 5 seed GPU smoke，验证 rep0 零推理复用、4 个新 replicate 恰好 8 次 endpoint
   调用和显存；
3. 独立 output 目录正式运行 100 state × 5 seed；
4. 完整性审计通过后才执行统计分析；
5. 生成 technical HTML report，并给出 GO / CONDITIONAL / NO-GO。

单张 H100 只运行一个 Collector。模型只加载一次，按 state 外循环、replicate 内循环。
多 worker 会复制模型并争抢同一 GPU 计算资源，通常不会加速。
