# Demo Video-Utility 多随机种子稳定性审计结果

## 可移植报告与审计证据

- [交互式 HTML 技术报告](reports/gate_utility_multiseed_100x5/report.html)
- [Canonical report artifact](reports/gate_utility_multiseed_100x5/artifact.json)
- [正式分析摘要](reports/gate_utility_multiseed_100x5/analysis_summary.json)
- [独立复算审计](reports/gate_utility_multiseed_100x5/independent_audit.json)

HTML 已通过 canonical artifact validation 与 portable packaging；当前运行环境没有 Chromium，因此浏览器级 QA 状态为 `structural_only`，不是统计或内容校验失败。

## 技术结论

**结论是 `NO_GO`：当前单随机种子 Demo utility 标签不应直接进入 Tiny MLP 训练。** 这一结论只回答“现有标签是否足够稳定，可以开始离线 Gate 拟合”，不代表 UniShare 的 video 分支无用，也不代表动态 Gate 在 closed-loop LIBERO 中一定无效。

正式审计覆盖 100 个 demonstration state、5 个推理随机种子，共 500 条配对 `N=0/N=10` 记录；其中 seed 42 的 100 条结果严格复用 Pilot，seed 43–46 新推理 400 条。数据网格完整，缺失 0 条、错误 0 条。预注册的 15 项 GO 检查仅通过 5 项、失败 10 项；同时，Conditional GO 的关键下限也未达到：strong-state 新种子多数方向一致率为 0.62，低于 0.65。

当前结果最适合支持以下工程决策：

1. 暂停用单次 `U = E0 - E10` 作为 Tiny MLP 的直接监督标签。
2. 优先把标签改成多 seed 聚合，并显式建模不确定性。
3. 进一步加强 `N=0/N=10` 的 common-random-number / shared-random-path 配对，降低差分标签的方差。
4. 用相同预注册门槛重新审计新标签；通过后才进入离线 Tiny MLP，更不能跳过该关直接做 closed-loop Gate。

## 关键发现：尾部排序有信号，但逐状态方向不够可靠

### 1. Strong-state 方向保持不足，是最直接的停止信号

在 Pilot 中属于强正/强负 utility 的 50 个状态上，seed 43–46 的多数方向与 Pilot 方向一致率只有 **0.62**，95% bootstrap 区间为 **[0.48, 0.74]**，未达到预注册的 0.80。拆开看：

| Pilot strata | 状态数 | 新 4 seed 多数方向保持率 | GO 门槛 | 结果 |
|---|---:|---:|---:|---|
| SP：`U > 1e-3` | 25 | 0.52 | 0.75 | 失败 |
| SN：`U < -1e-3` | 25 | 0.72 | 0.75 | 失败 |
| Strong overall | 50 | 0.62 | 0.80 | 失败 |

这说明 Pilot 中“看起来很强”的正向状态，跨随机种子后尤其不稳定。独立 post-hoc 审计进一步拆解为：SP 中 13/25 保持原方向、4/25 明确反向、8/25 不明确；SN 中 18/25 保持原方向、0/25 明确反向、7/25 不明确。该拆解不是预注册判定项，只用于定位问题。

### 2. 排名稳定性中等，尚不足以支撑连续 utility regression

seed 42 与独立 seed 43–46 平均 utility 的 Spearman 相关为 **0.438**，低于 0.50 门槛；其 bootstrap 95% 下界为 **0.231**，低于 0.30 门槛。五次 leave-one-seed-out 的 Spearman 全部为正，但中位数只有 **0.388**，略低于 0.40 门槛。

尾部集合比全排序更稳定：top-20% recall/Jaccard 为 **0.45/0.290**，bottom-20% 为 **0.55/0.379**，四项均通过门槛并明显高于随机期望（recall 0.20、Jaccard 0.111）。因此数据不是“完全没有信号”，但目前更像粗粒度尾部筛选信号，而不是稳定的逐状态连续回归标签。

### 3. 单次 utility 的绝对可靠性偏低

`U` 的 ICC(1,1) 为 **0.278**，低于 0.35；5-seed 均值的 ICC(1,5) 提升到 **0.658**，但仍低于 0.75。ICC(1,5) 的 bootstrap 95% 区间为 **[0.327, 0.820]**，不确定性较大。

独立 post-hoc 诊断发现：`E0` 单次 ICC 约为 **0.950**，`E10` 约为 **0.790**，但它们的差 `U=E0-E10` 只有 **0.278**；`N=0/N=10` 的 seed 残差相关约 **0.20**。这与“两个本身较稳定的量相减，但随机波动没有被充分抵消，导致差分信噪比下降”的机制一致。该解释是诊断性推断，不是因果证明。

### 4. 结论对 deadband 的选择不构成实质反转

| Deadband ε | mean-U 正/负/近零状态数 | 五 seed 符合 mean 方向的平均比例 | ≥4/5 同 mean 方向（未加权） | Pilot 人群加权 ≥4/5 |
|---:|---:|---:|---:|---:|
| `1e-5` | 39 / 60 / 1 | 0.798 | 0.59 | 0.574 |
| `1e-4`（主分析） | 32 / 51 / 17 | 0.764 | 0.47 | 0.475 |
| `1e-3` | 22 / 36 / 42 | 0.782 | 0.26 | 0.188 |

更大的 deadband 会把更多状态归为近零，严格的 4/5 方向保持比例反而下降。不同 ε 下都没有证据支持“单 seed 连续 utility 标签已经足够稳定”。

## 审计范围、数据和指标定义

### 数据范围

- 模型与数据口径：Frozen UniShare checkpoint，同一 LIBERO demonstration state 配对运行 `N=0` 和 `N=10`。
- 状态数：100；每个状态 5 个 base seed（42、43、44、45、46）。
- 记录数：500；seed 42 复用 Pilot 100 条，seed 43–46 新推理 400 条。
- 数据完整性：expected/completed = 500/500，errors = 0。
- 抽样：每个 LIBERO suite 25 个状态；按 Pilot utility 分层为 SP 25、SN 25、MP 13、MN 12、NZ 25。
- 重要口径：该 100-state panel 有意过采样 utility 尾部，未加权指标描述的是诊断 panel，而不是 Pilot-500 总体。

### Utility 定义

对同一个 state、同一个 GT action chunk，在 normalized action space 中只计算非 padding action step 的 MSE：

```text
E_N = sum_{t in valid, d}(predicted_action_N[t,d] - GT_action[t,d])²
      / (number_of_valid_steps × action_dimension_D)
U   = E_0 - E_10
```

也就是在 `valid action step × action dimension` 上做 **per-element MSE**，不是先对 action dimension 求和后只按 valid step 数归一化。

- `U > 0`：`N=10` 比 `N=0` 更接近 demonstration action，video 在该离线代理指标下有帮助。
- `U < 0`：`N=0` 更接近 demonstration action，video 在该代理指标下有害或没有必要。
- 主分析 deadband 为 `ε=1e-4`；同时报告 `1e-5`、`1e-3` 敏感性。

这里的 utility 是离线 action-MSE 差，不是 rollout return、task success 或真实计算收益。

### 主要稳定性指标

- **方向一致率**：跨 seed 的 utility 符号是否保持。
- **Spearman/Kendall**：状态 utility 排序是否保持。
- **LOSO rank stability**：每次留出一个 seed，与其他四个 seed 的平均排序比较。
- **Top/Bottom 20% overlap**：极端 useful/harmful 状态集合是否复现。
- **ICC(1,1) / ICC(1,5)**：单次标签及 5-seed 均值的绝对可靠性。
- **Pilot prevalence weighting**：按 Pilot-500 的 strata 占比做 post-stratification，纠正尾部过采样。

## 预注册判定：5/15 通过，10/15 失败

| 检查项 | 观察值 | 门槛 | 结果 |
|---|---:|---:|---|
| Strong overall 新 seed 多数一致 | 0.620 | ≥0.800 | 失败 |
| SP 新 seed 多数一致 | 0.520 | ≥0.750 | 失败 |
| SN 新 seed 多数一致 | 0.720 | ≥0.750 | 失败 |
| Strong states ≥4/5 预期方向 | 0.620 | ≥0.700 | 失败 |
| seed42 vs new4-mean Spearman | 0.438 | ≥0.500 | 失败 |
| Spearman bootstrap 95% 下界 | 0.231 | ≥0.300 | 失败 |
| LOSO median Spearman | 0.388 | ≥0.400 | 失败 |
| LOSO positive-rho seed 数 | 5 | ≥4 | 通过 |
| Top-20% recall | 0.450 | ≥0.400 | 通过 |
| Top-20% Jaccard | 0.290 | ≥0.250 | 通过 |
| Bottom-20% recall | 0.550 | ≥0.400 | 通过 |
| Bottom-20% Jaccard | 0.379 | ≥0.250 | 通过 |
| ICC(1,5) | 0.658 | ≥0.750 | 失败 |
| ICC(1,1) | 0.278 | ≥0.350 | 失败 |
| Pilot 加权 strong-mean ≥4/5 同方向 | 0.505 | ≥0.600 | 失败 |

Conditional GO 要求同时满足 strong majority ≥0.65、Spearman ≥0.30、ICC(1,5) ≥0.50。后两项满足，但 strong majority = 0.62，因此最终为 `NO_GO`。

## 方法与可复现性

1. 使用预先生成且带完整 provenance 的 100-state selection plan。
2. 对每个 state 以 5 个 base seed 做 `N=0/N=10` paired inference；两条 route 使用相同 state、instruction、proprio 与该 replicate 对应的随机种子。
3. 对每个 state 计算 utility mean/median/std/SEM、t 区间、seed 符号计数与 deadband 方向。
4. 计算 seed42 对独立 new4 mean 的排序一致性、LOSO 排名、top/bottom overlap、ICC 与 variance components。
5. 用 2,000 次 bootstrap 估计 Spearman、strong agreement 和 ICC(1,5) 的 95% 区间；bootstrap seed 为 20260812。
6. 用 Pilot-500 strata prevalence（SP 0.152、SN 0.202、MP 0.262、MN 0.184、NZ 0.200）对诊断 panel 做 post-stratification。
7. 按预注册门槛生成 GO / Conditional / NO_GO，不根据结果回调门槛。

正式输出：

```text
/root/feihong/FastWAM/utility_results/
  libero_demo_utility_multiseed100x5_730f27b_seed42_46/
    manifest.json
    records.jsonl
    errors.jsonl
    analysis/analysis_summary.json
    analysis/per_state.csv
    analysis/seed_metrics.csv
    analysis/stratum_metrics.csv
```

完整性标识：

- records SHA-256：`57abaacb551b4d6094e09812212c2be8098c9d823e58fb1de71d4f40469d4fb8`
- manifest SHA-256：`c7476d522f47f71df30fb96ebaba5d09f6dd7a0a83400456a79ab1146506d0b9`
- selection plan SHA-256：`59f13375c815a556073f72cdff5243b73a19fc9dbec7bd58c4d419b4d72e90db`

## 局限性与稳健性边界

- 这是离线 demonstration action-MSE 稳定性审计，不测 Gate calibration、compute saving 或 closed-loop success。
- 100-state panel 是分层诊断样本，不是对 Pilot-500 的简单随机抽样；总体解释必须看加权指标。
- 状态是基于 seed42 Pilot utility 分层选择的，因此 strong strata 的复现评估天然面临 selection-on-noise；本审计正是用独立 seed 检验这一点。
- 5 个 seed 可以暴露明显不稳定，但对每个 state 的不确定性估计仍较宽。
- 只审计一个 checkpoint、当前 scheduler 和 `N=0/N=10` 两个 endpoint；不能外推到 checkpoint variation、中间 N 或其他推理器。
- Utility 是两个误差估计的差。即使 `E0`、`E10` 各自稳定，若随机残差不高度相关，`U` 仍可能不稳定。
- Post-hoc 的 E0/E10 ICC、残差相关与 SP/SN 细分只用于形成下一轮假设，不能替代预注册指标。

## 建议的下一步：先修标签，再重新过 Gate

### P0：把单 seed 标签改为多 seed target

- 以多 seed mean 或稳健均值作为 `U_target`，至少保留 per-state 标准差、SEM 和置信区间。
- 训练时使用 uncertainty weighting，或剔除/降权跨 seed 方向不确定的 state。
- 不把 `U≈0` 或置信区间跨 0 的样本强行标成正/负。

### P0：加强 paired randomness

- 审计并实现 `N=0/N=10` 的 shared initial noise / common random numbers。
- 尽可能共享 route 分叉前的随机路径与 solver 随机量，使 `E0-E10` 的随机误差正相关、减少差分方差。
- 用同一 100-state panel 做前后对照，重点看 `U` 的 ICC、strong agreement 和 Spearman 下界是否提升。

### P1：重新运行稳定性门槛，而不是直接训练 MLP

- 沿用本轮门槛，避免看到结果后降低标准。
- 若需要扩大样本，应优先增加独立 state 和 seed，并预先固定抽样/停止规则。
- 只有重新审计达到 GO，才开始 Tiny MLP offline train/val。

### 当前明确不做

- 不用 seed42 单次 `U` 直接训练 Tiny MLP。
- 不把 top/bottom overlap 的局部通过解释为连续标签已可靠。
- 不跳过离线标签稳定性与 Gate offline validation，直接进入 closed-loop threshold sweep。

## 仍需回答的问题

1. Shared-random-path 能把 `N=0/N=10` seed 残差相关从约 0.20 提升到什么程度？
2. 多 seed target 需要 3、5 还是更多 seed 才能使 ICC(1,k) 稳定超过 0.75？
3. 方向不确定主要来自某些 task、episode phase、valid action length，还是模型推理随机性？
4. 使用 per-dimension action loss、夹爪维度单独诊断或 rollout-aware utility 后，标签是否更接近 closed-loop 决策价值？
5. 若只训练一个“极端状态筛选器”而不是连续 utility regressor，当前 top/bottom 稳定性是否足够？这需要单独预注册目标与评估标准，不能由本轮结果直接推出。
