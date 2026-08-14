# Gate Phase 3b：Remainder-400 样本量归因复验预注册

冻结时间：2026-08-14 12:39 UTC

状态：**在 Pilot-500 remainder utility 完成、解盲和分析之前冻结。** 本文只使用已封存的
original-100 Tiny-MLP 结果、身份字段和 fold membership；写作时未读取 remainder-400
的 `E0`、`E10`、`U`、方向、高置信标签或任何分布统计。

## 1. 为什么做这次复验

original-100 exact-V1 offline Gate 已通过数据、fold、特征和统计完整性审计，但结果为
`NO_GO`：20 项 GO 仅 6 项通过，6 项 CONDITIONAL 仅 1 项通过。主要结果为：

- task-held-out Spearman `0.030855`，预注册 task permutation `p=0.434313`；
- high-confidence balanced accuracy `0.34444`；
- top/bottom 20% recall 为 `0.20/0.25`；
- 20% full-route budget 的 Validation4 mean utility 为 `-0.001039`；
- suite-held-out pooled Spearman `-0.03664`；
- 五个初始化之间的 median prediction Spearman `0.43685`。

这不是标签质量或实现错误：Target5 与独立 Validation4 的 state-level Spearman 为
`0.70717`，独立审计重算了所有正式指标、hash、join、seed grid 和 fold isolation。

本次只回答一个问题：

> 在不改变 exact-V1 特征、模型、损失、fold、阈值和测试 panel 的情况下，把训练数据从
> 原先每折约 69--70 个 state 扩展到 Pilot-500 的 remainder-400，能否使原 100-state
> locked external test 上的表现达到预注册门槛？

这是一项一次性的样本量归因复验，不是重新调参。

## 2. 不可变证据锚

original-100 正式结果目录：

```text
/root/feihong/FastWAM/utility_results/
libero_gate_offline_tiny_mlp_target100_v1_99e2107/validation4_analysis
```

独立审计：

```text
independent_audit.json file SHA256
d3e6634544973d1757ce4ced4a1d034d251af5cd2e6e284c2b77d6837bf908d6

independent_audit.md file SHA256
5adc293099abdecb26eb468020327817286c326b99f53dc2e6314da7bce43da2
```

正式 follow-up 必须绑定：Pilot-500、Phase2.5 100-state panel、existing Target100、
remainder-400 Target5、combined Target500、feature extractor、exact-V1 trainer 源码和本文的
文件 SHA。任一 source/hash/order/count 不匹配即 fail closed。

## 3. 数据隔离

### 3.1 训练标签

- 训练和 inner validation **只使用 remainder-400 的 Target5**（base seeds 42--46）。
- seed 42 可以从 Pilot-500 严格复用；43--46 必须是新 paired N=0/N=10 inference。
- original-100 的 Target5、Validation4 和 feature rows 均不得进入 fit、early stopping、
  preprocessing、threshold 或模型选择。
- seeds 47--50 始终只属于 original-100 locked external evaluation。

### 3.2 外部测试

- 测试集合固定为 original-100，同一 state 必须恰好 OOF 一次。
- 直接复用已封存的 5 个 task-held-out folds；每折测试 8 个 task、20 个 state。
- 每个外折从 remainder-400 中删除这 8 个 held-out task 的全部 state 和 episode。
- original-100 已经被一次正式分析解盲，因此本次只能称为 **locked external re-test**，
  不能称为全新 confirmatory holdout。

### 3.3 Group 定义

- task group：`(suite, task_index, task)`；禁止只用跨 suite 冲突的裸 `task_index`。
- episode group：`(dataset_name, episode_index)`；同一 episode 永不跨 train/inner-val。
- global source key 使用 `sample_id` 或 top-level global `source_index`，禁止使用可能跨 suite
  冲突的 dataset-local `source_metadata.source_index`。

## 4. Fold 与 learning curve

### 4.1 Inner validation

沿用 exact-V1 namespace `libero_gate_mlp_inner_v1`。每个外折删掉 held-out tasks 后，在每个
suite 的剩余 task groups 中按既有 identity-only SHA 顺序取首个 task 作为 inner validation，
共 4 个 task；其全部 remainder states 均为固定 inner-val。其余 task 才能进入 inner-train。

### 4.2 Nested learning curve

只对 inner-train 做 `q={25%,50%,75%,100%}` 四档嵌套抽样；inner-val 和 original-100 test
始终不变。namespace 固定为 `libero_gate_remainder400_curve_v1`。

在每个 `(suite, task_index, task)` 内，把完整 episode groups 按

```text
sha256(namespace || dataset_name || episode_index)
```

排序；档位 `q` 取前 `max(1, ceil(q * G_task))` 个 episode groups。这样 q25 是 q50 的子集，
q50 是 q75 的子集，q75 是 q100 的子集，并保留每个可用 train task。若任一应有 task 没有
eligible remainder state，直接失败。报告每折、每档实际 state/task/episode 数，不把名义 q
伪装成恰好比例。

所有 fold membership、ordered episode lists、档位 membership 和 SHA 必须在任何 fit 前写入
不可变 plan 并 fsync；最终 completion seal 再次绑定并复核。

## 5. Primary：exact-V1 137 维

Primary 必须逐字复用 original-100 正式协议：

- visual 64：当前两相机 first-frame VAE latent 的 channel/grid 统计，经冻结、fit-free 的
  Rademacher 432→64 投影；
- instruction 65：当前 cached context 的 masked mean/RMS 冻结投影各 32，加 active fraction；
- normalized current proprio 8；
- 总维度 137；outer-train z-score，常数列置 0；
- MLP `137→32→16→1`、SiLU、dropout、五个固定 init seeds、Target5-only early stopping、
  uncertainty-weighted Huber + pairwise ranking loss等全部保持 exact-V1；
- 禁止 future frame、GT action、padding/valid length、E0/E10/U、seed统计、CI/HC/direction、
  selection bin、suite/task/source/frame/episode ID 进入 feature。

Primary 科学判定只看 `137/q100`。q25/q50/q75 只用于样本量曲线和预注册 attribution delta。

## 6. Exploratory：未压缩视觉 505 维

在看过 original-100 `NO_GO` 后才提出，因此只作 exploratory diagnosis，不得改变 Primary
判定。它不引入新 observation，只把同一 VAE current-frame statistics 保留为 visual 432，
再拼 instruction 65 和 proprio 8，总计 505；模型、优化器、fold 和 Target5-only boundary
不变。

Exploratory 预测也必须在读取 Validation4 前与 Primary 一起 sealed。无论表现多好，都只能
生成“视觉压缩可能是限制”的新假设，必须在新的预注册/新 holdout 上确认。

## 7. 结果隔离与封存

训练入口不得接受 Validation4 路径、hash 或 loader。它只读取 remainder Target5、frozen
features 和 frozen plan，并先输出不可变：

```text
fold_plan.json
oof_predictions.jsonl
run_manifest.json
completion.json
```

四者通过 file SHA、compatibility fingerprint、ordered row hashes 和 completion seal 互绑后，
独立 analyzer 才能加载 original-100 Validation4。改变 Validation4 的任何 byte 都不得改变
trainer 输出的任何 byte。

## 8. 判定规则

### A. Sample-size supported

必须同时满足：

1. `137/q100` 通过 original Phase-3 **全部 20/20 GO** 门槛；
2. `Δrho_sample = rho(137/q100) - rho(137/q25) >= 0.05`；
3. 上述 paired task-cluster bootstrap 差值的 95% CI lower bound `> 0`。

只能结论为“样本量是可证实的贡献因素”。由于 original-100 已揭盲，下一步仍必须是全新
confirmatory holdout；不授权 closed-loop。

### B. V1 learnable, sample-size attribution not established

若 `137/q100` 通过 20/20 GO，但任一 sample-size delta 条件失败，则 exact-V1 在 locked test
上可学习，但不能归因于样本量曲线；仍需全新 holdout 确认。

### C. Conditional only

若 `137/q100` 未达 GO、但满足原协议全部 CONDITIONAL 门槛，只允许 high-confidence 小规模
offline diagnostic；不允许全面训练、阈值 sweep 或 closed-loop。

### D. Exact-V1 NO-GO

若连原 CONDITIONAL 也未满足，停止继续“救” exact-V1。下一步只能重新预注册更强视觉
表征、prepared-conditioning API 或新的 utility target；不能在同一 original-100 test 上反复
调参与重试。

### Exploratory representation attribution

505 维仅在相对 `137/q100` 同时满足

```text
Δrho_feature >= 0.05
paired task-bootstrap 95% lower bound > 0
```

时记录“64维视觉压缩可能是限制”。它永不升级 A--D 的 Primary 结论。

## 9. 原 Phase-3 GO/CONDITIONAL 口径保持不变

所有 primary 指标都以 original-100 Validation4 mean utility 计算；Target5 只用于训练、
inner decisions 和预定义 HC cohort eligibility。正式门槛继续包括：task-held-out rank 与 exact
task permutation、HC balanced accuracy/两类 recall、top/bottom 20%、相对最佳 nonvisual
baseline 的 paired delta、20% budget utility 与 compute-matched random、suite-held-out guardrail、
within-task shortcut 和五 init agreement。禁止因本次结果修改阈值。

## 10. 明确不授权的事项

无论本次结果如何，均不直接授权：

- 修改 UniShare training loss 或 inference semantics；
- 把 Gate 接入正式 LIBERO closed-loop；
- threshold/Pareto sweep；
- 用 original-100 Validation4 继续调模型；
- 把 exploratory 505 结果当 confirmatory evidence。

只有新的 confirmatory holdout 再次通过预注册标准后，才可以单独设计小规模 closed-loop
learned-vs-random router smoke。
