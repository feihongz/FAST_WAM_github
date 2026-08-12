# Gate Phase 2：LIBERO Demo Video-Utility Collector

## 1. 这一步解决什么问题

Phase 2 暂时不训练 MLP Gate。它只冻结正式 UniShare checkpoint，在完全相同的
LIBERO demonstration state 上成对比较两个 prefix endpoint：

```text
当前 observation / instruction / proprio / GT action chunk
                         │
              相同的确定性 inference seed
                  ┌──────┴──────┐
                  │             │
                N=0           N=full
                  │             │
                 â0            âfull
                  └──────┬──────┘
                         │
              E0, Efull, U=E0-Efull
```

目标是先验证 demonstration 中是否存在稳定、具有 state dependency 的 video
utility 信号，再决定是否值得训练 Gate。

本 PR 不包含 MLP、Gate optimizer、joint training，也不修改
`training_loss()` 或现有 LIBERO evaluator 行为。

## 2. Phase 0 前置证据

Collector 开发建立在已经通过的 endpoint 验收上。提交 `ac4c5ed` 使用同一
checkpoint、seed 和 LIBERO-10 的 10 tasks × 50 episodes 正式复现：

| Router | 成功数 | route | 历史 endpoint |
| --- | ---: | --- | --- |
| `fixed_0` | 477 / 500 | 13,438 / 13,438 replans 为 `N=0` | 与 canonical `wo` / custom `N=0` 逐 episode 一致 |
| `fixed_full` | 490 / 500 | 12,744 / 12,744 replans 为 `N=10` | 与 canonical `w` / custom `N=10` 逐 episode 一致 |

正式验收报告位于运行机：

```text
/root/feihong/FastWAM/evaluate_results/gate_phase0_ac4c5ed/
formal_libero10_50ep_2workers/phase0_formal_acceptance.json
```

报告中的 `canonical_custom_reference_match` 和
`router_endpoint_episode_outcomes_match` 均为 `true`。因此 Phase 2 比较的是已经
经过 closed-loop endpoint 等价验证的两个 route。

## 3. 无未来泄漏约束

每条 dataset sample 虽然包含 demonstration video 序列，但 Collector 的核心函数
只允许模型看到：

- `video[:, 0]`：当前两相机拼接 observation；
- `proprio[0]`：当前 normalized proprio；
- 已缓存的当前 instruction `context/context_mask`；
- GT normalized action chunk 只用于 inference 完成后的误差计算。

`video[:, 1:]` 不会传给模型，也不会作为任何 endpoint 的条件。测试使用 fake
model 检查两个调用收到同一个 current observation/context/proprio/seed，唯一的
route 差异是 `video_prefix_steps=0` 或 `video_prefix_steps=full`。

## 4. Utility 定义

设有效 action timestep 集合为
`V = {t | action_is_pad[t] == false}`，action dimension 为 `D`。第一版在 dataset
已经使用的 normalized action space 中计算普通 element-wise chunk MSE：

```text
E_N = (1 / (|V| D)) Σ_{t∈V} ||â_{N,t} - a*_t||²₂
U   = E_0 - E_full
```

- `U > 0`：full endpoint 更接近 demonstration GT，video 对该状态有帮助；
- `U < 0`：`N=0` 更接近 GT，video harmful 或不必要；
- `U ≈ 0`：两个 endpoint 的 action MSE 几乎无差别。

Padding 不参与分子或分母；没有任何有效 action 的样本会显式报错。第一版不加
gripper 权重、step 权重或 task-specific loss。

## 5. 数据和模型配置

入口使用独立配置
[`configs/collect_libero_demo_utility.yaml`](../configs/collect_libero_demo_utility.yaml)：

- task 为 `libero_unified_shared_2cam224_1e-4`；
- `model.load_text_encoder=false`，直接消费预计算 instruction embedding；
- `skip_dit_load_from_pretrain=true`、`action_dit_pretrained_path=null`，随后只加载指定
  的正式 checkpoint；
- dataset 强制 `is_training_set=false`、`pretrained_norm_stats=<resolved path>`、
  `strict_getitem=true`、`return_metadata=true`、
  `skip_padding_as_possible=false`；
- model `eval()`、所有 parameter `requires_grad=false`，成对调用运行在
  `torch.inference_mode()` 中。

正式 UniShare checkpoint 必须完整覆盖当前 model 的全部 `mot` 参数，并在启用
proprio 时完整包含 `proprio_encoder`。Collector 拒绝 legacy `dit`、缺键、多键或
shape 不一致的部分 checkpoint，避免随机初始化残留参数静默参与标签生成。

`strict_getitem` 很重要：读取失败时不允许用随机样本替换请求的 source index，否则
`sample_id`、episode/frame metadata 和 utility label 会错位。

Collector 严格拒绝
`full_prefix_steps != num_inference_steps`。Phase 2 只构造 `N=0` 与正式 full endpoint
的标签，不在这里研究中间 N。

## 6. 运行采集

建议第一轮采 500～2000 个 state：

```bash
cd /root/feihong/FAST_WAM_github

PYTHONPATH=src:. python experiments/libero/gate/collect_demo_utility.py \
  ckpt=/absolute/path/to/checkpoints/weights/latest.pt \
  COLLECTOR.dataset_stats_path=/absolute/path/to/dataset_stats.json \
  COLLECTOR.output_dir=/absolute/path/to/demo_utility_run \
  COLLECTOR.num_samples=1000 \
  COLLECTOR.seed=42
```

若不显式指定 stats，程序会从 checkpoint 的若干父目录寻找
`dataset_stats.json`；找不到时 fail fast，不会重新估计 normalization stats。

默认 `num_inference_steps=10`，因此两次调用为 `N=0` 与 `N=10`。两者使用同一个
由 source identity 和 base seed 稳定派生的 inference seed/action initial noise。

## 7. 抽样为什么不是“前 1000 条”

Collector 在解码图像前调用 dataset 的公共 `dataset_index_ranges()`，按
`dataset_index`（LIBERO spatial/object/goal/10）分层：

1. 按 population 比例分配 quota；样本数允许时保证每个 suite 至少一条；
2. 每层用 `base seed + dataset_index` 派生的独立 seed 打乱完整 source range；
3. 每层无放回取 quota；
4. 再确定性打乱合并后的 index，使不同 suite 在执行顺序中交错。

`manifest.json` 为每层保存 population、allocated、seed 和该层有序 source index
的 SHA-256，并保存全局有序 index 列表及 SHA-256。相同 manifest 配置会重建完全
相同的计划，不受 worker 调度或 resume 影响。

## 8. 输出和恢复

一个 run 目录的核心文件：

```text
demo_utility_run/
├── manifest.json       # 一次创建，之后只校验、不覆盖
├── records.jsonl       # 每个成功 state 一行，append + flush + fsync
├── errors.jsonl        # 每个失败 state 的 source index/异常/traceback
└── dataset_stats.json  # dataset 初始化时保存的 stats 副本
```

Manifest 固化：

- checkpoint/stats 的 resolved absolute path、SHA-256、size 和 mtime；
- 实际 loader 使用的 VAE artifact 完整 SHA-256，并与
  `model.model_paths["vae"]` 交叉验证；
- 四个 LIBERO source dataset 和 text embedding cache 的目录内容 SHA-256；
- 完整 resolved Hydra config 及 SHA-256；
- git commit、branch、dirty flag、porcelain 状态和 tracked diff SHA-256；
- 关键 label-generation 源文件的内容 SHA-256（包括尚未提交的新文件）；
- dataset suite ranges、每层 allocation/seed、完整采样计划及 SHA-256；
- Python/PyTorch/CUDA/runtime provenance；
- 影响 paired inference 的所有参数。

恢复时必须显式复用原 output 目录和原配置：

```bash
PYTHONPATH=src:. python experiments/libero/gate/collect_demo_utility.py \
  ckpt=/same/checkpoint/latest.pt \
  COLLECTOR.dataset_stats_path=/same/dataset_stats.json \
  COLLECTOR.output_dir=/absolute/path/to/demo_utility_run \
  COLLECTOR.num_samples=1000 \
  COLLECTOR.seed=42 \
  COLLECTOR.resume=true
```

程序先校验 manifest compatibility fingerprint，再读取现有 `sample_id` 和 requested
source index 去重。任何 manifest 不兼容、损坏 JSONL、重复 identity 或 plan 外记录
都会拒绝恢复。失败样本写入 `errors.jsonl`，修复问题后仍用同一命令重试；已成功的
state 不会重新 inference。

恢复时每一条已有 record 还会重新校验完整 schema、`E0/Efull/U`、shape、route、
稳定 seed、source identity、checkpoint/stats/VAE hash 和实际输入 hash；残缺或被
篡改的一行不会被误判为“已完成”。

## 9. Record schema

`records.jsonl` 每行至少包含：

```text
schema_version, sample_id
dataset_id, dataset_name, suite
episode_index/episode_id, frame_index, task_index/task_id, task_id_source, task
seed, num_inference_steps, n0, nfull
e0, efull, utility, valid_length
target_action_shape, pred_n0_shape, pred_nfull_shape
n0_latency_ms, nfull_latency_ms, total_latency_ms
n0_route, nfull_route

source_metadata:
  requested_sample_idx, source_sample_idx, dataset_index,
  dataset_id/name, episode_index, frame_index, task_index,
  task, timestamp, source_index

current_proprio
input_hashes:
  input_image, proprio, context, context_mask,
  valid_target_action, action_is_pad, combined

checkpoint_sha256, dataset_stats_sha256, vae_sha256, git_sha
manifest_compatibility_fingerprint
```

这里保存 normalized current proprio 和可追溯 source metadata，但不复制体积很大的
`context`/text feature，也不保存 demonstration future video。原始 source identity、
stats、checkpoint 和 config 足以按需重新提取后续 Gate feature。

`input_hashes` 只保存内容摘要，不复制大 tensor。其中 image hash 严格只覆盖当前
`video[:, 0]`，不会把 future demonstration frame 纳入 record。

## 10. Utility 分布分析

采集完整后运行：

```bash
PYTHONPATH=src:. python experiments/libero/gate/analyze_demo_utility.py \
  --records /absolute/path/to/demo_utility_run/records.jsonl \
  --output-dir /absolute/path/to/demo_utility_run/analysis \
  --near-zero-epsilon 1e-4 \
  --bins 60
```

当同目录存在 `manifest.json` 时，分析器默认要求 records 100% 覆盖 immutable
sampling plan；不完整采集会直接报错。如果仅为中途诊断而有意分析 plan 内子集，需
显式加 `--allow-incomplete`。即使开启该选项，plan 外记录、重复 source index 和
fingerprint 不一致仍会被拒绝。

输出：

```text
analysis/
├── summary.json
├── overall.csv
├── by_suite.csv
├── by_task.csv
├── histogram.csv
└── utility_histogram.png   # 环境安装 matplotlib 时生成；否则其他输出照常完成
```

overall/suite/task 统计都包含 count、positive/negative/near-zero 数量与比例、
mean、median、mean absolute utility、min/max 和 1/5/25/50/75/95/99% quantile。
`summary.json` 还记录 records/manifest SHA-256。`matplotlib` 不是项目硬依赖；缺失时
只跳过 PNG，不跳过 JSON/CSV。它同时记录 completeness、覆盖率和缺失 source index
示例，避免把 partial 数据误当成完整 scientific readout。

分析的 stop point 是检查 `U` 是否有足够量级、正负结构和 task/state dependency。
若绝大多数 `|U|` 都挤在 numerical-noise 区域，应先改进 supervision，而不是直接
开始训练 MLP。
