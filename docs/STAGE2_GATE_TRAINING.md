# Stage 2 Binary Video Gate：正式标签与训练链

Stage 2 在已经完成的 Stage 3 之上学习动态路由：对每个 LIBERO 当前观测，判断
是否值得走带未来想象的 `w / N=10` 分支。它不是继续微调 5B 模型。

| 阶段 | 加载什么 | 实际训练什么 |
| --- | --- | --- |
| Stage 3 Action Alignment | UnifiedShared 5B base + Adapter | 只训练小型 Action Alignment Adapter；5B base 冻结，但参与前向计算 |
| Stage 2 标签生成 | 冻结的 5B base + 已训练 Adapter | 不训练参数，只用 `torch.inference_mode()` 生成同噪声的 `N=0/N=10` 配对结果 |
| Stage 2 Gate 训练 | 当前图像、当前 proprio、文本 context 和离线标签 | 只实例化并训练小型 `BinaryVideoGate`，不加载 5B base、Adapter 或 VAE |

因此，“Stage 2 只训练小网络”对 **Gate 训练**是准确的；但前面的离线标签生成仍需
把冻结的 5B+Adapter 放到 GPU 上做推理。Stage 3 同样只更新小 Adapter，5B base
虽然冻结，仍要加载并参与前向。

## 正式范围与前置条件

当前 formal v1 只支持 LIBERO。不要把 `data=robotwin` 当作已完成的正式 Stage 2
链路。开始前需要准备：

- 完成并冻结的 Stage 3 Adapter-only export；它必须绑定本次使用的 base、VAE 和
  data manifest；
- Stage 3 使用的同一份 UnifiedShared base checkpoint、Wan2.2 VAE、normalization
  stats 和 selected-data manifest，以及每个文件的 SHA256；
- manifest 所绑定的 LIBERO dataset roots 和 text embedding cache；
- 可复现的干净 Git 状态。正式入口默认拒绝 tracked dirty，以及
  `src/`、`configs/`、`scripts/`、`tests/` 下的未跟踪文件；
- 标签生成所需的 CUDA 显存。Gate 本身很小，但正式配置同样默认
  `runtime.require_cuda=true`。

先按 [Stage 3 Action Alignment](./STAGE3_ALIGNMENT_TRAINING.md) 生成并验证 data
manifest、训练 Adapter。下面所有 `/absolute/path/...` 和 `*_SHA256` 都是占位符，
必须替换成用户实际资产；本仓库不会替用户猜测资产路径或 hash。

为使三条命令易读，先定义一次路径：

```bash
export FASTWAM_STAGE2_ROOT=/absolute/path/stage2_run
export FASTWAM_LABEL_JOB=${FASTWAM_STAGE2_ROOT}/label_job
export FASTWAM_MERGED_LABELS=${FASTWAM_STAGE2_ROOT}/merged_labels
export FASTWAM_GATE_RUN=${FASTWAM_STAGE2_ROOT}/gate_run

export FASTWAM_BASE_CHECKPOINT=/absolute/path/unified_shared_base.pt
export FASTWAM_BASE_SHA256=BASE_CHECKPOINT_SHA256
export FASTWAM_STAGE3_ADAPTER=/absolute/path/stage3_adapter_export.pt
export FASTWAM_STAGE3_ADAPTER_SHA256=STAGE3_ADAPTER_SHA256
export FASTWAM_WAN22_DIR=/absolute/path/Wan2.2-TI2V-5B
export FASTWAM_TOKENIZER_DIR=/absolute/path/Wan2.1-T2V-1.3B
export FASTWAM_VAE=${FASTWAM_WAN22_DIR}/Wan2.2_VAE.pth
export FASTWAM_VAE_SHA256=VAE_SHA256
export FASTWAM_NORMALIZATION_STATS=/absolute/path/libero_norm_stats.json
export FASTWAM_NORMALIZATION_STATS_SHA256=NORMALIZATION_STATS_SHA256
export FASTWAM_DATA_MANIFEST=/absolute/path/libero_stage3_data_manifest.json
export FASTWAM_DATA_MANIFEST_SHA256=DATA_MANIFEST_SHA256

export FASTWAM_LIBERO_SPATIAL=/absolute/path/libero_spatial_no_noops_lerobot
export FASTWAM_LIBERO_OBJECT=/absolute/path/libero_object_no_noops_lerobot
export FASTWAM_LIBERO_GOAL=/absolute/path/libero_goal_no_noops_lerobot
export FASTWAM_LIBERO_10=/absolute/path/libero_10_no_noops_lerobot
export FASTWAM_LIBERO_TEXT_CACHE=/absolute/path/libero_text_embeds_cache
```

`FASTWAM_WAN22_DIR` 必须解析到实际加载 `FASTWAM_VAE` 的同一模型目录；仅在
`assets.vae.path` 写一个不同位置的同 hash 文件仍会被拒绝。四个 dataset root、text
cache 和 stats 也必须与 data manifest 完全一致。

## 第一步：生成 E0/E10 标签分片

单进程入口：

```bash
python scripts/generate_gate_labels.py \
  data=libero_2cam \
  model=fastwam_unified_aligned \
  output_dir="${FASTWAM_LABEL_JOB}" \
  base.checkpoint="${FASTWAM_BASE_CHECKPOINT}" \
  base.expected_sha256="${FASTWAM_BASE_SHA256}" \
  adapter.checkpoint="${FASTWAM_STAGE3_ADAPTER}" \
  adapter.expected_sha256="${FASTWAM_STAGE3_ADAPTER_SHA256}" \
  assets.vae.path="${FASTWAM_VAE}" \
  assets.vae.expected_sha256="${FASTWAM_VAE_SHA256}" \
  assets.normalization_stats.path="${FASTWAM_NORMALIZATION_STATS}" \
  assets.normalization_stats.expected_sha256="${FASTWAM_NORMALIZATION_STATS_SHA256}" \
  data_manifest.path="${FASTWAM_DATA_MANIFEST}" \
  data_manifest.expected_sha256="${FASTWAM_DATA_MANIFEST_SHA256}" \
  data.train.dataset_dirs="[${FASTWAM_LIBERO_SPATIAL},${FASTWAM_LIBERO_OBJECT},${FASTWAM_LIBERO_GOAL},${FASTWAM_LIBERO_10}]" \
  data.train.text_embedding_cache_dir="${FASTWAM_LIBERO_TEXT_CACHE}" \
  data.train.pretrained_norm_stats="${FASTWAM_NORMALIZATION_STATS}" \
  model.model_id="${FASTWAM_WAN22_DIR}" \
  model.tokenizer_model_id="${FASTWAM_TOKENIZER_DIR}"
```

每个样本用相同的一组随机 seed 分别执行 `wo / N=0` 和 `w / N=10` 推理，再在
normalized action space 中与完整 action target 计算 `E0`、`E10`。默认 margin 下，
只有 `E10 < 0.95 * E0` 才标为应走 `N=10`；未来真值图像不会作为模型输入。
base 和 Adapter 全程 eval/frozen，标签生成没有 optimizer，也不会写入模型权重。

多 GPU 时用 `torchrun`。当 `labeling.shard_indices=null` 时，`RANK/WORLD_SIZE`
自动把全部 shard 交错分给各 rank；`LOCAL_RANK` 自动选择 GPU：

```bash
torchrun --standalone --nproc_per_node=NUM_GPUS \
  scripts/generate_gate_labels.py \
  data=libero_2cam \
  model=fastwam_unified_aligned \
  output_dir="${FASTWAM_LABEL_JOB}" \
  base.checkpoint="${FASTWAM_BASE_CHECKPOINT}" \
  base.expected_sha256="${FASTWAM_BASE_SHA256}" \
  adapter.checkpoint="${FASTWAM_STAGE3_ADAPTER}" \
  adapter.expected_sha256="${FASTWAM_STAGE3_ADAPTER_SHA256}" \
  assets.vae.path="${FASTWAM_VAE}" \
  assets.vae.expected_sha256="${FASTWAM_VAE_SHA256}" \
  assets.normalization_stats.path="${FASTWAM_NORMALIZATION_STATS}" \
  assets.normalization_stats.expected_sha256="${FASTWAM_NORMALIZATION_STATS_SHA256}" \
  data_manifest.path="${FASTWAM_DATA_MANIFEST}" \
  data_manifest.expected_sha256="${FASTWAM_DATA_MANIFEST_SHA256}" \
  data.train.dataset_dirs="[${FASTWAM_LIBERO_SPATIAL},${FASTWAM_LIBERO_OBJECT},${FASTWAM_LIBERO_GOAL},${FASTWAM_LIBERO_10}]" \
  data.train.text_embedding_cache_dir="${FASTWAM_LIBERO_TEXT_CACHE}" \
  data.train.pretrained_norm_stats="${FASTWAM_NORMALIZATION_STATS}" \
  model.model_id="${FASTWAM_WAN22_DIR}" \
  model.tokenizer_model_id="${FASTWAM_TOKENIZER_DIR}" \
  labeling.num_shards=64
```

`NUM_GPUS` 不能大于 `labeling.num_shards`。也可让独立作业显式传递已排序且互不重叠的
`labeling.shard_indices='[0,1,...]'`；显式 subset 只能用于 `WORLD_SIZE=1` 的独立
单进程作业，不能和 `torchrun` 叠加；所有作业的其余配置必须完全相同。

该步发布：

```text
label_job/
├── episode_split.json
├── label_runtime_config.json
├── label_contract.json
└── shard-XXXXX/chunk-XXXXXXXX.json
```

从 `episode_split.json` 记录
`assignment_sha256`，从 `label_contract.json` 记录 `contract_sha256`，供后两步显式
锁定。重跑同一命令就是标签任务的恢复方式：已存在的完整 chunk 会先严格验证再跳过；
不同内容不会覆盖原文件。

## 第二步：严格合并标签

```bash
export FASTWAM_EPISODE_ASSIGNMENT_SHA256=EPISODE_ASSIGNMENT_SHA256
export FASTWAM_LABEL_CONTRACT_SHA256=LABEL_CONTRACT_SHA256

python scripts/merge_gate_labels.py \
  label_job_dir="${FASTWAM_LABEL_JOB}" \
  data_manifest.path="${FASTWAM_DATA_MANIFEST}" \
  data_manifest.expected_sha256="${FASTWAM_DATA_MANIFEST_SHA256}" \
  episode_split.path="${FASTWAM_LABEL_JOB}/episode_split.json" \
  episode_split.expected_assignment_sha256="${FASTWAM_EPISODE_ASSIGNMENT_SHA256}" \
  label_contract.path="${FASTWAM_LABEL_JOB}/label_contract.json" \
  label_contract.expected_sha256="${FASTWAM_LABEL_CONTRACT_SHA256}" \
  output.directory="${FASTWAM_MERGED_LABELS}"
```

merge 不用 glob 猜测输入，而是从 immutable contract 重算完整 shard/chunk plan；缺失、
额外、重复、坐标错误或 coverage 不完整都会失败。它先构建并复核 `labels.jsonl`，最后
发布 `manifest.json` 作为 commit marker。若进程在两者之间中断，重跑会要求孤立 rows
与重算结果逐字节相同，再补发 manifest；已经完整发布的结果只会验证并复用，不会覆盖。

从 `${FASTWAM_MERGED_LABELS}/manifest.json` 记录 `manifest_sha256`。

## 第三步：只训练 BinaryVideoGate

```bash
export FASTWAM_LABEL_MANIFEST_SHA256=LABEL_MANIFEST_SHA256

python scripts/train_video_gate.py \
  data=libero_2cam \
  output_dir="${FASTWAM_GATE_RUN}" \
  data_manifest.path="${FASTWAM_DATA_MANIFEST}" \
  data_manifest.expected_sha256="${FASTWAM_DATA_MANIFEST_SHA256}" \
  episode_split.path="${FASTWAM_LABEL_JOB}/episode_split.json" \
  episode_split.expected_assignment_sha256="${FASTWAM_EPISODE_ASSIGNMENT_SHA256}" \
  label_contract.path="${FASTWAM_LABEL_JOB}/label_contract.json" \
  label_contract.expected_sha256="${FASTWAM_LABEL_CONTRACT_SHA256}" \
  label_manifest.path="${FASTWAM_MERGED_LABELS}/manifest.json" \
  label_manifest.expected_sha256="${FASTWAM_LABEL_MANIFEST_SHA256}" \
  source_identities.base_checkpoint_sha256="${FASTWAM_BASE_SHA256}" \
  source_identities.adapter_checkpoint_sha256="${FASTWAM_STAGE3_ADAPTER_SHA256}" \
  assets.normalization_stats.path="${FASTWAM_NORMALIZATION_STATS}" \
  assets.normalization_stats.expected_sha256="${FASTWAM_NORMALIZATION_STATS_SHA256}" \
  data.train.dataset_dirs="[${FASTWAM_LIBERO_SPATIAL},${FASTWAM_LIBERO_OBJECT},${FASTWAM_LIBERO_GOAL},${FASTWAM_LIBERO_10}]" \
  data.train.text_embedding_cache_dir="${FASTWAM_LIBERO_TEXT_CACHE}" \
  data.train.pretrained_norm_stats="${FASTWAM_NORMALIZATION_STATS}"
```

这一入口不会打开 `base.checkpoint` 或 `adapter.checkpoint`；只要求它们的 SHA 与 label
contract 一致。底层 current-only reader 的每样本查询只请求当前图像和当前 proprio，并返回
`input_image/context/context_mask/proprio/sample_identity`。它不在 Gate 样本查询计划中
注册或返回 action，也不会为 Gate 构造未来视频序列；`Stage2GateDataset` 再加入 label、
sample weight 和 sample ID。optimizer 中唯一的参数来自 `BinaryVideoGate`。
Gate 训练是单进程入口，不能用 `torchrun`；输出目录的 writer lock 也会拒绝两个普通
进程同时写入同一组 state/export/summary。generate/train 的正式 data 配置都设置
`save_stats_copy=false`，不会覆盖全局 work directory 中的 `dataset_stats.json`。

默认输出：

```text
gate_run/
├── run_identity.json
├── training_state.pt   # optimizer/progress/RNG 和严格恢复状态
├── gate_best.pt        # Gate-only export
├── gate_last.pt        # Gate-only export
└── summary.json
```

### Gate 恢复

同目录恢复必须继续使用 canonical state，并保持身份绑定的配置不变：

```bash
python scripts/train_video_gate.py \
  data=libero_2cam \
  output_dir="${FASTWAM_GATE_RUN}" \
  checkpoint.resume="${FASTWAM_GATE_RUN}/training_state.pt" \
  data_manifest.path="${FASTWAM_DATA_MANIFEST}" \
  data_manifest.expected_sha256="${FASTWAM_DATA_MANIFEST_SHA256}" \
  episode_split.path="${FASTWAM_LABEL_JOB}/episode_split.json" \
  episode_split.expected_assignment_sha256="${FASTWAM_EPISODE_ASSIGNMENT_SHA256}" \
  label_contract.path="${FASTWAM_LABEL_JOB}/label_contract.json" \
  label_contract.expected_sha256="${FASTWAM_LABEL_CONTRACT_SHA256}" \
  label_manifest.path="${FASTWAM_MERGED_LABELS}/manifest.json" \
  label_manifest.expected_sha256="${FASTWAM_LABEL_MANIFEST_SHA256}" \
  source_identities.base_checkpoint_sha256="${FASTWAM_BASE_SHA256}" \
  source_identities.adapter_checkpoint_sha256="${FASTWAM_STAGE3_ADAPTER_SHA256}" \
  assets.normalization_stats.path="${FASTWAM_NORMALIZATION_STATS}" \
  assets.normalization_stats.expected_sha256="${FASTWAM_NORMALIZATION_STATS_SHA256}" \
  data.train.dataset_dirs="[${FASTWAM_LIBERO_SPATIAL},${FASTWAM_LIBERO_OBJECT},${FASTWAM_LIBERO_GOAL},${FASTWAM_LIBERO_10}]" \
  data.train.text_embedding_cache_dir="${FASTWAM_LIBERO_TEXT_CACHE}" \
  data.train.pretrained_norm_stats="${FASTWAM_NORMALIZATION_STATS}"
```

`training.num_epochs` 表示该 run 的总目标 epoch 数，不是本次额外增加的 epoch；严格恢复
要求沿用原值。fresh run 若发现任一输出已存在会拒绝覆盖；从其他目录导入 state 时，
新的 `output_dir` 必须为空。恢复会验证 Gate 结构、optimizer/RNG、label/base/Adapter、
data/split、训练配置和 Git identity；summary 若因中断领先 state 一个 epoch，会裁掉该条
记录并重跑该 epoch。

## 身份、源数据漂移与不可覆盖语义

标签合同绑定 data manifest、episode split、base、Adapter、VAE、stats、解析后的 data
配置、model/dtype、数值库与 CUDA/GPU identity、Git identity、配对 seed、推理参数、
shard 数、chunk size 和规划算法。不同模型配置或数值环境产生的标签不能混入同一任务。
Gate export 继续绑定 merged-label manifest、base/Adapter hash、data manifest、split、
训练配置和 Git identity。

Stage 2 还会对 manifest 选中的 parquet、MP4、text cache 和 stats 建立源文件快照。在
长任务的 chunk/epoch 边界做低成本文件身份检查，并在合并或最终发布前做完整内容复核；
路径、inode、size、mtime/ctime 或 SHA256 发生漂移都会 fail closed。manifest 只证明
生成时的内容，这些运行期检查用来防止文件在长任务中被原地替换。

所有 durable 标签 artifact 都采用 no-clobber 发布：目标不存在时才原子创建；目标已存在
时只能接受严格一致的完整内容。Gate fresh run 同样拒绝覆盖，只有通过身份校验的显式
resume 才能继续。

## 已验证边界

本链路的 contract、标签数学、分片/合并、current-only Gate dataset、Gate optimizer 与
严格恢复由 CPU 单元和入口测试覆盖。本文没有声称已经完成真实 5B H100 端到端标签生成
或 Gate GPU 验收；使用用户自己的绝对资产路径和 hash 跑正式任务前，仍需完成该环境的
CUDA smoke、短标签分片、merge 和 Gate save→resume 验收。
