# Stage 2 Binary Video Gate：正式标签与训练链

Stage 2 在各 benchmark 已完成的 Stage 3 之上学习动态路由：对每个当前观测，判断
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

正式代码支持两种数据身份：LIBERO 使用 schema-v1 inline manifest；RoboTwin 2.0
使用 schema-v2 manifest，并通过 descriptor + fixed-record binary index 绑定大规模文本
cache。v2 reader 会先按 index 校验 payload 字节的 SHA256，再从同一份已校验字节反序列化；
不会在 cache miss 时退回未经验证的 `torch.load(path)`。

这表示 RoboTwin 的 Stage 2 软件链已经有正式入口，不表示它现在可以越过 Stage 3
直接训练。每个 benchmark 开始 Stage 2 前都需要准备：

- 完成并冻结的 Stage 3 Adapter-only export；它必须绑定本次使用的 base、VAE 和
  data manifest；
- Stage 3 使用的同一份 UnifiedShared base checkpoint、Wan2.2 VAE、normalization
  stats 和 selected-data manifest，以及每个文件的 SHA256；
- manifest 所绑定的数据 roots、text embedding cache；RoboTwin 还需要 manifest 所绑定的
  descriptor/index；
- 可复现的干净 Git 状态。正式入口默认拒绝 tracked dirty，以及
  `src/`、`configs/`、`scripts/`、`tests/` 下的未跟踪文件；
- 标签生成所需的 CUDA 显存。Gate 本身很小，但正式配置同样默认
  `runtime.require_cuda=true`。

先按 [Stage 3 Action Alignment](./STAGE3_ALIGNMENT_TRAINING.md) 生成并验证 data
manifest、训练 Adapter。LIBERO 与 RoboTwin 必须分别拥有自己的 base、manifest、
Adapter export、label job、merged labels 和 Gate；不能混用或把两个 benchmark 合并成
一个 Adapter/Gate。下面的长命令是 LIBERO 通用示例，其中所有 `/absolute/path/...` 和
`*_SHA256` 都必须替换成实际资产。

为使 LIBERO 的三条命令易读，先定义一次路径：

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

LIBERO 正式运行优先使用已锁定的专用 task。它内置 schema-v1 manifest
`08da49109a57b55c67f3fa4ac31fbfa44e44dd541a194a5d3420838537d0d320`、base、VAE、
normalization stats、两相机 224 配置和严格标签合同；durable 输出目录没有 fallback，最终
Stage 3 Adapter 路径/SHA 在冻结前保持 fail-closed placeholder：

```bash
export FASTWAM_LIBERO_STAGE2_LABEL_JOB=/absolute/durable/path/libero/label_job
export FASTWAM_LIBERO_STAGE3_ADAPTER=/absolute/path/libero_stage3_adapter.pt
export FASTWAM_LIBERO_STAGE3_ADAPTER_SHA256=REAL_LIBERO_ADAPTER_SHA256

torchrun --standalone --nproc_per_node=8 \
  scripts/generate_gate_labels.py \
  task=libero_stage2_gate_labels_2cam224
```

只有显式迁移已锁资产时，才覆盖对应的
`FASTWAM_LIBERO_STAGE3_BASE_{CHECKPOINT,SHA256}`、
`FASTWAM_LIBERO_STAGE3_DATA_MANIFEST{,_SHA256}`、`FASTWAM_LIBERO_STAGE3_VAE` 和
`FASTWAM_LIBERO_STATS` 环境变量；VAE 与 stats 的 SHA 仍由 task 固定。下面的长 override 入口
保留给自定义路径、单进程诊断和旧部署迁移；其合同语义与专用 task 相同。

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

RoboTwin 不再复制上面的长 override 列表，而使用已锁定的正式 task。它固定
`robotwin_formal`、3-camera/384 配置、`seed=42`、两对 seed、10-step、64 shards、
chunk size 64、BF16 和 strict TorchCodec：

```bash
export FASTWAM_ROBOTWIN_STAGE2_LABEL_JOB=/absolute/durable/path/robotwin/label_job
export FASTWAM_ROBOTWIN_STAGE3_ADAPTER=/absolute/path/robotwin_stage3_adapter.pt
export FASTWAM_ROBOTWIN_STAGE3_ADAPTER_SHA256=REAL_ROBOTWIN_ADAPTER_SHA256

torchrun --standalone --nproc_per_node=8 \
  scripts/generate_gate_labels.py \
  task=robotwin_stage2_gate_labels_3cam384
```

task 已内置正式 schema-v2 manifest SHA，只有 Adapter 路径/SHA 的默认值仍是故意无效的
placeholder；最终 Stage 3 export 尚未冻结时会 fail closed。只有显式迁移或重新生成数据
身份时，才应连同 manifest 路径一起覆盖
`FASTWAM_ROBOTWIN_STAGE3_DATA_MANIFEST_SHA256`。`FASTWAM_ROBOTWIN_STAGE2_LABEL_JOB` 没有
本地 fallback，避免 8 个 rank 因时间戳或工作目录漂移写到不同任务。
`label_contract.json` 的 `contract_sha256` 只能在所有身份和运行环境确定后计算，因此它不是
预填输入：由该步骤原子发布，再由 merge 和 Gate 配置显式锁定。

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

merge 入口与 benchmark 无关；RoboTwin 必须把本段变量指向它自己的 label job、schema-v2
data manifest 和输出目录。不要把 LIBERO chunk 与 RoboTwin chunk 放进同一目录。

## 第三步：只训练 BinaryVideoGate

当前已合并的 LIBERO formal 子集先运行完整一轮单卡验收：

```bash
bash /root/feihong/FAST_WAM_github/scripts/jihe/run_libero_stage2_gate_smoke
```

这个短入口固定消费 `selection_426b635d_d75c04a_formal_d114ac25`，使用普通单进程
Python（不能用 torchrun），在一张 H100 上训练 1 epoch：48,768 个 train 样本、5,408 个
validation 样本、batch 64、共 762 次更新。它只实例化 658,977 参数的
`BinaryVideoGate`；5B base 和 Stage 3 Adapter 只校验 SHA，不加载权重。视频解码期间每
60 秒输出一次 heartbeat，完整 epoch 结束后才发布训练指标。

验收通过必须同时得到：

```text
gate_run/run_identity.json
gate_run/training_state.pt
gate_run/gate_best.pt
gate_run/gate_last.pt
gate_run/summary.json
verification_receipt.json
```

验收器会严格加载 best/last 两份 Gate export，检查 epoch/step、样本数、有限指标、参数量、
当前 clean Git 身份以及 label/base/Adapter/data/split 的全部哈希绑定；同时重算完整训练
contract 的 canonical SHA，并用正式标签类别计数重建 Gate/AdamW，真实恢复
`training_state.pt`。恢复后会核对 optimizer 的 762 step、best/last 权重，并额外执行一次
不落盘的 synthetic update，确认状态确实能够继续训练。该 1-epoch smoke 的训练身份与正式
20-epoch run 不同，不能把 smoke checkpoint 直接续成正式训练。

通用的显式 Gate 命令仍如下；正式 launcher 应从新目录 fresh 启动：

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

RoboTwin Gate 训练把 `data=libero_2cam` 改为 `data=robotwin_formal`，并使用 RoboTwin
自己的 manifest、split、label contract/manifest、base/Adapter SHA 和 stats。其余严格
合并、恢复和 no-clobber 语义相同；两个 Gate 仍是两个独立 run。

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

Stage 2 还会对 manifest 选中的 parquet、MP4、text cache 身份和 stats 建立源文件快照。
schema-v1 直接绑定选中的 cache payload；schema-v2 绑定 descriptor/index，并在实际读取
每个 cache payload 时先核对 index 中的 size/SHA256。长任务还会在 chunk/epoch 边界检查
源身份，并在合并或最终发布前完整复核；路径、inode、size、mtime/ctime 或 SHA256 漂移
都会 fail closed。

正式标签任务通过临时 SQLite B-tree 对选中 sample ID 做磁盘外排；Python 内存只保留固定
插入批次和当前 chunk，不会一次物化 RoboTwin 的数百万 sample plan。外排仍严格保持
`shard -> sample_id -> chunk_index` 的既有 canonical 顺序，resume artifact 路径不变。

所有 durable 标签 artifact 都采用 no-clobber 发布：目标不存在时才原子创建；目标已存在
时只能接受严格一致的完整内容。Gate fresh run 同样拒绝覆盖，只有通过身份校验的显式
resume 才能继续。

## 已验证边界

本链路的 schema-v1/v2 contract、标签数学、分片/合并、current-only Gate dataset、Gate
optimizer、严格恢复和 LIBERO/RoboTwin Hydra task 由 CPU 单元与入口测试覆盖。RoboTwin 的真实
AV1/data-shape smoke 已通过，但本文没有声称完成任一 benchmark 的真实 5B H100 端到端
标签生成或 Gate GPU 验收。正式长任务前仍需分别完成 CUDA 短标签分片、merge 和 Gate
save→resume 验收。
