# Stage 3 Action Alignment：正式训练链

本链路独立于旧 Stage 2/Gate 实验，只训练 `N=10` 分支使用的
Action Alignment Adapter。冻结的 UnifiedShared 5B base 不会交给
Accelerate/DeepSpeed，也不会进入 optimizer 或训练 checkpoint。

## 唯一入口

```text
scripts/train_stage3_alignment.py
  -> fastwam.alignment.runtime.run_stage3_alignment_training
  -> FastWAMUnifiedAligned.load_frozen_base_checkpoint (strict=True)
  -> Stage3AlignmentTrainer
  -> AlignmentVelocityModule (Adapter only)
```

不要用 `scripts/train.py` 或 `Wan22Trainer` 训练 Stage 3；它们属于原始模型训练链。

## 先锁定训练数据

正式入口不会把“路径相同、长度接近”当成同一份数据。两个 benchmark 使用相同的
fail-closed 运行原则，但 manifest 的 cache 表达不同：

| benchmark | manifest | 文本 cache 身份 |
| --- | --- | --- |
| LIBERO | schema-v1 / `stage3_libero_data_manifest` | 选中 cache payload 与 parquet、两路 MP4、metadata、stats 一起 inline 枚举并哈希 |
| RoboTwin 2.0 | schema-v2 / `stage3_robot_video_data_manifest` | manifest 绑定 descriptor + fixed-record binary index；index 覆盖训练 split 的精确 prompt 集与每个 payload 的 size/SHA256 |

schema-v2 的 index 只从实例化后的训练数据推导精确 cache 路径，不遍历整个 cache 目录；
正式 loader 在 cache miss 时先验证 payload 字节，再从同一份字节反序列化。descriptor、
index、prompt set 或 payload 任一身份不一致都会失败。

LIBERO schema-v1 manifest 已生成并锁定：

```bash
python scripts/build_stage3_data_manifest.py \
  task=libero_stage3_alignment_2cam224_1e-4 \
  runtime.expected_dataset_length=273465 \
  runtime.expected_dataset_episodes=1693 \
  +data.train.save_stats_copy=false \
  data_manifest.path=/root/feihong/FastWAM/formal_runs/contracts/stage3/libero_current_273465f_1693e/libero_stage3_data_manifest.json
```

当前机器上的 LIBERO 数据是 `273465` frames / `1693` episodes；2026-07-01 base
训练记录与 stats 是 `277713` frames / `1712` episodes。2026-08-26 已明确接受当前
数据用于 Stage 3，同时继续锁定 base 对应的 normalization stats。任务配置默认绑定：

```text
path=/root/feihong/FastWAM/formal_runs/contracts/stage3/libero_current_273465f_1693e/libero_stage3_data_manifest.json
canonical_sha256=08da49109a57b55c67f3fa4ac31fbfa44e44dd541a194a5d3420838537d0d320
frames=273465
episodes=1693
```

若在另一台机器使用内容完全相同的 manifest 副本，可通过环境变量同时覆盖路径与
canonical SHA256：

```text
FASTWAM_STAGE3_DATA_MANIFEST=/absolute/path/libero_stage3_data_manifest.json
FASTWAM_STAGE3_DATA_MANIFEST_SHA256=08da49109a57b55c67f3fa4ac31fbfa44e44dd541a194a5d3420838537d0d320
```

RoboTwin 必须先建 cache index，再建 schema-v2 manifest：

```bash
python scripts/build_text_cache_index.py \
  task=robotwin_stage3_alignment_3cam384_1e-4 \
  +text_cache_index.path=/root/feihong/FastWAM/formal_runs/contracts/stage3/robotwin_train_6011575f_27225e/robotwin_text_cache_index.bin \
  +text_cache_index.descriptor_path=/root/feihong/FastWAM/formal_runs/contracts/stage3/robotwin_train_6011575f_27225e/robotwin_text_cache_index.json \
  +text_cache_index.workers=32

python scripts/build_stage3_data_manifest.py \
  task=robotwin_stage3_alignment_3cam384_1e-4
```

当前已确认的 RoboTwin formal train split 是 `6011575` frames / `27225` episodes，含
`914763` 个唯一 selected prompt。index 与 schema-v2 manifest 已发布，task 默认锁定
canonical manifest SHA
`1190b75b1ef19a7abd949bdff5679da59afa7e51a043eeb43663cf2c4495173c`。

## 训练配置与启动顺序

两类 8×H100 save/resume smoke、200-step pilot 和单 episode `w` 分支 endpoint
health smoke 均已通过。现在同时启动两个独立的正式 Stage 3 run；不要先生成 Stage 2
标签，标签合同必须绑定各自最终冻结的 Stage 3 Adapter。

### 锁定的正式预算

| benchmark | optimizer steps | 数据暴露 | checkpoint | 8×H100 预计耗时 |
| --- | ---: | ---: | ---: | ---: |
| LIBERO | `56,970` | 10 epochs / 2,734,560 windows | 每 1,899 步，保留 32 份 | 36–43 小时 |
| RoboTwin 2.0 | `20,000` | 0.1597 epoch / 960,000 windows | 每 500 步，保留 41 份 | 84–96 小时 |

LIBERO 的一个 epoch 是 5,697 optimizer steps，因此沿用 task 原本的 10-epoch
训练语义。RoboTwin 的一个 epoch 是 125,241 steps，按 pilot 吞吐约 21.5 天；禁止让
`max_steps=null` 落入默认 10 epochs，正式预算显式截为 20,000 steps。

两边均锁定 8×H100、每卡 batch 2、gradient accumulation 3、有效 global batch 48、
BF16、AdamW `lr=1e-4` / `betas=(0.9,0.95)` / `weight_decay=1e-4`、gradient clip 1.0、
5% linear warmup 后 cosine 到 `1e-6`、seed 42，以及现有 N=10、margin 和 loss 权重。
日志每 100 个成功 optimizer steps 输出一次，指标是该窗口内全部 accumulation
microbatch 的跨 rank 均值，而不是最后一个 microbatch 的瞬时值。

正式提交只使用下面两个零参数入口，分别作为两个独立极核任务：

```bash
bash scripts/jihe/run_libero_stage3_full_8xh100.sh
```

```bash
bash scripts/jihe/run_robotwin_stage3_full_8xh100.sh
```

200-step pilot 的 scheduler/strict-resume 合同已经在 step 200 结束，不能作为正式 run
的 resume 起点。正式 run 必须 fresh 启动；若正式任务中断，可用同一正式合同的完整
`states/step_*` 或 `states/LATEST` 启动新 attempt：

```bash
RESUME_STATE=/absolute/previous/attempt/checkpoints/states/LATEST \
  bash scripts/jihe/run_libero_stage3_full_8xh100.sh
```

resume 会写入新的唯一输出目录；不能把 Adapter-only `.pt` export 当作训练状态。

下面的单卡和裸 `accelerate` 命令仅用于开发，不是正式极核入口。

LIBERO 单卡入口：

```bash
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_stage3_single_gpu.yaml \
  scripts/train_stage3_alignment.py \
  task=libero_stage3_alignment_2cam224_1e-4
```

多卡 ZeRO-2 使用专用配置：

```bash
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_stage3_zero2.yaml \
  --num_processes 8 \
  scripts/train_stage3_alignment.py \
  task=libero_stage3_alignment_2cam224_1e-4
```

RoboTwin 使用同一 launcher、不同 task 和独立输出目录：

```bash
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_stage3_zero2.yaml \
  --num_processes 8 \
  scripts/train_stage3_alignment.py \
  task=robotwin_stage3_alignment_3cam384_1e-4
```

两个 task 均锁定每卡 batch 2、gradient accumulation 3，有效 global batch 48。
LIBERO 每 epoch 的 global micro-batch 数为 `17091`、tail 为 9；RoboTwin 为
`375723`、tail 为 7；两者都能构成完整 accumulation group。

任务配置锁定以下身份：

- UnifiedShared base checkpoint SHA256；
- Wan2.2 VAE SHA256；
- 对应 benchmark 的 normalization stats SHA256；
- selected-data manifest SHA256；
- Git commit，以及 Torch/Accelerate/DeepSpeed/decoder/data library 版本；
- 有效 DeepSpeed、dataloader、batch、gradient accumulation 合同。

正式运行要求 tracked worktree clean，且 `src/configs/scripts/tests` 下不能存在未跟踪
文件。数据读取采用 strict mode：坏样本直接报错，固定 `torchcodec`，禁止静默切换
pyav 或随机换样本。路径可以通过任务配置中列出的
`FASTWAM_STAGE3_*` / `FASTWAM_ROBOTWIN_STAGE3_*` 环境变量覆盖，但对应 SHA 也必须
一起更新。

正式运行合同支持 schema-v1 和 schema-v2 manifest、ZeRO 0–2、无 optimizer offload，
并要求每个 epoch 都由完整的 gradient-accumulation group 组成。base/VAE/stats 和 data
manifest 的完整内容只由 global main rank 哈希一次；随后每个 rank 都会独立核对本地
split、episode/file topology、size、prompt set 与 decoder，并验证 descriptor/index、把
manifest 固定的不可变 cache identity 绑定到 dataset。所有 rank 完成这些步骤后，才允许
加载 cache payload 和训练。

## 8 卡 smoke 的可重复验收

极核上优先使用 benchmark 对应的一键脚本。它会自动创建唯一输出目录，连续完成 fresh
两步训练、从 step 1 恢复到独立目录、严格比较两个 step-2 Adapter，并保存 JSON 验收回执：

```bash
bash scripts/jihe/run_libero_stage3_smoke_8xh100.sh
bash scripts/jihe/run_robotwin_stage3_smoke_8xh100.sh
```

只有 8 张 H100 时应顺序提交这两个任务。下面的命令是脚本内部执行的等价手工流程，正常
启动不需要再填写这些参数。

每个 benchmark 单独跑一次不间断的两个 optimizer step，保留 step 1 和 step 2：

```bash
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_stage3_zero2.yaml \
  --num_processes 8 \
  scripts/train_stage3_alignment.py \
  task=robotwin_stage3_alignment_3cam384_1e-4 \
  output_dir=/absolute/smoke/uninterrupted \
  training.max_steps=2 \
  checkpoint.save_every=1 \
  checkpoint.keep_last=2 \
  checkpoint.save_final=true
```

`max_steps` 属于严格训练合同，恢复运行必须仍为 2。`LATEST` 此时已经指向 step 2，因此要
显式从 step 1 恢复到独立输出目录，使恢复路径重新执行且只执行第二个 optimizer step：

```bash
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_stage3_zero2.yaml \
  --num_processes 8 \
  scripts/train_stage3_alignment.py \
  task=robotwin_stage3_alignment_3cam384_1e-4 \
  output_dir=/absolute/smoke/resume_from_step1 \
  training.max_steps=2 \
  checkpoint.save_every=1 \
  checkpoint.keep_last=2 \
  checkpoint.save_final=true \
  checkpoint.resume=/absolute/smoke/uninterrupted/checkpoints/states/step_000001
```

新 writer 使用 training-state schema v2。fresh step 1/2 的 `strict_resume_provenance` 为
`null`；恢复后的 step 2 必须记录所加载 step 1 的 manifest/COMPLETE SHA、cursor、training
contract、world size 和 ZeRO stage。旧 schema-v1 state 仍可恢复，但缺少 lineage，不能作为
新的正式验收产物。

两个 step-2 export 的原始文件 SHA 不一定相同，因为 `torch.save` 容器字节不保证确定性。
正式验收同时传入三个 state 和两个外部 export；下面三个 SHA 必须替换为本 benchmark 的
锁定值，其中 training contract SHA 从 fresh run 的 state manifest 读取：

```bash
python scripts/verify_stage3_resume_equivalence.py \
  /absolute/smoke/uninterrupted/checkpoints/states/step_000002 \
  /absolute/smoke/uninterrupted/checkpoints/states/step_000001 \
  /absolute/smoke/resume_from_step1/checkpoints/states/step_000002 \
  /absolute/smoke/uninterrupted/checkpoints/exports/step_000002.pt \
  /absolute/smoke/resume_from_step1/checkpoints/exports/step_000002.pt \
  --expected-final-step 2 \
  --expected-resume-step 1 \
  --expected-world-size 8 \
  --expected-zero-stage 2 \
  --expected-batch-size-per-rank 2 \
  --expected-gradient-accumulation-steps 3 \
  --expected-base-checkpoint-sha256 REAL_BASE_SHA256 \
  --expected-data-manifest-sha256 REAL_DATA_MANIFEST_SHA256 \
  --expected-training-contract-sha256 REAL_TRAINING_CONTRACT_SHA256 \
  --expected-git-commit REAL_40_OR_64_HEX_GIT_COMMIT
```

验收器会重新验证三个 `COMPLETE`、manifest/file inventory、training/DeepSpeed 内部 SHA、
world size 8、ZeRO-2、每卡 batch 2、accumulation 3、8 份可加载 random state、外部 export
与 state 内 export 的绑定，以及恢复 step 2 对指定 step 1 的精确 provenance。最后才比较
完整 metadata 和 16 个 Adapter tensor。LIBERO 使用相同流程，只替换 task、锁定 SHA 和独立
输出目录；两个 benchmark 的产物不得交叉复用。

运行验收器的 shell 必须保持保存时的 8 张 GPU 全部可见；不要把
`CUDA_VISIBLE_DEVICES` 缩窄到单卡，否则 CUDA RNG device-count 校验会按设计拒绝该 state。

## Checkpoint 与恢复

每个保存点包含两种轻量产物：

```text
checkpoints/
├── exports/step_XXXXXX.pt       # eval 直接加载的 Adapter-only export
└── states/step_XXXXXX/          # optimizer/scheduler/RNG 严格恢复状态
```

恢复时传 state 目录或 `LATEST` 文件，不能把 export 当作 resume state：

```bash
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_stage3_single_gpu.yaml \
  scripts/train_stage3_alignment.py \
  task=libero_stage3_alignment_2cam224_1e-4 \
  checkpoint.resume=/absolute/run/checkpoints/states/LATEST
```

`COMPLETE` 绑定 manifest SHA；恢复前会复核文件 inventory、RNG 可加载性、
accumulation boundary、scheduler/global step、base/assets 和运行合同。

## 已验证范围

CPU contract/unit tests 已覆盖 Adapter-only optimizer/checkpoint、mid-epoch 与 epoch-tail
严格恢复、RNG、分布式 tail 分片算法、10-step 接入、strict data mode，以及 schema-v2
descriptor/index 的生成、验证和 runtime loader binding。RoboTwin 已用真实 formal train
样本通过 strict TorchCodec AV1/data-shape smoke，真实 index/manifest 也已发布；单卡 H100
真实 5B CUDA 一步和严格 save→resume 已在提交 `001ba77` 上通过，正式 per-rank
batch 2 / accumulation 3 也在提交 `4946d17` 上通过，实测峰值显存 `15957 MiB`。RoboTwin
随后完成真实 8 进程 ZeRO-2 fresh 两步、从 step 1 严格恢复并重放第二步，验收器确认
两个 step-2 Adapter 逐 tensor 完全一致。

LIBERO 在提交 `83345ba` 上以 `batch_size=2`、accumulation 1 完成真实 H100 schema-v2
两步连续训练，并从连续 run 的 `step_000001` 恢复到独立目录、只重放第二步。恢复日志确认
model、optimizer、scheduler、sampler 和 RNG 全部加载，游标为
`step=1 epoch=0 batch=1`。正式验收器同时核验 fresh step 1、fresh step 2、resumed step 2
及两份外部 export，结果为 `status=ok`；恢复 provenance 精确绑定 source manifest、
`COMPLETE`、cursor、training contract、world size 1 和 ZeRO stage 0。两条路径的 step-2
Adapter export SHA256 均为
`309acf72066cca2a6dbc02e6d80dd2cbb6e426f49c6058decf65aeb267f0c0cc`，16 个 tensor、
`1577479` 个参数逐值相等，training contract SHA256 均为
`4416aff2f3a54d74abcc40b78e67b8053a834eac005325790f9fa7dbf4a24c59`。产物位于
`formal_runs/smokes/stage3/libero_single_replay_83345ba_001`。LIBERO 后续也完成真实 8 进程
ZeRO-2、每卡 batch 2、accumulation 3 的 fresh/resume exact-equivalence 验收。两个
benchmark 的 200-step pilot 和单 episode endpoint connectivity smoke 均已通过；任一
smoke 或 pilot export 都不得当作正式 Stage 3 最终 Adapter 使用。

本链锁定的 official Wan2.2 VAE PTH 可保证本次 Stage 3 自身稳定；原 base 日志使用的
converted safetensors 当前不在机器上，因此在恢复原文件或完成 tensor digest 对比前，
不要声称 Stage 3 VAE 与 2026-07-01 训练容器逐 tensor 完全相同。

## Stage 3 之后

每个 benchmark 按以下顺序继续：

1. 冻结预先约定的最终 step Adapter，并记录 export、base、data、VAE、stats、training
   contract 和 Git SHA；
2. 只跑极小的 `w` 分支 health smoke，确认最终 artifact 能严格加载和执行，不做完整
   `w-only` benchmark eval；
3. 直接用冻结 Adapter 生成 matched `wo/w` 数据和 Stage 2 `E0/E10` 标签；
4. 训练对应的 Binary Gate；
5. 最后一次性做有意义的完整对比：always-`wo`、always-`w` 和 Gate 路由。

跳过的是单独的完整 `w-only` eval，不是 Adapter 身份冻结和最终 health smoke。两个
benchmark 可以并行推进，但 Adapter、标签、split、contract、merged manifest、Gate 和
输出目录必须完全分开。最终 eval 只路由 `N=0` 或 `N=10`，阈值扫描使用
`N_eff = 10 * n_w / n_queries` 作为横坐标绘制 success-compute Pareto 图。完整的标签分片、
严格合并和 Gate-only 训练命令见
[Stage 2 Binary Video Gate](./STAGE2_GATE_TRAINING.md)。
