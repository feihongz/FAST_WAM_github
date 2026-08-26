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

正式入口不会把“路径相同、长度接近”当成同一份数据。先对最终选定的数据生成
manifest；该过程会绑定 episode 顺序与边界、parquet、两路 MP4、40 个 text cache、
metadata、normalization stats、decoder 和采样配置，并读取所有选中文件计算 SHA256：

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

## 当前 LIBERO 训练配置

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

任务配置锁定以下身份：

- UnifiedShared base checkpoint SHA256；
- Wan2.2 VAE SHA256；
- LIBERO normalization stats SHA256；
- selected-data manifest SHA256；
- Git commit，以及 Torch/Accelerate/DeepSpeed/decoder/data library 版本；
- 有效 DeepSpeed、dataloader、batch、gradient accumulation 合同。

正式运行要求 tracked worktree clean，且 `src/configs/scripts/tests` 下不能存在未跟踪
文件。数据读取采用 strict mode：坏样本直接报错，固定 `torchcodec`，禁止静默切换
pyav 或随机换样本。路径可以通过任务配置中列出的
`FASTWAM_STAGE3_*` 环境变量覆盖，但对应 SHA 也必须一起更新。

当前 formal v1 只支持 ZeRO 0–2、无 optimizer offload，并要求每个 epoch 都由完整的
gradient-accumulation group 组成。base/VAE/stats 和 data manifest 会在每个 rank 独立
校验内容，之后才允许加载和训练。

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
严格恢复、RNG、分布式 tail 分片算法、10-step 接入和 strict data mode。真实 5B CUDA
单批、单卡 ZeRO-2 save→resume，以及真实多进程恢复仍是正式训练前的 GPU 验收项。

本链锁定的 official Wan2.2 VAE PTH 可保证本次 Stage 3 自身稳定；原 base 日志使用的
converted safetensors 当前不在机器上，因此在恢复原文件或完成 tensor digest 对比前，
不要声称 Stage 3 VAE 与 2026-07-01 训练容器逐 tensor 完全相同。

## Stage 3 之后

Stage 3 Adapter 冻结并完成 closed-loop endpoint eval 后，才重新生成 Stage 2 的
`E0/E10` 标签并训练 Binary Gate。最终 eval 只路由 `N=0` 或 `N=10`，阈值扫描使用
`N_eff = 10 * n_w / n_queries` 作为横坐标绘制 success-compute Pareto 图。完整的
标签分片、严格合并和 Gate-only 训练命令见
[Stage 2 Binary Video Gate](./STAGE2_GATE_TRAINING.md)。
