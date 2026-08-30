# RoboTwin 2.0 Stage 2/3 formal readiness

RoboTwin 2.0 已有独立 Stage 3 Hydra 入口
`task=robotwin_stage3_alignment_3cam384_1e-4`，以及 Stage 3 完成后使用的 Stage 2
标签入口 `task=robotwin_stage2_gate_labels_3cam384`。两个入口都复现 UnifiedShared base
的 `seed=42`、1% validation episode split；formal train 固定为 `6011575` frames /
`27225` episodes。

schema-v2 text-cache index/descriptor 与 data manifest 已发布，Stage 3 和 Stage 2 task
均默认锁定 canonical manifest SHA
`1190b75b1ef19a7abd949bdff5679da59afa7e51a043eeb43663cf2c4495173c`。数据身份门槛已
完成；真实单卡 H100、8×H100 ZeRO-2 严格 save/resume、200-step pilot 和单 episode
endpoint health smoke 均已通过。当前下一步是正式 Stage 3 长训练。

## 已确认的真实数据与资产

完整 RoboTwin 数据为 `27500` episodes / `6075103` frames，包含 `27500` 个 parquet、
`82500` 个 MP4（三路相机）和 `921032` 个 text-cache payload；数据约 75 GB，text cache
约 903 GB。按正式 seed/split 实例化后，train 部分含 `914763` 个唯一 selected prompt，
schema-v2 index 只绑定这组 prompt，不把未选 cache 混入身份。

真实 formal-train 样本已经在 strict TorchCodec、无 fallback 模式下通过：

```text
video         (3, 9, 384, 320)  float32
action        (32, 14)          float32
proprio       (32, 14)          float32
context       (128, 4096)       bfloat16
context_mask  (128,)            bool
```

首、中、末 episode 的三路真实 AV1 共 9 个视频也已通过严格解码 smoke。当前软件环境已
确认 Torch `2.7.1+cu128`、TorchCodec `0.5+cu128`、FFmpeg `4.4.2`。这些是数据/decoder
smoke，不是 5B 模型的 GPU 训练 smoke。

已确认的固定资产为：

- base：`/root/feihong/FastWAM/formal_runs/FAST_WAM_github/robotwin_unified_shared_3cam_384_1e-4/2026-07-01_00-51-30/checkpoints/weights/latest.pt`；
- base SHA256：`368a99ca9575a78d01f4cdcdee8820ec74d30c4528cf7aff07b83361a17cbbda`；
- normalization stats SHA256：`7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095`；
- Wan2.2 VAE SHA256：`20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36`。

## 已完成的数据身份

以下命令可复现选中 prompt 的 compact index/descriptor；builder 从数据集推导精确 cache 路径，
不扫描整个 cache root，`workers=32` 只影响并行哈希吞吐，不影响排序或 canonical identity：

```bash
python scripts/build_text_cache_index.py \
  task=robotwin_stage3_alignment_3cam384_1e-4 \
  +text_cache_index.path=/root/feihong/FastWAM/formal_runs/contracts/stage3/robotwin_train_6011575f_27225e/robotwin_text_cache_index.bin \
  +text_cache_index.descriptor_path=/root/feihong/FastWAM/formal_runs/contracts/stage3/robotwin_train_6011575f_27225e/robotwin_text_cache_index.json \
  +text_cache_index.workers=32
```

再生成 schema-v2 manifest：

```bash
python scripts/build_stage3_data_manifest.py \
  task=robotwin_stage3_alignment_3cam384_1e-4
```

已发布结果为：

- index：`914763` records / `65863096` bytes，SHA256
  `57d5820e1d7c7cc327884c22c13c721fc7830938e126259f6f61548c0e3b4228`；
- descriptor semantic SHA256：
  `3d5dc4b56703705803b1431090d00c63fd255a97d8524b3db326105ad056365e`；
- schema-v2 manifest：`108904` 个静态文件 / `40490762` bytes，canonical SHA256
  `1190b75b1ef19a7abd949bdff5679da59afa7e51a043eeb43663cf2c4495173c`。

发布后已在新实例化的 formal split 上完成一次独立 `full_content_verify=True`：逐文件内容、
episode topology、prompt set、stats、descriptor/index 全部一致；随后真实读取的 Stage 3
完整样本和 Stage 2 `label_only()` 样本也都通过 cache payload SHA 与 AV1 解码检查。

task 已内置该 canonical SHA；只有显式迁移或重新生成身份时才应通过
`FASTWAM_ROBOTWIN_STAGE3_DATA_MANIFEST_SHA256` 连同对应路径一起覆盖。正式 runtime 会复核
manifest、descriptor、index、prompt set、静态数据/stats，并把验证后的 index 绑定到
dataset；实际 cache payload 必须先通过 size/SHA256 校验，之后才能反序列化。

## Batch 合同与 GPU 验收

8 卡 H100 的锁定配置是每卡 batch 2、gradient accumulation 3：每个 epoch 有
`375723` 个 global micro-batch，能组成完整 accumulation group，有效 global batch 48，
tail 为 7 个 sample。manifest SHA 有效后先跑单卡一步：

```bash
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_stage3_single_gpu.yaml \
  scripts/train_stage3_alignment.py \
  task=robotwin_stage3_alignment_3cam384_1e-4 \
  training.batch_size=1 \
  training.gradient_accumulation_steps=1 \
  training.max_steps=1
```

提交 `001ba776ad8d06ea471e60c781ccca72d21cd3a1` 上的真实单卡 H100 smoke 已完成：batch 1、
accumulation 1 的一步训练得到 loss `0.000199`、grad norm `0.206138`，训练计算约 `8.4s`；
`COMPLETE`、Adapter-only export 和完整恢复 state 均已发布。随后从独立进程严格恢复到
`step=1 epoch=0 batch=1`，model、optimizer、scheduler、sampler 和 RNG 全部加载成功，恢复前后
Adapter SHA256 均为
`4686621e6eda37133a63506839772786275da3fdd473af68aefeefedfe79aec2`，training contract SHA256
均为 `216ea9a6b7567415b84a7aaf6fccc4168fdb7cac422d8f96a23488d8ff7a5d0f`。该 export 仅是
smoke 证据，不能用于正式 Stage 2 标签。后续 8×H100 ZeRO-2 fresh/resume exact
equivalence 已通过，200-step pilot 也已完成；两类产物都不能替代正式最终 Adapter。

另在提交 `4946d17` 上用正式每卡合同 `batch_size=2`、accumulation 3 跑完一个 optimizer
step：loss `0.004577`、grad norm `0.198295`、三次 micro-batch 计算共 `24.0s`。对 H100
每 `0.5s` 采样得到峰值显存 `15957 MiB`、峰值利用率 83%，因此无需把正式合同降到
batch 1 / accumulation 6；后续 8 卡 NCCL/ZeRO smoke 已确认该正式 batch 合同可运行。

## 与 LIBERO 对齐后的顺序

RoboTwin 与 LIBERO 的 data contract、单卡检查、8 卡 ZeRO-2 save/resume、200-step pilot
和 endpoint connectivity smoke 均已完成。下一步同时启动两个独立正式 Stage 3 run。
各自 Adapter 冻结并通过最终小规模 `w` 分支 health smoke 后，
直接开始 Stage 2 标签生成；完整 endpoint 对比推迟到 Gate 完成后统一执行：
生成标签、严格 merge、训练 Gate。两个 Stage 2 task 当前都保留无效 Adapter placeholder；
在各自 Stage 3 最终 export 与真实 SHA 出现前不会启动。两个 benchmark 绝不共享 Adapter、
label contract、label manifest 或 Gate checkpoint。
