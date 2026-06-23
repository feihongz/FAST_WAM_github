# Incremental Fast-WAM Training

This repository includes two incremental Fast-WAM variants for comparing whether
`wo-video` and `w-video` action learning should share one ActionDiT or use two
separate ActionDiTs.

## Variants

| Config | Model class | Training behavior |
| --- | --- | --- |
| `fastwam_unified_shared` | `FastWAMUnifiedShared` | One shared WAN/video DiT and one shared ActionDiT. Each batch runs both `wo-video` and `w-video` masks; the two action losses update the same ActionDiT. |
| `fastwam_unified_two_action` | `FastWAMUnifiedTwoAction` | One shared WAN/video DiT and two ActionDiTs. `action_expert_wo` trains with the Fast-WAM mask; `action_expert_w` trains with the Joint mask. Both branches update the shared video DiT. |

The `w-video` path is Joint-style video/action denoising, not IDM-style
video-then-action inference.

## Required external files

Large data and checkpoints are intentionally not committed.

Before training, prepare:

1. Wan2.2 model files under `checkpoints/` or another directory pointed to by
   `DIFFSYNTH_MODEL_BASE_PATH`.
2. The ActionDiT backbone:

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"

python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

3. The preprocessed LIBERO or RoboTwin datasets described in `README.md`.

## LIBERO training

Precompute text embeddings once. The unified shared and two-action LIBERO
configs use the same text cache.

```bash
python scripts/precompute_text_embeds.py task=libero_unified_shared_2cam224_1e-4 +overwrite=false
```

Train Unified-Shared:

```bash
bash scripts/train_zero1.sh 8 task=libero_unified_shared_2cam224_1e-4
```

Train Unified-TwoAction:

```bash
bash scripts/train_zero1.sh 8 task=libero_unified_two_action_2cam224_1e-4
```

## RoboTwin training

Precompute text embeddings:

```bash
python scripts/precompute_text_embeds.py task=robotwin_unified_shared_3cam_384_1e-4 +overwrite=false
```

Train Unified-Shared:

```bash
bash scripts/train_zero1.sh 8 task=robotwin_unified_shared_3cam_384_1e-4
```

Train Unified-TwoAction:

```bash
bash scripts/train_zero1.sh 8 task=robotwin_unified_two_action_3cam_384_1e-4
```

RoboTwin is substantially larger than LIBERO. Use more GPUs if available.

## Smoke training

The smoke commands below run a few real training steps and save a final
checkpoint. They assume the external data, Wan2.2 files, and ActionDiT backbone
are already available.

Unified-Shared smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 4 \
  scripts/train.py \
  task=libero_unified_shared_2cam224_1e-4 \
  output_dir=./runs/smoke_unified_shared \
  max_steps=3 \
  batch_size=1 \
  num_workers=0 \
  log_every=1 \
  eval_every=0 \
  save_every=0 \
  wandb.enabled=false \
  model.mot_checkpoint_mixed_attn=true
```

Unified-TwoAction smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 6 \
  scripts/train.py \
  task=libero_unified_two_action_2cam224_1e-4 \
  output_dir=./runs/smoke_unified_two_action \
  max_steps=3 \
  batch_size=1 \
  num_workers=0 \
  log_every=1 \
  eval_every=0 \
  save_every=0 \
  wandb.enabled=false \
  model.mot_checkpoint_mixed_attn=true
```

On H100 80GB GPUs, Unified-Shared completed the smoke run on 4 GPUs and
Unified-TwoAction completed on 6 GPUs. Fewer GPUs may run out of memory during
ZeRO optimizer steps.

### RoboTwin smoke

RoboTwin 2.0 has a much larger instruction table than LIBERO. A full text-cache
precompute scans all tasks and can take a long time. For a minimal training
smoke, use one fixed instruction for all samples and precompute only that prompt:

```bash
python scripts/precompute_text_embeds.py \
  task=robotwin_unified_shared_3cam_384_1e-4 \
  '+override_instruction=Grab the smooth green plastic bottle and lift it with the left arm' \
  +overwrite=false
```

Unified-Shared RoboTwin smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/train_zero1.sh 6 \
  task=robotwin_unified_shared_3cam_384_1e-4 \
  output_dir=./runs/smoke_robotwin_unified_shared \
  batch_size=1 \
  num_workers=0 \
  max_steps=3 \
  log_every=1 \
  save_every=0 \
  eval_every=0 \
  model.mot_checkpoint_mixed_attn=true \
  '+data.train.override_instruction=Grab the smooth green plastic bottle and lift it with the left arm' \
  '+data.val.override_instruction=Grab the smooth green plastic bottle and lift it with the left arm'
```

Unified-TwoAction RoboTwin smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/train_zero1.sh 6 \
  task=robotwin_unified_two_action_3cam_384_1e-4 \
  output_dir=./runs/smoke_robotwin_unified_two_action \
  batch_size=1 \
  num_workers=0 \
  max_steps=3 \
  log_every=1 \
  save_every=0 \
  eval_every=0 \
  model.mot_checkpoint_mixed_attn=true \
  '+data.train.override_instruction=Grab the smooth green plastic bottle and lift it with the left arm' \
  '+data.val.override_instruction=Grab the smooth green plastic bottle and lift it with the left arm'
```

On H100 80GB GPUs, both RoboTwin smoke commands above completed with 6 GPUs.
With fewer GPUs, the ZeRO optimizer step may run out of memory.

## Static evaluation mode

Unified models expose:

```python
model.infer_action_mode(..., inference_mode="wo")
model.infer_action_mode(..., inference_mode="w")
```

Evaluation configs default to:

```yaml
EVALUATION:
  inference_mode: wo
```

Use command-line overrides to evaluate the static modes:

```bash
EVALUATION.inference_mode=wo
EVALUATION.inference_mode=w
```

There is no dynamic Gate in this version.
