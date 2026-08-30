# JiHe launch scripts

## Stage 3 endpoint pilot smoke (1xH100)

Stage 3 200-step pilot endpoint connectivity smokes use two independent
one-H100 jobs:

```bash
bash scripts/jihe/eval_libero_stage3_pilot_endpoint_1xh100.sh
bash scripts/jihe/eval_robotwin_stage3_pilot_endpoint_1xh100.sh
```

They run one closed-loop episode with `inference_mode=w` and strict
base/Adapter/data/VAE/stats identity checks. These are pilot connectivity
checks, not formal benchmark evaluations. See
`docs/STAGE3_ENDPOINT_EVAL.md`.

## Formal training (8xH100)

Four independent bash entrypoints:

```bash
bash scripts/jihe/train_libero_unified_shared_8xh100.sh
bash scripts/jihe/train_libero_unified_two_action_8xh100.sh
bash scripts/jihe/train_robotwin_unified_shared_8xh100.sh
bash scripts/jihe/train_robotwin_unified_two_action_8xh100.sh
```

They explicitly use the FAST-WAM paired datasets under `data/libero_mujoco3.3.2` and `data/robotwin2.0/robotwin2.0`.
Default effective global batch is 128 on 8 GPUs. LIBERO uses per-GPU batch 8 with gradient accumulation 2; RoboTwin uses per-GPU batch 4 with gradient accumulation 4.
LIBERO runs 10 epochs; RoboTwin runs 5 epochs.
Outputs go to `/root/nas/temp_nas/FastWAM/formal_runs/FAST_WAM_github`; logs go to `/root/nas/temp_nas/FastWAM/formal_logs/FAST_WAM_github`.

W&B online logging follows the previous StarVLA stage2 launch style. All four scripts first source `WANDB_ENV_FILE` and default it to `/root/nas/zian/.secrets/wandb.env`; they then default to entity `smap` and project `fast-wam-formal`. Explicit `WANDB_ENTITY` / `WANDB_PROJECT` values set before launch override those defaults.

To override the target explicitly:

```bash
export WANDB_ENV_FILE=/root/nas/zian/.secrets/wandb.env
export WANDB_ENTITY=deduktive
export WANDB_PROJECT=fast-wam-formal
```

If you do not use `WANDB_ENV_FILE`, either export `WANDB_API_KEY` or store the key in the fallback NAS secret file:

```bash
mkdir -p /root/nas/temp_nas/FastWAM/secrets
printf '%s' 'YOUR_WANDB_API_KEY' > /root/nas/temp_nas/FastWAM/secrets/wandb_api_key
chmod 600 /root/nas/temp_nas/FastWAM/secrets/wandb_api_key
```

RoboTwin full training requires full text cache first:

```bash
bash scripts/jihe/precompute_robotwin_text_cache_8xh100.sh
```
