# JiHe 8xH100 formal launch scripts

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
Outputs go to `/root/feihong/FastWAM/formal_runs/FAST_WAM_github`; logs go to `/root/feihong/FastWAM/formal_logs/FAST_WAM_github`.

W&B online logging follows the previous StarVLA stage2 launch style. All four scripts first source `WANDB_ENV_FILE` and default it to `/root/feihong/FastWAM/secrets/wandb.env`; they then default to entity `smap` and project `fast-wam-formal`. Explicit `WANDB_ENTITY` / `WANDB_PROJECT` values set before launch override those defaults.

To override the target explicitly:

```bash
export WANDB_ENV_FILE=/root/feihong/FastWAM/secrets/wandb.env
export WANDB_ENTITY=deduktive
export WANDB_PROJECT=fast-wam-formal
```

If you do not use `WANDB_ENV_FILE`, either export `WANDB_API_KEY` or store the key in the fallback NAS secret file:

```bash
mkdir -p /root/feihong/FastWAM/secrets
printf '%s' 'YOUR_WANDB_API_KEY' > /root/feihong/FastWAM/secrets/wandb_api_key
chmod 600 /root/feihong/FastWAM/secrets/wandb_api_key
```

RoboTwin full training requires full text cache first:

```bash
bash scripts/jihe/precompute_robotwin_text_cache_8xh100.sh
```

RoboTwin UniShare n-step sweep (uses the 93.15% Shared+wo checkpoint):

```bash
bash scripts/jihe/eval_robotwin_unified_shared_prefix_8xh100.sh
```

The script evaluates `video_prefix_steps=n` for `n=0..10`, one worker per GPU, and writes one `summary.json`/`summary.csv` per n under `evaluate_results/robotwin/latest/n{n}` plus launcher logs under `FastWAM/evaluate_logs`. Override `CKPT`, `RUN_DIR`, `N_START`, `N_END`, or `EVAL_NUM_EPISODES` when needed.
