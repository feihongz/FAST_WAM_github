# JiHe launch scripts

## LIBERO Stage 2 label smoke (1xH100)

Before launching the formal 64-shard label campaign, run the exact one-sample
E0/E10 generation and resume acceptance workflow:

```bash
bash scripts/jihe/run_libero_stage2_label_smoke_1xh100.sh
```

The launcher uses the frozen LIBERO step-30,000 Adapter, two paired seeds and
ten solver steps for both N=0 and N=10. It generates one deterministic real
sample, reruns the same immutable output directory to prove strict resume, and
writes a verification receipt. Its 1,048,576-shard singleton contract is
smoke-only and explicitly cannot be merged into the formal 64-shard labels.

## LIBERO Stage 2 formal labels (8xH100)

After the singleton smoke passes, start the formal 64-shard label campaign on
one JiHe node with eight H100s:

```bash
bash scripts/jihe/run_libero_stage2_labels_8xh100.sh
```

The launcher fixes the final step-30,000 Adapter and every base/data/VAE/stats
identity, then runs one eight-rank `torchrun` job. Each rank receives 8 stable
shards and all ranks publish into one persistent label directory. The default
run ID is `formal_<git-short-sha>`, so rerunning the same command at the same
commit strictly validates and resumes existing chunks. Set `RUN_ID` only when
you intentionally need a different formal job. To remap the eight visible
devices, set `FASTWAM_CUDA_VISIBLE_DEVICES` to exactly eight comma-separated
logical indices or GPU UUIDs.

The retired `run_libero_stage2_labels_4xh100.sh` path is retained only as a
compatibility entrypoint for already copied commands; it prints a notice and
forwards to this same strict eight-GPU launcher.

The formal launcher pins Ubuntu FFmpeg to
`7:4.4.2-0ubuntu0.22.04.1`. Its immutable numerical-runtime identity also
records the exact libav versions loaded by TorchCodec and the node-wide NVIDIA
driver release, without GPU indices or UUIDs. A resumed container with a
different decoder or driver fails before it can append chunks.

The frozen plan contains 273,465 samples, 64 shards and 4,307 chunks. The direct
label phase of the accepted one-sample smoke took about 7.8 seconds; that
projects to roughly 3.1 ideal days on eight H100s. Allow 3--5 days for
multi-rank I/O and long-run variance. After the one-time model startup, the
first 64-sample chunk on each rank should provide a better estimate in about
8--12 minutes (roughly 18--25 minutes from initial launch, including startup).
This projection is provisional until that first formal chunk
completes. The launcher reports progress every five minutes and safely resumes
at completed chunk boundaries. An exclusive non-blocking job lock rejects
accidentally overlapping launchers that target the same output directory.

`generation_success.json` is written only after rebuilding the canonical plan
and strictly validating every chunk. The marker also binds a deterministic
ordered inventory SHA over every relative chunk path, chunk SHA and row count.
It records `merge_completed=false`; the separate strict merge step must still
pass before Gate training starts.

## LIBERO Stage 3 final health smoke (1xH100)

After freezing the LIBERO step-30,000 Adapter, run exactly one tiny `w`-path
load/execute health check:

```bash
bash scripts/jihe/eval_libero_stage3_final_health_1xh100.sh
```

The launcher locks the final Adapter path/SHA, training-contract SHA, and
global step. It runs one closed-loop trial with two solver steps and does not
produce a formal benchmark success-rate result.

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
