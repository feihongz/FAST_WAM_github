# FastWAM

**Fast-WAM: Do World Action Models Need Test-time Future Imagination?** 的训练与评估代码仓库。

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836.svg)](./README_zh.md)

[![arXiv](https://img.shields.io/badge/arXiv-2603.16666-b31b1b.svg)](https://arxiv.org/abs/2603.16666)
[![Project Page](https://img.shields.io/badge/Project_Page-Fast--WAM-2ea44f.svg)](https://yuantianyuan01.github.io/FastWAM/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-f7c843)](https://huggingface.co/yuanty/fastwam)
[![Hugging Face Dataset - LIBERO](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20LIBERO-f7c843)](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)
[![Hugging Face Dataset - RoboTwin](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20RoboTwin-f7c843)](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)

本仓库基于 Wan2.2-TI2V-5B 构建机器人操作策略，支持在 LIBERO 和 RoboTwin 上训练、评估 Fast-WAM，以及两个 incremental unified 变体：

- `FastWAM`：主链路。训练 video denoising 和 action denoising，但 action 推理时只依赖当前观测，不需要先想象未来视频。
- `FastWAMUnifiedShared`：incremental 变体一。共享一个 video DiT 和一个 ActionDiT，同时训练 `wo-video` 与 `w-video` 两种 action loss。
- `FastWAMUnifiedTwoAction`：incremental 变体二。共享一个 video DiT，但为 `wo-video` 与 `w-video` 分别使用一个 ActionDiT。

## 目录

- [仓库结构](#仓库结构)
- [核心思路](#核心思路)
- [FAST WAM 主链路](#fast-wam-主链路)
- [两个 incremental 链路](#两个-incremental-链路)
- [配置入口](#配置入口)
- [环境安装](#环境安装)
- [模型准备](#模型准备)
- [数据准备](#数据准备)
- [训练](#训练)
- [评估](#评估)
- [常见注意点](#常见注意点)
- [致谢](#致谢)
- [引用](#引用)

## 仓库结构

```text
FAST_WAM_github/
├── configs/
│   ├── data/                    # LIBERO / RoboTwin 数据配置
│   ├── model/                   # FastWAM、Joint、IDM、Unified 模型配置
│   ├── task/                    # 训练 task 配置
│   ├── sim_libero.yaml          # LIBERO 评估配置
│   └── sim_robotwin.yaml        # RoboTwin 评估配置
├── scripts/
│   ├── train.py                 # Hydra + Accelerate 训练主入口
│   ├── train_zero1.sh           # DeepSpeed ZeRO-1 通用训练入口
│   ├── train_zero2.sh           # DeepSpeed ZeRO-2 通用训练入口
│   ├── preprocess_action_dit_backbone.py
│   ├── precompute_text_embeds.py
│   └── jihe/                    # 8xH100 正式实验封装脚本
├── src/fastwam/
│   ├── models/wan22/            # FastWAM 核心模型实现
│   ├── datasets/                # LeRobot 数据读取、transform、processor
│   ├── trainer.py               # 训练器、保存/恢复 checkpoint
│   └── runtime.py               # Hydra model factory
├── experiments/
│   ├── libero/                  # LIBERO policy/eval manager
│   └── robotwin/                # RoboTwin policy/eval manager
├── docs/
│   └── INCREMENTAL_TRAINING.md  # incremental 训练补充说明
├── third_party/RoboTwin/        # RoboTwin 评估代码适配
├── checkpoints/                 # Wan / ActionDiT / release checkpoint
├── data/                        # 数据目录
├── runs/                        # 默认训练输出
└── evaluate_results/            # 默认评估输出
```

## 核心思路

Fast-WAM 是一个 World Action Model。它保留 video world model 的训练信号，但目标不是在测试时先生成未来视频再做动作，而是让 action expert 在训练中吸收 world model 的表征和时序结构，推理时直接从当前观测和语言输出 action chunk。

主模型由几部分组成：

- `video_expert`：来自 Wan2.2 的 video DiT，负责视频 latent denoising。
- `action_expert`：ActionDiT，负责连续 action chunk denoising。
- `MoT`：Mixture-of-Transformers 容器，把 video tokens 和 action tokens 放在同一个 transformer 层级里，用 attention mask 控制两类 token 的可见性。
- `VAE`：把训练视频编码到 Wan latent 空间，评估时也用于首帧 conditioning。
- `T5 text embedding cache`：训练时默认使用预计算文本 embedding，避免每步重复跑文本 encoder。
- `proprio_encoder`：如果数据 processor 提供 proprio state，则把 proprio 编码成额外 context token。

训练时，视频和动作分别走 flow matching scheduler：

```text
video frames -> VAE -> video latents -> add noise -> video_expert/MoT -> video denoising loss
actions      -> action tensor      -> add noise -> action_expert/MoT -> action denoising loss
```

总损失默认是：

```text
loss = lambda_video * loss_video + lambda_action * loss_action
```

## FAST WAM 主链路

主链路实现位于 [`src/fastwam/models/wan22/fastwam.py`](./src/fastwam/models/wan22/fastwam.py)，配置入口是 [`configs/model/fastwam.yaml`](./configs/model/fastwam.yaml)。

Fast-WAM 的关键点是 `wo-video` action 学习：action tokens 不是完全看未来视频，而是通过 mask 只看首帧 video tokens 和 action 自身 tokens。代码里 `_build_mot_attention_mask` 的语义是：

```text
video -> video: 走 video_expert 自己的 video attention mask
action -> action: action tokens 彼此可见
action -> video: 只允许看 first-frame video tokens
```

因此推理时 `infer_action` 不需要先生成未来视频。它只编码当前输入图像为首帧 latent，再对 action latent 做迭代 denoising，得到一个 action chunk。这个路径对应论文和配置里的 `FastWAM` / `uncond` 任务。

主链路 task 示例：

```text
libero_uncond_2cam224_1e-4
robotwin_uncond_3cam_384_1e-4
```

主链路适合回答这个问题：如果 world/action 共同训练，但测试时不使用未来视频想象，action policy 能否仍然获得足够强的 world-aware 表征。

## 两个 incremental 链路

本仓库额外提供两个 incremental unified 变体，用来比较 `wo-video` 和 `w-video` 两种 action 学习方式应该共享一个 ActionDiT，还是拆成两个 ActionDiT。

这里的 `w-video` 是 Joint-style video/action denoising：action tokens 可以 attend 到完整 video tokens。它不是 IDM-style 的 teacher-forcing video-then-action 链路。IDM 变体单独在 [`fastwam_idm.py`](./src/fastwam/models/wan22/fastwam_idm.py) 中实现。

### 1. Unified Shared

实现文件：

- [`src/fastwam/models/wan22/fastwam_unified_shared.py`](./src/fastwam/models/wan22/fastwam_unified_shared.py)
- [`configs/model/fastwam_unified_shared.yaml`](./configs/model/fastwam_unified_shared.yaml)

结构：

```text
shared video_expert
shared action_expert
shared MoT(video, action)
```

每个 batch 会构造两张 attention mask：

```text
mask_wo: Fast-WAM mask
  action 只看 first-frame video tokens + action tokens

mask_w: Joint mask
  action 看 full video tokens + action tokens
```

训练时同一个 `action_expert` 跑两次 action denoising：

```text
tokens_out_wo -> pred_action_wo -> loss_action_wo
tokens_out_w  -> pred_action_w  -> loss_action_w
```

混合 action loss：

```text
loss_action = alpha_wo * loss_action_wo + alpha_w * loss_action_w
loss_total  = lambda_video * loss_video + lambda_action * loss_action
```

默认配置里 `alpha_wo=0.5`，`alpha_w=0.5`。这个链路用于测试：同一个 ActionDiT 同时学习 `wo-video` 和 `w-video` 是否会互相促进，还是相互干扰。

评估时通过静态模式选择路径：

```bash
EVALUATION.inference_mode=wo  # Fast-WAM style action-only
EVALUATION.inference_mode=w   # Joint-style action with full video path
```

### 2. Unified TwoAction

实现文件：

- [`src/fastwam/models/wan22/fastwam_unified_two_action.py`](./src/fastwam/models/wan22/fastwam_unified_two_action.py)
- [`configs/model/fastwam_unified_two_action.yaml`](./configs/model/fastwam_unified_two_action.yaml)

结构：

```text
shared video_expert
action_expert_wo
action_expert_w
mot_wo = MoT(shared video_expert, action_expert_wo)
mot_w  = MoT(shared video_expert, action_expert_w)
```

训练时同一个 batch 仍然同时跑 `wo` 和 `w` 两条 action loss，但 action head 分离：

```text
mask_wo + action_expert_wo -> loss_action_wo
mask_w  + action_expert_w  -> loss_action_w
```

video DiT 是共享的，两个分支都会通过共享 video tokens 影响 video/world 表征；ActionDiT 则分开，避免 `wo-video` 与 `w-video` 的 action denoising 目标直接抢同一组 action 参数。

这个链路用于测试：如果 `wo-video` 和 `w-video` 的 action 条件差异较大，把 action expert 拆开是否比共享更稳。

评估模式和 Unified Shared 一样：

```bash
EVALUATION.inference_mode=wo
EVALUATION.inference_mode=w
```

当前版本没有 dynamic gate，不会在推理时自动选择 `wo` 或 `w`。

## 配置入口

常用 task：

| 场景 | task | model config | 模型类 |
| --- | --- | --- | --- |
| LIBERO FastWAM | `libero_uncond_2cam224_1e-4` | `fastwam` | `FastWAM` |
| LIBERO Unified Shared | `libero_unified_shared_2cam224_1e-4` | `fastwam_unified_shared` | `FastWAMUnifiedShared` |
| LIBERO Unified TwoAction | `libero_unified_two_action_2cam224_1e-4` | `fastwam_unified_two_action` | `FastWAMUnifiedTwoAction` |
| RoboTwin FastWAM | `robotwin_uncond_3cam_384_1e-4` | `fastwam` | `FastWAM` |
| RoboTwin Unified Shared | `robotwin_unified_shared_3cam_384_1e-4` | `fastwam_unified_shared` | `FastWAMUnifiedShared` |
| RoboTwin Unified TwoAction | `robotwin_unified_two_action_3cam_384_1e-4` | `fastwam_unified_two_action` | `FastWAMUnifiedTwoAction` |

相关对照变体：

| task/model | 含义 |
| --- | --- |
| `fastwam_joint` | action 可以 attend 到完整 video tokens 的 Joint-style 变体 |
| `fastwam_idm` | teacher-forcing video conditioning 的 IDM-style 变体 |

Hydra 会通过 `configs/task/*.yaml` 选择 `configs/data/*.yaml` 和 `configs/model/*.yaml`，再由 [`src/fastwam/runtime.py`](./src/fastwam/runtime.py) 创建对应模型。

## 环境安装

推荐 Python 3.10 和 CUDA 12.8 对应的 PyTorch：

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam

pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

如果在本机已有专用虚拟环境，也可以直接设置：

```bash
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH:-}"
```

## 模型准备

训练和推理都需要 Wan2.2 组件以及 ActionDiT backbone。默认模型目录是 `./checkpoints`，也可以用 `DIFFSYNTH_MODEL_BASE_PATH` 指向其他目录：

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

预生成 ActionDiT backbone：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

该文件会被 `configs/model/*.yaml` 中的 `action_dit_pretrained_path` 引用。

## 数据准备

### LIBERO

预处理数据发布在：

- https://huggingface.co/datasets/yuanty/LIBERO-fastwam

下载 4 个压缩包后解压：

```bash
mkdir -p data/libero_mujoco3.3.2
cd data/libero_mujoco3.3.2

for f in *.tar.gz; do
  tar -xzf "$f"
done
```

期望目录：

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

### RoboTwin

预处理数据发布在：

- https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam

下载全部分卷后拼接解压：

```bash
mkdir -p data/robotwin2.0
cd data/robotwin2.0

cat robotwin2.0.tar.gz.part-* | tar -xzf -
```

期望目录：

```text
data/robotwin2.0/
└── robotwin2.0/
    ├── data/
    ├── meta/
    └── videos/
```

如果已有：

```text
data/robotwin2.0/dataset_stats.json
```

可以直接作为当前配置的 normalization stats。也可以首次训练时让程序重新生成。

## 训练

### 1. 预计算文本 embedding

训练默认使用预计算 T5 embedding cache：

```bash
# LIBERO FastWAM
python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4

# LIBERO incremental，两条 unified 配置可共用同一类 LIBERO text cache
python scripts/precompute_text_embeds.py task=libero_unified_shared_2cam224_1e-4 +overwrite=false

# RoboTwin FastWAM
python scripts/precompute_text_embeds.py task=robotwin_uncond_3cam_384_1e-4

# RoboTwin incremental
python scripts/precompute_text_embeds.py task=robotwin_unified_shared_3cam_384_1e-4 +overwrite=false
```

多卡预计算示例：

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/precompute_text_embeds.py task=libero_unified_shared_2cam224_1e-4 +overwrite=false
```

### 2. 通用训练入口

第一次训练某个新数据配置时，如果没有现成 stats，可以先把对应 `configs/data/*.yaml` 里的 `pretrained_norm_stats` 设为 `null`。训练会在 run 目录生成 `dataset_stats.json`，后续可把 `pretrained_norm_stats` 指向该文件。

FastWAM 主链路：

```bash
# LIBERO
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4

# RoboTwin
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4
```

Incremental Unified Shared：

```bash
# LIBERO
bash scripts/train_zero1.sh 8 task=libero_unified_shared_2cam224_1e-4

# RoboTwin
bash scripts/train_zero1.sh 8 task=robotwin_unified_shared_3cam_384_1e-4
```

Incremental Unified TwoAction：

```bash
# LIBERO
bash scripts/train_zero1.sh 8 task=libero_unified_two_action_2cam224_1e-4

# RoboTwin
bash scripts/train_zero1.sh 8 task=robotwin_unified_two_action_3cam_384_1e-4
```

### 3. 正式实验脚本

`scripts/jihe/` 下提供了 8xH100 封装脚本，适合当前集群路径和 W&B 配置：

```bash
bash scripts/jihe/train_libero_unified_shared_8xh100.sh
bash scripts/jihe/train_libero_unified_two_action_8xh100.sh
bash scripts/jihe/train_robotwin_unified_shared_8xh100.sh
bash scripts/jihe/train_robotwin_unified_two_action_8xh100.sh
```

这些脚本默认把大输出写到 NAS 路径，并设置 global batch、日志目录和 W&B 参数。细节见 [`scripts/jihe/README.md`](./scripts/jihe/README.md)。

### 4. Smoke run

incremental smoke 命令见：

```text
docs/INCREMENTAL_TRAINING.md
```

该文档包含 Unified Shared / Unified TwoAction 在 LIBERO 和 RoboTwin 上的短步数训练命令，以及 H100 80GB 上的显存经验。

## 评估

### Release 权重

release checkpoint 和对应 dataset stats 在：

- https://huggingface.co/yuanty/fastwam

下载示例：

```bash
pip install -U huggingface_hub

huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  robotwin_uncond_3cam_384.pt \
  robotwin_uncond_3cam_384_dataset_stats.json \
  --local-dir ./checkpoints/fastwam_release
```

### LIBERO 评估

先按 [LIBERO 官方仓库](https://github.com/Lifelong-Robot-Learning/LIBERO) 安装模拟环境，并保持 mujoco 版本和数据版本一致：

```bash
pip install mujoco==3.3.2
```

评估 release FastWAM：

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

评估自己训练的 checkpoint：

```bash
python experiments/libero/run_libero_manager.py \
  task={task_name} \
  ckpt={ckpt_path} \
  MULTIRUN.num_gpus=8
```

评估 incremental 静态模式：

```bash
# wo-video path
python experiments/libero/run_libero_manager.py \
  task=libero_unified_shared_2cam224_1e-4 \
  ckpt={ckpt_path} \
  EVALUATION.inference_mode=wo

# w-video path
python experiments/libero/run_libero_manager.py \
  task=libero_unified_shared_2cam224_1e-4 \
  ckpt={ckpt_path} \
  EVALUATION.inference_mode=w
```

`libero_unified_two_action_2cam224_1e-4` 同理。

### RoboTwin 评估

仓库已经包含 `third_party/RoboTwin` 评估适配代码，但仍需按 [RoboTwin 官方仓库](https://github.com/RoboTwin-Platform/RoboTwin) 完成环境安装和 assets 下载。

创建 policy 软链接：

```bash
ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" \
  "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"
```

评估 release FastWAM：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

评估 incremental：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_unified_two_action_3cam_384_1e-4 \
  ckpt={ckpt_path} \
  EVALUATION.inference_mode=wo
```

为了加速 RoboTwin 评估，默认配置 [`configs/sim_robotwin.yaml`](./configs/sim_robotwin.yaml) 打开了：

```yaml
EVALUATION:
  skip_get_obs_within_replan: true
```

这会在一次 replan 窗口内连续执行 action chunk 时跳过 RGB 渲染，评估更快，但保存视频看起来帧率较低。如果需要完整视频，可以覆盖为 `false`。

## 常见注意点

- `w-video` incremental 路径是 Joint-style full-video attention，不是 IDM teacher-forcing。
- unified 评估没有 dynamic gate，必须显式设置 `EVALUATION.inference_mode=wo` 或 `EVALUATION.inference_mode=w`。
- `FastWAMUnifiedShared` 共享 ActionDiT，适合观察两种 action supervision 是否可以共用容量。
- `FastWAMUnifiedTwoAction` 分离 ActionDiT，适合观察 `wo` 和 `w` 的 action 目标是否存在参数冲突。
- `model.mot_checkpoint_mixed_attn=true` 可以降低显存，但会增加计算开销；task 配置中默认可能关闭，smoke/正式脚本可按显存情况覆盖。
- RoboTwin 任务和 text cache 更大，完整预计算和训练时间明显高于 LIBERO。
- 大模型、数据和正式训练输出不要放系统盘，优先使用 NAS 或大容量数据盘。

## 致谢

RoboTwin 评估代码基于官方 [RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin) 适配。感谢 RoboTwin 团队公开代码和 assets。

## 引用

如果本仓库对你的研究有帮助，请引用：

```bibtex
@article{yuan2026fastwam,
  title={Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author={Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal={arXiv preprint arXiv:2603.16666},
  year={2026},
  url={https://arxiv.org/abs/2603.16666}
}
```
