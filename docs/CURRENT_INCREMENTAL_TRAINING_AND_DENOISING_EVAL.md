# FastWAM Incremental 训练与去噪步数评测进展

> 状态快照：2026-08-10（UTC）  
> 仓库：`/root/feihong/FAST_WAM_github`  
> 分支 / 提交：`main` / `bf9262d`  
> 说明：本文同时核对了仓库代码、未提交工作区、训练日志、checkpoint 和评测 `summary.json`。当前机器上没有正在运行的 FastWAM 训练或评测进程；下文的“进行中”指研发或实验链路尚未收尾，不表示此刻仍有作业运行。

## 1. 一页结论

当前工作围绕同一个问题展开：**action 推理时是否应使用生成中的 future-video，以及有/无 video 的 action 学习应共享一个 ActionDiT，还是使用两个 ActionDiT。**

已经建立两种 incremental 模型：

| 变体 | 模型 | 核心设计 |
|---|---|---|
| Unified Shared（UniShare） | `FastWAMUnifiedShared` | 共享一个 video DiT 和一个 ActionDiT；每个 batch 同时训练 `wo-video` 与 `w-video` 两种 mask，两项 action loss 更新同一个 ActionDiT。 |
| Unified TwoAction | `FastWAMUnifiedTwoAction` | 共享一个 video DiT，但 `wo-video` 和 `w-video` 各自使用一个 ActionDiT；两个 action 分支都更新共享 video DiT。 |

这里的 `w-video` 是 **Joint-style video/action 同步去噪**，不是先完整生成视频再做 action 的 IDM 两阶段推理。当前没有动态 gate，eval 通过静态 `wo`、`w` 或 UniShare 专用的 `prefix` 模式进行切换。

当前总体状态：

- LIBERO：两种模型的 10-epoch 全量训练和四组 `Shared/TwoAction × wo/w` eval 已完成；UniShare 的原始 scheduler N=0…10 sweep 已完成；action scheduler shift=1.0 的重训及第二轮 N=0…10 sweep 也已完成。
- RoboTwin 2.0：两种模型的 5-epoch 全量训练和四组 `Shared/TwoAction × wo/w` eval 已完成；UniShare N-step sweep 已完成 N=0…5，N=6…10 尚未产生 summary。
- RoboTwin 1/5：确定性、task-balanced 的 1/5 数据链路已经开发；UniShare 已完成 46,890 steps，TwoAction 尚未发现正式 checkpoint，四组 1/5 eval 因而尚未启动完整闭环。
- 当前最重要的实验事实：LIBERO 中 `w` 有小幅正收益；RoboTwin 中 `w` 反而下降，且 UniShare 降幅更明显。已有 N-step 结果均没有显示“更多 joint prefix 一定更好”的单调关系。

## 2. 两条 benchmark 训练链路

### 2.1 共同训练设置

正式 JiHe 脚本均使用 8×H100、ZeRO-1、有效 global batch 128，并将输出写到：

```text
/root/feihong/FastWAM/formal_runs/FAST_WAM_github
```

| Benchmark | 输入 | 每卡 batch | 梯度累积 | Epochs | 训练数据 |
|---|---:|---:|---:|---:|---|
| LIBERO | 2 cameras, 224 | 8 | 2 | 10 | 4 suites 的预处理 LeRobot 数据 |
| RoboTwin 2.0 | 3 cameras, 384 | 4 | 4 | 5 | 50 tasks、27,500 episodes、6,075,103 frames |

训练前置包括 Wan2.2 权重、预处理的 ActionDiT backbone、dataset stats 和完整 text embedding cache。RoboTwin instruction 表较大，因此有独立的 8-GPU text-cache 预计算入口。

### 2.2 LIBERO 全量训练

| 模型 | Run | 最终状态 |
|---|---|---|
| Unified Shared | `libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20` | 完成，`21,700/21,700` steps，存在 `latest.pt`。 |
| Unified TwoAction | `libero_unified_two_action_2cam224_1e-4/2026-07-01_00-50-11` | 完成，`21,700/21,700` steps，存在 `latest.pt`。 |
| Unified Shared, action shift=1 | `libero_unified_shared_action_shift1_2cam224_1e-4/2026-08-03_22-44-39` | 完成，存在 `latest.pt`，随后完成 N=0…10 sweep。 |

原始两模型的正式 eval 每个设置覆盖 40 tasks × 50 episodes = 2,000 episodes：

| 模型 | 推理模式 | 总成功率 | 相对同模型 `wo` |
|---|---|---:|---:|
| Shared | `wo` | 97.85% | — |
| Shared | `w` | **98.50%** | +0.65 pp |
| TwoAction | `wo` | 97.30% | — |
| TwoAction | `w` | 98.40% | +1.10 pp |

结论：LIBERO 上 video-conditioned joint inference 有小幅收益；Shared 的总体最好结果为 98.50%。TwoAction 的 `w-wo` 差值更大，但绝对值没有超过 Shared+w。

### 2.3 RoboTwin 2.0 全量训练

| 模型 | Run | 最终状态 |
|---|---|---|
| Unified Shared | `robotwin_unified_shared_3cam_384_1e-4/2026-07-01_00-51-30` | 多次 resume 后完成，`234,830/234,830` steps，存在 `latest.pt`。 |
| Unified TwoAction | `robotwin_unified_two_action_3cam_384_1e-4/2026-07-01_00-57-01` | 多次 resume 后完成，`234,830/234,830` steps，存在 `latest.pt`。 |

正式 eval 覆盖 50 tasks，每个 task 分 clean/randomized 两个 phase，各 100 episodes。下表“总体”是 clean 与 randomized task mean 的平均：

| 模型 | 模式 | Clean | Randomized | 总体 | 相对同模型 `wo` |
|---|---|---:|---:|---:|---:|
| Shared | `wo` | 93.70% | 92.60% | **93.15%** | — |
| Shared | `w` | 91.32% | 90.80% | 91.06% | -2.09 pp |
| TwoAction | `wo` | 92.14% | 92.22% | 92.18% | — |
| TwoAction | `w` | 91.88% | 91.26% | 91.57% | -0.61 pp |

结论与 LIBERO 相反：RoboTwin 上加入 joint future-video 会产生负迁移。TwoAction 将下降从 2.09 pp 缓解到 0.61 pp，但没有稳定优于 Shared+wo。已有分析还显示 Shared+w 的额外失败高度集中在少数任务，说明问题具有明显的 task dependency，而不是均匀退化。

## 3. RoboTwin 1/5 子集实验

这条链路用于判断 RoboTwin 负迁移与训练规模/数据量的关系，同时降低迭代成本。它不复制或改写原数据，只向 dataset 传入固定 episode indices。

子集定义：

- 50 个 task，每个 task 原始有 50 clean + 500 randomized episodes。
- 固定 seed 42，每个 task 选择 10 clean + 100 randomized。
- 共 5,500 episodes，随后按 99%/1% 切成 5,445 train + 55 val。
- 对应 1,200,269 train frames + 12,626 val frames。
- normalization 继续使用全量 RoboTwin 的 `dataset_stats.json`，使变量尽量只剩训练数据量。
- 每个 run 保存 `subset_manifest.json`、episode 列表及 SHA-256；Shared/TwoAction 使用相同索引。

训练设置保持 8×H100、global batch 128、5 epochs；每 epoch 9,378 optimizer steps，总计 46,890 steps。

当前状态：

| 项目 | 状态 |
|---|---|
| Episode selector、配置、单测和独立 launcher | 已开发。 |
| Unified Shared 1/5 | 已完成；run `2026-08-04_18-30-38` 到达 `46,890/46,890`，存在 `latest.pt`。 |
| Unified TwoAction 1/5 | 尚未发现 run/checkpoint，待训练。 |
| 1/5 四组 eval | 独立入口已开发；需等待 TwoAction 完成后运行 Shared/TwoAction × wo/w。 |

早期启动曾因 DiffSynth 无法识别空的 `wan_video_vae` 文件失败，后续有效 run 已解决并完整训练结束。当前模型配置将 `redirect_common_files=false`，相关启动脚本也增加了模型资产、数据、text cache、输出盘和环境的 preflight 检查。

## 4. 去噪步数（N-step prefix）eval

### 4.1 当前实现测量的到底是什么

UniShare 新增 `inference_mode=prefix` 和 `video_prefix_steps=N`。当总 solver steps `T=10` 时：

```text
N=0 : wo wo wo wo wo wo wo wo wo wo
N=1 : w  wo wo wo wo wo wo wo wo wo
N=5 : w  w  w  w  w  wo wo wo wo wo
N=9 : w  w  w  w  w  w  w  w  w  wo
N=10: w  w  w  w  w  w  w  w  w  w
```

因此 N 表示：**每次 action chunk sampling 的前 N 次 model call 使用 joint video-action vector field，后 T-N 次改为 first-frame-only action denoising。**

它不是环境 timestep、action 数或 video frame 数，也不能直接解释为“action 使用了去噪 N 步后的视频”。每个 joint iteration 中 action 与 video 同时读取当前 latent、同时预测、再同时 update；切换到 `wo` 后，已经部分去噪的 future-video latent 被丢弃，只保留前 N 次 joint update 对 action latent 轨迹造成的影响。

更准确的实验名应是 `joint_denoise_prefix_steps`，而不是 `N-step video quality`。

### 4.2 N 与实际去噪量并不线性

原始设置 `T=10, shift=5` 时，前 N 步覆盖的累计 `|Δsigma|` 为：

| N | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 累计路径 | 0% | 2.17% | 4.76% | 7.89% | 11.76% | 16.67% | 23.08% | 31.82% | 44.44% | 64.29% | 100% |

最后一步单独占 35.71% 的 sigma 路径。因此 N=5 不等于 50% video contribution，N=9→10 也不是普通的“再加 10%”。这正是补做 action scheduler `shift=1.0` 实验的动机之一：让 action solver 的步长更均匀，减少原 shift=5 下末步权重过大的混淆。

### 4.3 LIBERO：原始 scheduler sweep（已完成）

设置：UniShare、10 inference steps、40 tasks、每 task 50 episodes、每个 N 共 2,000 episodes；N=0…10 全部完成且 invalid episodes 为 0。

| N | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Success rate | 97.85% | 97.60% | 98.00% | 97.50% | 97.80% | 97.80% | 97.75% | 97.60% | 97.95% | 97.65% | **98.50%** |

结果没有呈现随 N 单调上升；N=10 最好，但 N=1…9 基本在 97.5%–98.0% 间波动。custom endpoint sanity 在 LIBERO-10 上验证：custom N=0 与 canonical `wo` 都是 477/500，custom N=10 与 canonical `w` 都是 490/500，且逐 episode 成败列表一致。

### 4.4 LIBERO：action scheduler shift=1 sweep（已完成）

这次先用 `action train_shift=1.0 / infer_shift=1.0` 重训 UniShare；video scheduler 仍保持 shift=5.0。随后以相同 N=0…10、每个 N 2,000 episodes 的设置重跑。

| N | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Success rate | 97.35% | 97.25% | 97.05% | 96.95% | 97.00% | 96.95% | 96.95% | 96.95% | 96.70% | 97.10% | **98.20%** |

这一轮仍未出现单调收益。中间 prefix 普遍不优于 N=0，N=10 再次明显抬升；但其最佳值 98.20% 也低于原始 scheduler 的 98.50%。因此，仅把 action scheduler 改为更均匀的 shift=1 并没有消除 endpoint-heavy 的表现形态，也没有提高总体最好成绩。

### 4.5 RoboTwin：UniShare prefix sweep（部分完成）

入口使用全量 UniShare checkpoint（原静态 `wo` 为 93.15%），每个 N 评测 50 tasks × clean/randomized × 100 episodes，10 inference steps，replan 24。

截至本快照，仅 N=0…5 有完整 `summary.json`：

| N | Clean | Randomized | 总体 |
|---:|---:|---:|---:|
| 0 | 92.94% | 92.52% | 92.73% |
| 1 | 93.08% | 92.30% | 92.69% |
| 2 | 93.46% | 92.60% | **93.03%** |
| 3 | 92.66% | 92.60% | 92.63% |
| 4 | 93.44% | 92.60% | 93.02% |
| 5 | 93.08% | 92.26% | 92.67% |

阶段结论：N=0…5 仍然没有单调趋势，变化范围只有约 0.40 pp；目前最好是 N=2 的 93.03%，仍略低于历史 canonical Shared+wo 的 93.15%。N=6…10 缺失，尤其缺少可与 Shared+w 91.06% 对照的 N=10 endpoint，因此该 sweep 暂时不能下最终结论。

## 5. 当前代码开发内容

当前工作区相对 `bf9262d` 有大规模未提交修改，主要集中在以下几组：

1. Prefix inference：UniShare 新增 joint-prefix + cached action-only suffix；TwoAction 对 prefix 明确报 `NotImplementedError`，因为它还需要定义何时切换 action head。
2. LIBERO eval：传递 `video_prefix_steps`、写入结果元数据、兼容本地 LIBERO 路径，并增加 invalid episode retry/black-screen 相关稳定性处理。
3. RoboTwin eval：policy 部署链路支持 `prefix`；manager 支持显式 GPU ID、错峰启动，降低 SAPIEN/Vulkan 多 worker 同时初始化失败。
4. 训练可靠性：launcher 增加环境、资产、数据、text cache、输出目录和 W&B 的 preflight；关闭 `redirect_common_files`，避免共享文件重定向导致加载错误。
5. 数据子集：新增通用 episode selector、RoboTwin 1/5 配置、manifest 和 selector 单测。
6. 实验入口：新增 LIBERO action-shift=1 重训/sweep、RoboTwin prefix sweep、RoboTwin 1/5 训练/eval 与 resume 脚本。

## 6. 已知限制与风险

- `force_custom_prefix` 当前被 evaluator 和模型内部强制设为 true，配置项本身没有实际控制力，属于误导性 API。
- LIBERO 中若同时设置 `visualize_future_video=true` 和 `inference_mode=prefix`，当前调用优先级会走完整 `infer_joint`，可能静默绕过 prefix；正式 sweep 使用 false，因此已有 sweep 未受影响。
- Prefix 只在 UniShare 实现；TwoAction 尚无 head-switching prefix 定义。
- 已保存结果没有完整绑定 Git SHA、dirty diff 或源码快照；尤其原始 LIBERO sweep 横跨工作区修改期，严格复现能力有限。
- 固定 diffusion seed 有利于不同 N 的配对比较，但没有覆盖 sampling seed 方差；正式结论还需要多 seed。
- 当前 N 横轴既不线性对应 sigma 路径，也不代表 partial video quality；论文或汇报中应避免过度解释。
- RoboTwin prefix N=0 与历史 canonical `wo` 存在 0.42 pp 差异，需进一步确认是重复 rollout 随机性、环境差异还是 custom endpoint 数值差异。

## 7. 建议的收尾顺序

1. 完成 RoboTwin UniShare N=6…10，并首先核对 custom N=0/N=10 与 canonical `wo/w` 的 tensor-level 和 rollout-level 等价性。
2. 启动 RoboTwin 1/5 TwoAction；完成后运行 1/5 的四组 `Shared/TwoAction × wo/w` eval，与全量结果做同协议对照。
3. 修复 `force_custom_prefix` 和 `visualize_future_video` 两个确定问题，并加入 route-spy、N=0/N=T `allclose`、KV cache 等价性测试。
4. 冻结正式实验版本：保存 Git SHA、dirty patch、完整 Hydra config、结果 schema version 和随机种子。
5. 将横轴改为 sigma cutoff 或累计 `|Δsigma|`，并补充 `wo early → w late`、single-step ablation 和多 diffusion seeds。
6. 如果研究目标真的是“视频逐渐变清晰后是否帮助 action”，需另做 partial-video alignment：先独立去噪 video N 步，冻结该 partial video，再让 action 显式 attend 它。当前 prefix 实现不能回答这个问题。

## 8. 关键入口与产物

代码与说明：

- `docs/INCREMENTAL_TRAINING.md`：两种 incremental 模型及训练方式。
- `docs/LIBERO_N_STEP_EVAL_AUDIT.md`：prefix 语义与代码审计。
- `docs/ROBOTWIN_1OF5_TRAINING.md`：1/5 数据选择与训练设计。
- `scripts/jihe/train_*_8xh100.sh`：四个全量正式训练入口。
- `scripts/jihe/train_robotwin_*_1of5_8xh100.sh`：RoboTwin 1/5 训练入口。
- `scripts/jihe/eval_libero_unified_shared_action_shift1_prefix_8xh100.sh`：LIBERO shift=1 N-sweep。
- `scripts/jihe/eval_robotwin_unified_shared_prefix_8xh100.sh`：RoboTwin N-sweep。

主要结果目录：

```text
/root/feihong/FastWAM/evaluate_results/libero_incremental_1gpu_local/local_1gpu_2026-07-03_09-47-32
/root/feihong/FastWAM/evaluate_results/libero_prefix_shared_T10/prefix_shared_T10_20260710_2218
/root/feihong/FastWAM/evaluate_results/libero_prefix_shared_action_shift1_T10_8xh100/20260805_150828
/root/feihong/FAST_WAM_github/evaluate_results/robotwin/latest
```

