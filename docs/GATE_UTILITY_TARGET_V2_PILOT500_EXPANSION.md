# LIBERO Gate Utility Target V2：Pilot-500 扩展预注册与运行手册

## 1. 目的与边界

本阶段把已经独立验证的 100-state Target V2 扩展到 Pilot-500 的完整 500 个 demonstration states，为后续离线 Tiny-MLP 可学习性研究准备更大的训练标签池。

本阶段只构造标签，不训练 Gate，不修改 UniShare 的 `training_loss()`，不改变 evaluator 或 closed-loop policy，也不使用独立验证 seeds 47–50。完成 Pilot-500 扩展本身不代表 Gate 已经可以部署。

固定集合关系如下：

```text
Pilot-500 immutable order
  - exact Phase-2.5 / Target-V2 existing 100-state panel
  = exact remainder 400 states
```

400-state remainder 的 suite 计数固定为：

| suite | states |
|---|---:|
| libero_10 | 160 |
| libero_goal | 70 |
| libero_object | 98 |
| libero_spatial | 72 |
| 合计 | 400 |

## 2. 固定推理协议

Target V2 标签使用 seeds `[42, 43, 44, 45, 46]`。

- seed 42：从 immutable Pilot 原始记录复制，不重新推理。
- seeds 43–46：对 remainder 400 states 分别执行 paired N=0/N=10 inference。
- seeds 47–50：严格保留给独立验证，不得进入本扩展。
- 每个 replicate 的 N=0 与 N=10 必须共享 observation、instruction、proprio、GT action 和 paired diffusion seed。
- 路由固定为 prefix N=0 与 prefix N=10，`num_inference_steps=10`、`num_video_frames=9`、`force_custom_prefix=true`。

预期工作量：

```text
400 states × 5 seeds = 2,000 rows
400 reused seed-42 rows
1,600 newly inferred rows
0 errors
```

## 3. 外部输入锚点

正式运行必须显式提供并验证以下 SHA-256：

| artifact | SHA-256 |
|---|---|
| Pilot manifest | `bf28abb5d58168493017bdff59cebf76282fd36cd815af75a116e95af5f4829d` |
| Pilot records | `f310b546ec13f1526d74a484663933b7305a5b985b1d6e5098025b6848479994` |
| Phase-2.5 source manifest | `c7476d522f47f71df30fb96ebaba5d09f6dd7a0a83400456a79ab1146506d0b9` |
| Phase-2.5 source records | `57abaacb551b4d6094e09812212c2be8098c9d823e58fb1de71d4f40469d4fb8` |
| Phase-2.5 selection plan | `59f13375c815a556073f72cdff5243b73a19fc9dbec7bd58c4d419b4d72e90db` |
| Existing Target V2 manifest | `bc00fdf913b5ae257deb57f6d7101ac66b95c73f179578f194a1b9b53ad2bfc3` |
| Existing Target V2 targets | `918d33a324ffaf3103ff42e97771debb356a1c9af292bd87c92cea75cf70d023` |
| UniShare checkpoint | `17a5588cc2b8d162219c9daf818614f614ee4a7921933a4a26c5d678111330e9` |
| dataset stats | `30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638` |
| VAE | `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36` |

Collector 还会绑定 Pilot 中已经登记的 LIBERO dataset trees、text-embedding cache、dataset ranges、task tables 与 checkpoint load provenance。

## 4. Git 与封存完整性

正式运行必须满足以下条件：

1. 所有 tracked 文件相对 HEAD 都是 clean；staged 和 unstaged 修改都会被拒绝。
2. 每个 scientific source file 都必须被 Git 跟踪，并且工作区字节与 `HEAD:<path>` 完全一致。
3. 无关的 untracked 文件可以存在，但 untracked scientific source 会被拒绝。
4. 在生成 completion seal 前，Collector 会重新读取并哈希：Pilot manifest/records、Phase-2.5 manifest/records、existing Target V2 manifest/targets、checkpoint、dataset stats、VAE、所有 dataset trees、text cache 与 scientific source files。
5. 重新读取到的 component snapshot 会直接写入 `completion.json` 并计算独立摘要；封存后再重读一次，若期间发生变化则停止发布。

因此正式 GPU 运行只能在本 PR 的科学代码已经 review、commit 且 worktree tracked-clean 后开始。

## 5. 输出结构与恢复规则

Remainder source bundle：

```text
manifest.json
records.jsonl
errors.jsonl
completion.json
```

- Resume 主键为 `(source_index, replicate_index)`，同时检查唯一 `replicate_id`。
- 每条 row 有 canonical SHA-256；已有行会完整验证后复用。
- 未封存且无错误的 partial run 可以安全 resume。
- 存在非空 `errors.jsonl` 时必须使用新的输出目录，不允许继续。
- 已存在 `completion.json` 但仍有 pending cell 属于矛盾状态，必须 fail closed。
- 只有 exact 2,000-row grid、400 reused、1,600 inferred、0 errors 才能生成 completion seal。

随后发布两个不可变 Target bundle：

1. remainder-400 Target V2；
2. combined Pilot-500 Target V2，按 immutable Pilot-500 order 合并原 100 targets 与新 400 targets。

原 100-state target payload 不得改写；combined bundle 保留 component 与 source bindings。

## 6. 正式命令模板

```bash
PYTHONPATH=. /root/.venvs/fastwam/bin/python \
  experiments/libero/gate/collect_demo_utility_target_v2_pilot500.py \
  'data.train.dataset_dirs=[/root/feihong/FastWAM/datasets/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot,/root/feihong/FastWAM/datasets/libero_mujoco3.3.2/libero_object_no_noops_lerobot,/root/feihong/FastWAM/datasets/libero_mujoco3.3.2/libero_goal_no_noops_lerobot,/root/feihong/FastWAM/datasets/libero_mujoco3.3.2/libero_10_no_noops_lerobot]' \
  data.train.text_embedding_cache_dir=/root/feihong/FAST_WAM_github/data/text_embeds_cache/libero \
  ckpt="${CHECKPOINT_PATH}" \
  COLLECTOR.pilot_dir="${PILOT_DIR}" \
  COLLECTOR.phase25_dir="${PHASE25_SOURCE_DIR}" \
  COLLECTOR.existing_target_v2_dir="${TARGET_V2_100_DIR}" \
  COLLECTOR.dataset_stats_path="${DATASET_STATS_PATH}" \
  COLLECTOR.output_dir="${REMAINDER_SOURCE_DIR}" \
  COLLECTOR.remainder_target_v2_dir="${REMAINDER_TARGET_DIR}" \
  COLLECTOR.combined_target_v2_dir="${COMBINED_TARGET_500_DIR}" \
  COLLECTOR.expected_pilot_manifest_sha256=bf28abb5d58168493017bdff59cebf76282fd36cd815af75a116e95af5f4829d \
  COLLECTOR.expected_pilot_records_sha256=f310b546ec13f1526d74a484663933b7305a5b985b1d6e5098025b6848479994 \
  COLLECTOR.expected_phase25_manifest_sha256=c7476d522f47f71df30fb96ebaba5d09f6dd7a0a83400456a79ab1146506d0b9 \
  COLLECTOR.expected_phase25_records_sha256=57abaacb551b4d6094e09812212c2be8098c9d823e58fb1de71d4f40469d4fb8 \
  COLLECTOR.expected_phase25_selection_plan_sha256=59f13375c815a556073f72cdff5243b73a19fc9dbec7bd58c4d419b4d72e90db \
  COLLECTOR.expected_existing_target_v2_manifest_sha256=bc00fdf913b5ae257deb57f6d7101ac66b95c73f179578f194a1b9b53ad2bfc3 \
  COLLECTOR.expected_existing_target_v2_targets_sha256=918d33a324ffaf3103ff42e97771debb356a1c9af292bd87c92cea75cf70d023
```

上面的 `dataset_dirs` 顺序是冻结协议的一部分，四项必须使用绝对路径；text cache 同样必须显式使用绝对路径。路径内容仍会与 immutable Pilot 中登记的 tree hashes 比对，不能通过换路径改变数据。

## 7. 验收与 stop point

必须同时满足：

```text
selected_states = 400
expected_records = 2000
reused total = 400
inferred total = 1600
errors total = 0
remainder_target_states = 400
combined_target_states = 500
```

此外还要独立检查 source/target/completion hashes、seed grid、paired routes、`U=E0-E10`、cross-seed identity、suite counts、原 100 target payload 未变化，以及 completion 中 component snapshot 与真实输入一致。

验收通过后只能说明 Pilot-500 Target V2 数据集已可用于后续离线 Gate 可学习性研究；是否训练、是否进入 closed-loop eval，仍由独立的 Tiny-MLP protocol 与 validation 结果决定。
