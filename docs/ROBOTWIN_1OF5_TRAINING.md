# RoboTwin 2.0 1/5 子集训练

这是一条与原始 27,500-episode 训练配置分离的链路。它不复制、不删除、也不改写
RoboTwin 的 parquet、视频或 metadata，只在数据加载时传入确定性的 episode 索引。

## 子集组成

- 50 个任务，每个任务在源数据中连续存放 550 episodes。
- 每个任务的前 50 条为 clean，后 500 条为 randomized。
- 固定随机种子 42，从每个任务分别抽取 10 clean + 100 randomized。
- 共选择 5,500 episodes；再按原配置的 99%/1% 切分为 5,445 train 和 55 val。
- 实际为 1,200,269 train frames + 12,626 val frames，共 1,212,895 frames。
- 两个模型变体使用相同的 episode 索引。
- normalization 继续使用原始全量数据的 dataset_stats.json，使本实验只改变训练样本量。
- 每次启动都会在该 run 的输出目录保存 subset_manifest.json，记录完整的 selected/train/val
  episode ID 和文件 SHA-256，便于复现与审计。

## 启动

Unified Shared：

    bash scripts/jihe/train_robotwin_unified_shared_1of5_8xh100.sh

Unified TwoAction：

    bash scripts/jihe/train_robotwin_unified_two_action_1of5_8xh100.sh

脚本默认保持原训练设置：8×H100、有效 global batch 128、5 epochs、max_steps=null。
按当前数据长度计算为每个 epoch 9,378 optimizer steps，总计 46,890 steps，约为
全量训练的 1/5。

两个训练完成后，使用独立 eval 入口：

    bash scripts/jihe/eval_robotwin_incremental_1of5_8xh100.sh

该入口会分别发现两个 1/5 task 的最新完整 checkpoint，并继续执行原有的 Shared/TwoAction
以及有/无 video conditioning 的四组 RoboTwin 评测。

输出和日志默认写入单独的 robotwin_1of5 目录，不会与原训练运行混合。

注意：数据加载器按 frame 采样。每个任务虽然 episode 数相同，但较长 episode 的任务
仍然会贡献更多训练 samples。
