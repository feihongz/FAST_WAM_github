# Gate V1：Phase 0 / Phase 1 开发说明

## 1. 本阶段目标

本阶段只打通 LIBERO closed-loop eval 中动态选择 `N=0` 或
`N=num_inference_steps` 的最小闭环：

```text
一次 action chunk replan
        ↓
GateRouter.select(...)
        ↓
selected_n ∈ {0, num_inference_steps}
        ↓
model.infer_action_mode(
    inference_mode="prefix",
    video_prefix_steps=selected_n,
)
```

Gate 位于 evaluator，不修改 `FastWAMUnifiedShared`、训练 loss、checkpoint
结构或已有静态 prefix baseline。每当 `pending_actions` 消耗完，evaluator
重新预测 action chunk，Gate 也恰好决策一次。

## 2. 代码结构

```text
experiments/libero/
├── eval_libero_single.py       # replan 路由、调用模型、记录 closed-loop 指标
└── gate/
    ├── __init__.py
    └── router.py               # fixed/random/learned Router 接口

configs/sim_libero.yaml         # EVALUATION.gate 配置
tests/test_gate_router.py       # Router 和冲突配置单测
```

`learned` 接口只为 Phase 3 预留；当前没有模型文件时会显式报错，不会退化为
其他路由。

## 3. Router 模式

| mode | 行为 |
| --- | --- |
| `fixed_0` | 每次 replan 选择 `N=0` |
| `fixed_full` | 每次 replan 选择 `N=num_inference_steps` |
| `random` | 以 `full_probability` 的概率选择 full，否则选择 0 |
| `learned` | 预留给后续 utility MLP；本阶段配置会 fail fast |

`random` 使用 `seed + task + episode + replan` 生成无状态确定性随机数。
相同 rollout 标识在重试或调度变化后仍得到相同决策，便于复现。

## 4. 配置方法

Gate 默认关闭，因此原有 eval 行为不变。启用时必须同时使用 prefix：

```yaml
EVALUATION:
  inference_mode: prefix
  gate:
    enabled: true
    mode: fixed_0
    full_probability: 0.5
    seed: 42
    threshold: 0.0
```

常用 Hydra override：

```bash
# Phase 0：N=0 endpoint
EVALUATION.inference_mode=prefix \
EVALUATION.gate.enabled=true \
EVALUATION.gate.mode=fixed_0

# Phase 0：N=full endpoint（当前常用 num_inference_steps=10）
EVALUATION.inference_mode=prefix \
EVALUATION.gate.enabled=true \
EVALUATION.gate.mode=fixed_full

# Phase 1：约 30% replan 使用 full
EVALUATION.inference_mode=prefix \
EVALUATION.gate.enabled=true \
EVALUATION.gate.mode=random \
EVALUATION.gate.full_probability=0.3
```

Gate 与 `visualize_future_video=true` 不兼容。程序在加载 checkpoint 前就会
报错，防止 evaluator 通过 `infer_joint()` 静默绕过 prefix Gate。

## 5. 路由校验和结果字段

模型调用后，evaluator 强制校验返回的 `video_prefix_steps` 与
`selected_n` 一致；不一致立即抛出异常。

每次 replan 在日志和结果 JSON 中记录：

```text
task_key, episode_idx, replan_idx
gate_mode, gate_score, threshold, selected_n
num_inference_steps
gate_latency_ms, action_inference_latency_ms
```

每个 episode 汇总：

```text
success, num_replans, num_n0, num_nfull
n_full_fraction, n_eff
```

任务级 `gate_actual_n_eff` 按 closed-loop 中实际访问到的全部 replan 计算，
而不是直接使用配置概率。

## 6. Phase 0 / Phase 1 验收

自动检查覆盖：

- `fixed_0` 永远输出 0；
- `fixed_full` 永远输出 full；
- `random(p=0.3)` 长期 full 比例约为 30%；
- 随机决策可复现；
- Gate 非 prefix 配置直接报错；
- Gate 与 future-video visualization 同开直接报错。

提交前运行：

```bash
PYTHONPATH=src:. python -m pytest \
  tests/test_gate_router.py tests/test_episode_selector.py -q
```

完整 GPU / simulator 验收仍应使用同一 checkpoint、seed 和 episode 集合比较：

1. Gate `fixed_0` 与原静态 prefix `N=0`；
2. Gate `fixed_full` 与原静态 prefix `N=num_inference_steps`；
3. `random(p=0.3)` 的 closed-loop `n_full_fraction` 与 `N_eff`。

这三项通过后再进入 Phase 2 utility 标签采集；本阶段不训练 MLP，也不修改
UniShare checkpoint。
