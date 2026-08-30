# Stage 3 endpoint evaluation

Stage 3 export is an Adapter-only artifact. Endpoint evaluation must construct
the aligned model, then load the exact frozen UnifiedShared base followed by the
exact benchmark-specific Adapter. Passing the Adapter export as `ckpt` is
incorrect.

The endpoint loader is fail closed. Before model construction it verifies the
base, Adapter, data-manifest binding, training contract, global step, VAE and
normalization stats. After construction it also checks that the model actually
loaded the contract-bound VAE, loads base then Adapter strictly, freezes the
whole model, and rechecks artifact identities. An aligned Hydra task without an
Adapter is rejected instead of silently evaluating an untrained Adapter.

## 200-step pilot connectivity smoke

Submit these as two independent one-H100 JiHe jobs:

```bash
bash scripts/jihe/eval_libero_stage3_pilot_endpoint_1xh100.sh
```

```bash
bash scripts/jihe/eval_robotwin_stage3_pilot_endpoint_1xh100.sh
```

Both commands default to one episode, two diffusion steps and
`inference_mode=w`. The `w` branch is mandatory for this smoke because it is
the branch on which the Stage 3 Adapter is applied. The scripts print the full
command before launch; `FASTWAM_DRY_RUN=1` prints it without touching GPU or
output state.

The locked pilot artifacts are:

| benchmark | Adapter SHA256 | base SHA256 | data SHA256 | contract SHA256 |
| --- | --- | --- | --- | --- |
| LIBERO | `d18341299afdd21474affc2358ec5cf1d8fe34cef6f0c7b7149e6d2f97645ac5` | `17a5588cc2b8d162219c9daf818614f614ee4a7921933a4a26c5d678111330e9` | `08da49109a57b55c67f3fa4ac31fbfa44e44dd541a194a5d3420838537d0d320` | `57600680a8cc33b2cdc1a372de622bc189e34fdd23fcbfe26031c8a74a82ac23` |
| RoboTwin 2.0 | `e5d984edb0bab0cb29c97b5bf484b882294d4430d7b04490d972180a0ecd2780` | `368a99ca9575a78d01f4cdcdee8820ec74d30c4528cf7aff07b83361a17cbbda` | `1190b75b1ef19a7abd949bdff5679da59afa7e51a043eeb43663cf2c4495173c` | `e9c18c334bf7863039c4a14e4f2db6c7d344688a9c1e047de75054399e1283e7` |

Default Adapter locations:

- LIBERO: `/root/feihong/FastWAM/formal_runs/pilots/stage3/libero_stage3_alignment_2cam224_1e-4/2026-08-30_04-08-43/checkpoints/exports/step_000200.pt`
- RoboTwin 2.0: `/root/feihong/FastWAM/formal_runs/pilots/stage3/robotwin_stage3_alignment_3cam384_1e-4/2026-08-29_17-00-07/checkpoints/exports/step_000200.pt`

The environment variables in the launchers allow relocating a complete,
identity-consistent artifact set. Changing an expected SHA or contract creates
a different experiment; the endpoint loader still verifies every supplied
identity and never falls back to an unaligned model.

Both exports have `global_step=200`. They bind the same Wan2.2 VAE SHA256
`20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36`.

LIBERO writes the verified `model_artifact_identity` into its task result
JSON. RoboTwin writes
`<output>/<task>/stage3_endpoint_model_identity.json` before rollout and
saves its resolved Hydra config in the output root.

The SHA preflight reads the large base and VAE more than once to close
inspect/load races. Several minutes without rollout output can therefore be
normal; the launch log prints a preflight marker before this phase.

The LIBERO worker explicitly loads the benchmark's trusted local legacy
init-state files with full-pickle mode. This is required because PyTorch 2.6+
changed the default of `torch.load(weights_only=...)`; the loader constrains
the resolved file to the configured LIBERO init-state root before loading it.
The RoboTwin launcher also requires its external `assets`, `checkpoints`, and
`task_config` mounts or links to resolve before starting the simulator.

## Interpretation and next order

These scripts test the complete simulator-to-Adapter-to-action path. The
200-step checkpoints remain pilot artifacts, so their success rates must not be
reported as formal benchmark results.

After both connectivity smokes pass:

1. choose and run the full Stage 3 training schedule for each benchmark;
2. freeze each final Adapter and bind all artifact identities;
3. run only a tiny final `w`-branch health smoke, not a standalone full
   `w-only` benchmark evaluation;
4. use that frozen Adapter immediately to generate matched `wo/w` data and
   the benchmark's Stage 2 `E0/E10` labels;
5. train the small binary Gate;
6. run the meaningful final full comparison of always-`wo`, always-`w`, and
   Gate routing, including the success/compute threshold sweep.

The standalone full `w-only` pass is intentionally deferred and folded into
the final comparison; artifact freezing and the tiny load/execute health smoke
remain mandatory.

Stage 3 training used cached text context, so its export binds VAE and stats but
does not yet bind the online text encoder/tokenizer bytes. Pilot receipts record
the actual resolved text/tokenizer paths. Before a result is called formal,
their immutable identities must also be locked in the endpoint evaluation
contract.
