from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from experiments.libero.gate.collect_tiny_mlp_features import (
    BUNDLE_KIND,
    COMPLETION_FILENAME,
    EXPECTED_DIMS,
    FEATURES_FILENAME,
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    SCIENTIFIC_SOURCE_FILES,
    TENSOR_KEYS,
    _atomic_write_bytes,
    _load_progress_row,
    _scientific_source_provenance,
    _serialize_json,
    _serialize_jsonl,
    _validate_live_state,
    _write_progress_row,
    build_feature_record,
    build_projection_matrices,
    completion_payload,
    ensure_immutable_manifest,
    extract_allowed_features,
    feature_record_sha256,
    pool_instruction_context,
    pool_visual_latent,
    prepare_vae_input,
    projection_metadata,
    rademacher_projection,
    sha256_json,
    tensor_content_sha256,
    validate_completion,
    validate_feature_record,
    validate_manifest,
)
from experiments.libero.gate.demo_utility import (
    current_state_input_hashes,
    extract_current_state,
)


def _sample(*, future_value: float = 9.0, action_value: float = 3.0) -> dict:
    video = torch.zeros(3, 3, 4, 5, dtype=torch.float32)
    video[:, 0] = 1.0
    video[:, 1:] = future_value
    context = torch.zeros(4, 6, dtype=torch.float32)
    context[0] = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
    context[1] = torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.float32)
    return {
        "video": video,
        "action": torch.full((3, 7), action_value, dtype=torch.float32),
        "proprio": torch.arange(24, dtype=torch.float32).reshape(3, 8),
        "context": context,
        # RobotVideoDataset rewrites this to all ones after zero padding.
        "context_mask": torch.ones(4, dtype=torch.bool),
        "action_is_pad": torch.tensor([False, False, True]),
    }


def _small_projections() -> dict[str, torch.Tensor]:
    return {
        "visual": rademacher_projection(48 * 9, 64, 91),
        "instruction_mean": rademacher_projection(6, 32, 92),
        "instruction_rms": rademacher_projection(6, 32, 93),
    }


def _fake_encoder(image: torch.Tensor) -> torch.Tensor:
    # Deliberately depends only on the supplied [3,H,W] current image.
    assert image.ndim == 3 and image.shape[0] == 3
    base = image.float().mean()
    spatial = torch.arange(48 * 2 * 3, dtype=torch.float32).reshape(1, 48, 1, 2, 3)
    return spatial + base


def _extract_from_sample(sample: dict) -> dict[str, torch.Tensor]:
    state = extract_current_state(sample)
    return extract_allowed_features(
        input_image=state.input_image,
        context=state.context,
        proprio=state.proprio,
        encode_current_image=_fake_encoder,
        projections=_small_projections(),
        context_dim=6,
    )


def _target(sample: dict, *, source_index: int = 123, order: int = 0) -> dict:
    state = extract_current_state(sample)
    target_sha = "a" * 64
    return {
        "selection_order": order,
        "sample_id": "libero_spatial_no_noops_lerobot/episode_000002/frame_000004",
        "source_index": source_index,
        "suite": "libero_spatial",
        "dataset_id": "libero_spatial_no_noops_lerobot",
        "dataset_name": "libero_spatial_no_noops_lerobot",
        "episode_index": 2,
        "frame_index": 4,
        "task_index": 3,
        "task": "put the object on the table",
        "target_id": "target-v2/000",
        "target_sha256": target_sha,
        "input_hashes": current_state_input_hashes(state),
        "current_proprio": state.proprio.tolist(),
    }


def _full_features() -> dict[str, torch.Tensor]:
    visual = torch.linspace(-1, 1, EXPECTED_DIMS["visual"], dtype=torch.float32)
    instruction = torch.linspace(-2, 2, EXPECTED_DIMS["instruction"], dtype=torch.float32)
    proprio = torch.arange(EXPECTED_DIMS["proprio"], dtype=torch.float32)
    return {
        "visual": visual,
        "instruction": instruction,
        "proprio": proprio,
        "full": torch.cat((visual, instruction, proprio)),
    }


def _extractor() -> dict:
    contract = {
        "schema_version": 1,
        "feature_dimensions": dict(EXPECTED_DIMS),
        "projection": projection_metadata(torch.eye(2), seed=7),
    }
    contract["extractor_fingerprint"] = sha256_json(contract)
    return contract


def _minimal_manifest() -> dict:
    extractor = _extractor()
    compatibility = {
        "schema_version": 1,
        "kind": BUNDLE_KIND,
        "feature_record_schema_version": 1,
        "feature_dimensions": dict(EXPECTED_DIMS),
        "extractor": extractor,
        "extractor_fingerprint": extractor["extractor_fingerprint"],
        "num_states": 1,
        "global_join_contract": {
            "keys": ["sample_id", "source_index"],
            "source_index_semantics": "global requested_sample_idx/source_sample_idx",
            "dataset_local_source_metadata.source_index_allowed_as_join": False,
        },
    }
    return {
        "schema_version": 1,
        "kind": BUNDLE_KIND,
        "created_at_utc": "2026-08-13T00:00:00+00:00",
        "compatibility": compatibility,
        "compatibility_fingerprint": sha256_json(compatibility),
    }


def test_protocol_projection_matrices_are_deterministic_and_content_addressed():
    first = build_projection_matrices()
    second = build_projection_matrices()
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert {key: tuple(value.shape) for key, value in first.items()} == {
        "visual": (432, 64),
        "instruction_mean": (4096, 32),
        "instruction_rms": (4096, 32),
    }
    # These are intentional protocol anchors, not merely equality within one run.
    assert tensor_content_sha256(first["visual"]) == (
        "da3e212062de024fe7c07e670590fecd7bc456d3edb9c59890d1819f26cea50a"
    )
    assert tensor_content_sha256(first["instruction_mean"]) == (
        "13ddb4308889fdb2f9e2bd740abd3c82cccfb4bba26246f37a3fd3992bdfc280"
    )
    assert tensor_content_sha256(first["instruction_rms"]) == (
        "f25e240caa73a2286fea7836368248e4cc60878ebcb76635d90e066c596e76aa"
    )


def test_scientific_source_provenance_binds_runtime_feature_dependencies():
    required = {
        "experiments/libero/gate/collect_demo_utility.py",
        "src/fastwam/models/wan22/helpers/state_dict_converters.py",
        "src/fastwam/utils/config_resolvers.py",
    }
    assert required.issubset(set(SCIENTIFIC_SOURCE_FILES))
    provenance = _scientific_source_provenance()
    assert set(provenance) == set(SCIENTIFIC_SOURCE_FILES)
    for relative_path in required:
        digest = provenance[relative_path]
        assert len(digest) == 64
        assert digest == digest.lower()
        assert set(digest) <= set("0123456789abcdef")


def test_visual_pooling_is_channel_major_and_uses_population_std():
    latent = torch.tensor(
        [
            [[[[1.0, 3.0], [5.0, 7.0]]]],
            [[[[2.0, 4.0], [6.0, 8.0]]]],
        ]
    ).reshape(1, 2, 1, 2, 2)
    # pool 1x2 gives 4 values; then one population std per channel.
    projection = torch.eye(6, dtype=torch.float32)
    result = pool_visual_latent(
        latent,
        projection,
        latent_channels=2,
        pooled_height=1,
        pooled_width=2,
    )
    expected = torch.tensor(
        [3.0, 5.0, 4.0, 6.0, torch.tensor([1, 3, 5, 7.0]).std(correction=0),
         torch.tensor([2, 4, 6, 8.0]).std(correction=0)]
    )
    assert torch.allclose(result, expected)
    with pytest.raises(ValueError, match="exact protocol"):
        pool_visual_latent(torch.zeros(1, 2, 2, 2, 2), projection, latent_channels=2,
                           pooled_height=1, pooled_width=2)


def test_instruction_pooling_uses_nonzero_rows_mean_rms_and_active_fraction():
    context = torch.tensor(
        [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0], [0.0, 0.0]], dtype=torch.float32
    )
    result = pool_instruction_context(
        context, torch.eye(2), torch.eye(2), context_dim=2
    )
    expected = torch.tensor(
        [2.0, 3.0, (5.0**0.5), (10.0**0.5), 0.5], dtype=torch.float32
    )
    assert torch.allclose(result, expected)
    with pytest.raises(ValueError, match="no active"):
        pool_instruction_context(torch.zeros_like(context), torch.eye(2), torch.eye(2),
                                 context_dim=2)


def test_feature_extractor_cannot_observe_future_video_action_or_pad_fields():
    first = _sample(future_value=9.0, action_value=3.0)
    second = _sample(future_value=-900.0, action_value=-300.0)
    second["action_is_pad"] = ~first["action_is_pad"]
    features_a = _extract_from_sample(first)
    features_b = _extract_from_sample(second)
    for key in TENSOR_KEYS:
        assert torch.equal(features_a[key], features_b[key])
    # Audit validation still detects label/action changes, even though they are
    # impossible to pass into extract_allowed_features.
    assert current_state_input_hashes(extract_current_state(first)) != (
        current_state_input_hashes(extract_current_state(second))
    )
    assert tuple(features_a["full"].shape) == (137,)
    assert torch.equal(
        features_a["full"],
        torch.cat((features_a["visual"], features_a["instruction"], features_a["proprio"])),
    )


def test_prepare_vae_input_matches_unishare_shape_and_dtype():
    image = torch.linspace(-1, 1, 3 * 8 * 12, dtype=torch.float32).reshape(3, 8, 12)
    prepared = prepare_vae_input(
        image, device=torch.device("cpu"), dtype=torch.bfloat16
    )
    assert prepared.shape == (3, 1, 8, 12)
    assert prepared.dtype == torch.bfloat16
    assert torch.equal(prepared[:, 0], image.to(torch.bfloat16))


def test_prepare_vae_input_rejects_nonfinite_or_future_axis():
    with pytest.raises(ValueError, match=r"\[3,H,W\]"):
        prepare_vae_input(
            torch.zeros(3, 2, 8, 12),
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
    image = torch.zeros(3, 8, 12)
    image[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        prepare_vae_input(
            image, device=torch.device("cpu"), dtype=torch.bfloat16
        )


def test_live_join_uses_global_source_index_and_ignores_dataset_local_collision():
    sample = _sample()
    sample["metadata"] = {
        "dataset_index": 0,
        "dataset_id": "libero_spatial_no_noops_lerobot",
        "dataset_name": "libero_spatial_no_noops_lerobot",
        "suite": "libero_spatial",
        "episode_index": 2,
        "frame_index": 4,
        "task_index": 3,
        "task": "put the object on the table",
        "requested_sample_idx": 123,
        "source_sample_idx": 123,
        # This value is local to one constituent dataset and deliberately differs.
        "source_index": 7,
    }
    target = _target(sample)
    state = _validate_live_state(
        sample=sample,
        target=target,
        ranges=[{
            "dataset_index": 0,
            "dataset_id": "libero_spatial_no_noops_lerobot",
            "dataset_name": "libero_spatial_no_noops_lerobot",
            "start": 0,
            "stop": 1000,
            "population": 1000,
        }],
        task_tables={0: {3: "put the object on the table"}},
    )
    assert torch.equal(state.input_image, sample["video"][:, 0])
    broken = copy.deepcopy(target)
    broken["source_index"] = 7
    with pytest.raises((ValueError, AssertionError)):
        _validate_live_state(
            sample=sample,
            target=broken,
            ranges=[{
                "dataset_index": 0,
                "dataset_id": "libero_spatial_no_noops_lerobot",
                "dataset_name": "libero_spatial_no_noops_lerobot",
                "start": 0,
                "stop": 1000,
                "population": 1000,
            }],
            task_tables={0: {3: "put the object on the table"}},
        )


def test_feature_record_binds_target_input_extractor_and_exact_row_hashes():
    target = _target(_sample())
    features = _full_features()
    fingerprint = "b" * 64
    record = build_feature_record(target, features, extractor_fingerprint=fingerprint)
    validate_feature_record(
        record, features, target, extractor_fingerprint=fingerprint
    )
    assert set(record["feature_hashes"]) == set(TENSOR_KEYS)
    assert record["feature_record_sha256"] == feature_record_sha256(record)

    tampered = copy.deepcopy(record)
    tampered["source_index"] += 1
    tampered["feature_record_sha256"] = feature_record_sha256(tampered)
    with pytest.raises(ValueError, match="Target V2"):
        validate_feature_record(
            tampered, features, target, extractor_fingerprint=fingerprint
        )
    tampered_features = dict(features)
    tampered_features["visual"] = features["visual"].clone()
    tampered_features["visual"][0] += 1
    with pytest.raises(ValueError, match="content hash"):
        validate_feature_record(
            record, tampered_features, target, extractor_fingerprint=fingerprint
        )


def test_progress_resume_rejects_orphan_and_tampered_row(tmp_path: Path):
    progress = tmp_path / ".rows"
    target = _target(_sample())
    features = _full_features()
    fingerprint = "c" * 64
    record = build_feature_record(target, features, extractor_fingerprint=fingerprint)
    _write_progress_row(
        progress, record, features, extractor_fingerprint=fingerprint
    )
    loaded = _load_progress_row(
        progress,
        target,
        extractor_fingerprint=fingerprint,
        expected_dimensions=EXPECTED_DIMS,
    )
    assert loaded is not None and loaded[0] == record

    json_path = progress / "000000.json"
    stored = json.loads(json_path.read_text())
    stored["task"] = "tampered"
    # Even recomputing the row's self hash cannot evade the Target-V2 binding.
    stored["feature_record_sha256"] = feature_record_sha256(stored)
    json_path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="Target V2"):
        _load_progress_row(
            progress,
            target,
            extractor_fingerprint=fingerprint,
            expected_dimensions=EXPECTED_DIMS,
        )

    json_path.unlink()
    with pytest.raises(ValueError, match="orphaned"):
        _load_progress_row(
            progress,
            target,
            extractor_fingerprint=fingerprint,
            expected_dimensions=EXPECTED_DIMS,
        )


def test_manifest_resume_is_immutable_and_rejects_dataset_local_join(tmp_path: Path):
    manifest = _minimal_manifest()
    path = tmp_path / MANIFEST_FILENAME
    stored = ensure_immutable_manifest(path, manifest)
    assert stored["compatibility_fingerprint"] == manifest["compatibility_fingerprint"]
    changed = copy.deepcopy(manifest)
    changed["compatibility"]["num_states"] = 2
    changed["compatibility_fingerprint"] = sha256_json(changed["compatibility"])
    with pytest.raises(ValueError, match="differs"):
        ensure_immutable_manifest(path, changed)

    unsafe = copy.deepcopy(manifest)
    unsafe["compatibility"]["global_join_contract"][
        "dataset_local_source_metadata.source_index_allowed_as_join"
    ] = True
    unsafe["compatibility_fingerprint"] = sha256_json(unsafe["compatibility"])
    with pytest.raises(ValueError, match="dataset-local"):
        validate_manifest(unsafe)


def test_completion_seal_binds_all_file_bytes_and_tensor_contents(tmp_path: Path):
    manifest = _minimal_manifest()
    manifest_path = tmp_path / MANIFEST_FILENAME
    _atomic_write_bytes(manifest_path, _serialize_json(manifest))
    index_path = tmp_path / INDEX_FILENAME
    _atomic_write_bytes(index_path, _serialize_jsonl([{"row": 0}]))
    matrices = {key: torch.zeros(1, EXPECTED_DIMS[key]) for key in TENSOR_KEYS}
    features_path = tmp_path / FEATURES_FILENAME
    save_file(matrices, str(features_path))
    completion = completion_payload(
        manifest_path=manifest_path,
        index_path=index_path,
        features_path=features_path,
        matrices=matrices,
        manifest_fingerprint=manifest["compatibility_fingerprint"],
        num_states=1,
    )
    _atomic_write_bytes(tmp_path / COMPLETION_FILENAME, _serialize_json(completion))
    assert validate_completion(tmp_path)["completion_sha256"] == completion[
        "completion_sha256"
    ]

    index_path.write_text('{"row":1}\n')
    with pytest.raises(ValueError, match="feature_index_sha256"):
        validate_completion(tmp_path)


def test_config_has_only_target5_input_and_no_validation4_surface():
    text = (Path(__file__).parents[1] / "configs" / "collect_libero_gate_features.yaml").read_text()
    assert "target_v2_dir" in text
    assert "expected_target_manifest_sha256" in text
    assert "expected_target_records_sha256" in text
    assert "validation" not in text.lower()
