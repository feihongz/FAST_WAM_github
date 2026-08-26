from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts import generate_gate_labels as generate_cli
from scripts import train_video_gate as train_cli


def _compose_stage2(config_name: str):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(config_dir),
    ):
        return compose(
            config_name=config_name,
            overrides=["data=robotwin_formal"],
        )


def test_robotwin_formal_is_train_only_and_keeps_base_selection_contract():
    repo = Path(__file__).resolve().parents[1]
    raw = OmegaConf.load(repo / "configs" / "data" / "robotwin_formal.yaml")

    assert raw.val is None
    assert "strict_data_mode" not in raw.train
    assert "video_backend" not in raw.train
    assert "save_stats_copy" not in raw.train
    assert raw.train.val_set_proportion == 0.01
    assert raw.train.is_training_set is True
    assert raw.train.seed == 42
    assert int(27_500 * (1.0 - raw.train.val_set_proportion)) == 27_225
    assert raw.train.num_frames == 33
    assert raw.train.action_video_freq_ratio == 4
    assert list(raw.train.video_size) == [384, 320]
    assert raw.train.concat_multi_camera == "robotwin"
    assert raw.train.text_context_cache_max_entries == 16
    assert [item.key for item in raw.train.shape_meta.images] == [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]
    assert raw.train.processor.num_output_cameras == 3
    assert raw.train.processor.action_output_dim == 14
    assert raw.train.processor.proprio_output_dim == 14


def test_robotwin_formal_composes_for_both_stage2_entrypoints():
    repo = Path(__file__).resolve().parents[1]
    generated = _compose_stage2("generate_gate_labels")
    gate = _compose_stage2("train_video_gate")

    for config in (generated, gate):
        assert config.data.val is None
        assert config.data.train.strict_data_mode is True
        assert config.data.train.video_backend == "torchcodec"
        assert config.data.train.save_stats_copy is False
        assert config.data.train.seed == 42
        assert config.data.train.text_context_cache_max_entries == 16
        assert config.data.train.processor.action_output_dim == 14
        assert config.data.train.processor.proprio_output_dim == 14

    assert gate.gate.proprio_dim == 14
    generated_data = OmegaConf.to_container(generated.data, resolve=True)
    gate_data = OmegaConf.to_container(gate.data, resolve=True)
    generated_canonical = generate_cli._canonicalize_data_paths(
        generated_data,
        repo_dir=repo,
    )
    gate_canonical = train_cli._canonicalize_data_paths(
        gate_data,
        repo_dir=repo,
    )
    assert generated_canonical == gate_canonical
    assert generated_canonical["val"] is None
    assert generated_canonical["train"]["dataset_dirs"] == [
        str((repo / "data" / "robotwin2.0" / "robotwin2.0").resolve())
    ]
    assert generated_canonical["train"]["pretrained_norm_stats"] == str(
        (repo / "data" / "robotwin2.0" / "dataset_stats.json").resolve()
    )
