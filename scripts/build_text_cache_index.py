#!/usr/bin/env python3
"""Build a compact integrity index for the selected text-cache payloads.

The job derives exact cache paths from the instantiated training dataset.  It
does not enumerate the cache directory and therefore cannot accidentally bind
unrelated files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from fastwam.alignment.data_identity import (
    DEFAULT_PROMPT_TEMPLATE,
    TEXT_CACHE_FILENAME_SUFFIX_TEMPLATE,
    selected_text_cache_prompts,
)
from fastwam.alignment.runtime import (
    _canonicalize_data_paths,
    _repo_dir,
    _resolved_config,
    _validate_formal_dataset,
    _validate_required_environment,
)
from fastwam.alignment.text_cache_index import build_text_cache_index
from fastwam.runtime import build_datasets
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.pytorch_utils import set_global_seed


register_default_resolvers()


def _strict_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be bool")
    return value


def _strict_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="train_stage3_alignment",
)
def main(config: DictConfig) -> None:
    resolved = _resolved_config(config)
    runtime_config = dict(resolved["runtime"])
    _validate_required_environment(runtime_config)
    resolved["data"] = _canonicalize_data_paths(
        dict(resolved["data"]),
        repo_dir=_repo_dir(runtime_config),
    )

    raw_spec = resolved.get("text_cache_index")
    if not isinstance(raw_spec, dict):
        raise ValueError(
            "set +text_cache_index.path=/absolute/output/index.bin"
        )
    allowed = {"path", "descriptor_path", "overwrite", "progress", "workers"}
    if set(raw_spec) - allowed:
        raise ValueError("text_cache_index contains unsupported fields")
    output_value = raw_spec.get("path")
    if not isinstance(output_value, str) or not output_value:
        raise ValueError("text_cache_index.path must be a non-empty path")
    index_path = Path(output_value).expanduser().resolve()
    descriptor_value = raw_spec.get("descriptor_path")
    descriptor_path = (
        Path(descriptor_value).expanduser().resolve()
        if descriptor_value
        else index_path.with_suffix(index_path.suffix + ".json")
    )
    overwrite = _strict_bool(
        raw_spec.get("overwrite", False), field="text_cache_index.overwrite"
    )
    show_progress = _strict_bool(
        raw_spec.get("progress", True), field="text_cache_index.progress"
    )
    workers = _strict_positive_int(
        raw_spec.get("workers", 1), field="text_cache_index.workers"
    )
    misc.register_work_dir(index_path.parent)

    set_global_seed(int(resolved["training"]["seed"]))
    train_dataset, _ = build_datasets(OmegaConf.create(resolved["data"]))
    cardinality = _validate_formal_dataset(
        train_dataset,
        runtime_config=runtime_config,
    )
    prompts = selected_text_cache_prompts(train_dataset)
    context_len = int(train_dataset.context_len)
    filename_suffix = TEXT_CACHE_FILENAME_SUFFIX_TEMPLATE.format(
        context_len=context_len
    )

    with tqdm(
        total=len(prompts),
        desc="Hashing selected text cache",
        unit="prompt",
        dynamic_ncols=True,
        disable=not show_progress,
    ) as progress_bar:

        def update(completed: int, total: int) -> None:
            if total != progress_bar.total:
                raise RuntimeError("text cache progress total changed")
            progress_bar.update(completed - progress_bar.n)

        descriptor = build_text_cache_index(
            cache_root=train_dataset.text_embedding_cache_dir,
            prompts=prompts,
            context_len=context_len,
            prompt_template=DEFAULT_PROMPT_TEMPLATE,
            filename_suffix=filename_suffix,
            index_path=index_path,
            descriptor_path=descriptor_path,
            overwrite=overwrite,
            workers=workers,
            progress=update if show_progress else None,
        )

    print(
        "Text cache index written:\n"
        f"  index: {index_path}\n"
        f"  descriptor: {descriptor_path}\n"
        f"  descriptor_sha256: {descriptor['descriptor_sha256']}\n"
        f"  index_sha256: {descriptor['index']['sha256']}\n"
        f"  prompts: {descriptor['record_count']}\n"
        f"  workers: {workers}\n"
        f"  frames: {cardinality['dataset_length']}\n"
        f"  episodes: {cardinality['dataset_episodes']}"
    )


if __name__ == "__main__":
    main()
