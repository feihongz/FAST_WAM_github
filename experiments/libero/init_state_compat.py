from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch


def load_libero_task_init_states(
    task_suite: Any,
    task_id: int,
    *,
    init_states_root: str | Path,
):
    """Load trusted local LIBERO init states across the PyTorch 2.6 change."""

    task = task_suite.get_task(task_id)
    trusted_root = Path(init_states_root).resolve()
    init_states_path = (
        trusted_root / task.problem_folder / task.init_states_file
    ).resolve()
    try:
        init_states_path.relative_to(trusted_root)
    except ValueError as exc:
        raise ValueError(
            f"LIBERO init-state path escapes configured root: {init_states_path}"
        ) from exc
    if not init_states_path.is_file():
        raise FileNotFoundError(f"Missing LIBERO init-state file: {init_states_path}")

    logging.info("Loading trusted local LIBERO init states: %s", init_states_path)
    try:
        return torch.load(
            init_states_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # PyTorch releases predating ``weights_only`` already use the legacy
        # full-pickle behavior required by official LIBERO init-state files.
        return torch.load(init_states_path, map_location="cpu")
