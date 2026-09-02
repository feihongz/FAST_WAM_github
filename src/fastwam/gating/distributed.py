"""Strict helpers for single-node distributed Stage 2 Gate training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

import torch
import torch.distributed as dist


@dataclass(frozen=True, slots=True)
class DistributedGateContext:
    """The initialized process-group identity used by one Gate rank."""

    rank: int
    local_rank: int
    world_size: int
    backend: str

    def __post_init__(self) -> None:
        for name in ("rank", "local_rank", "world_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"distributed {name} must be an integer")
        if self.world_size <= 1:
            raise ValueError("distributed world_size must be greater than one")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("distributed rank is outside world_size")
        if not 0 <= self.local_rank < self.world_size:
            raise ValueError("distributed local_rank is outside world_size")
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("distributed backend must be a non-empty string")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def _require_initialized(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Gate distributed process group is not initialized")
        if dist.get_rank() != self.rank or dist.get_world_size() != self.world_size:
            raise RuntimeError("Gate distributed process-group identity drifted")
        if str(dist.get_backend()) != self.backend:
            raise RuntimeError("Gate distributed process-group backend drifted")

    def barrier(self) -> None:
        self._require_initialized()
        dist.barrier()

    def sum_tensor(self, value: torch.Tensor) -> torch.Tensor:
        """Return an all-reduced sum without mutating the caller's tensor."""

        self._require_initialized()
        if not isinstance(value, torch.Tensor):
            raise TypeError("distributed reduction value must be a tensor")
        reduced = value.detach().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        return reduced

    def gather_tensor(
        self,
        value: torch.Tensor,
        *,
        label: str,
    ) -> torch.Tensor:
        """Gather variable axis-zero lengths in rank order."""

        self._require_initialized()
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            raise TypeError(f"{label} must be a non-scalar tensor")
        recorded = self.gather_objects(tuple(int(item) for item in value.shape))
        if any(
            not isinstance(candidate, tuple)
            or not candidate
            or any(not isinstance(item, int) or item < 0 for item in candidate)
            for candidate in recorded
        ):
            raise RuntimeError(f"distributed {label} shapes are invalid")
        tail_shape = tuple(value.shape[1:])
        if any(tuple(candidate[1:]) != tail_shape for candidate in recorded):
            raise RuntimeError(
                f"distributed {label} trailing shapes differ across ranks: "
                f"{recorded}"
            )
        maximum = max(candidate[0] for candidate in recorded)
        padded_shape = (maximum, *tail_shape)
        padded = torch.zeros(
            padded_shape,
            dtype=value.dtype,
            device=value.device,
        )
        padded[: value.shape[0]].copy_(value)
        gathered = [torch.empty_like(padded) for _ in range(self.world_size)]
        dist.all_gather(gathered, padded)
        return torch.cat(
            [
                tensor[: candidate_shape[0]]
                for tensor, candidate_shape in zip(gathered, recorded)
            ],
            dim=0,
        )

    def gather_objects(self, value: Any) -> tuple[Any, ...]:
        """Return one picklable value from every rank in rank order."""

        self._require_initialized()
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return tuple(gathered)

    def assert_same(
        self,
        value: Mapping[str, Any],
        *,
        label: str,
    ) -> Mapping[str, Any]:
        """Fail if a canonical-JSON runtime contract differs across ranks."""

        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        gathered = self.gather_objects(encoded)
        if any(candidate != encoded for candidate in gathered):
            raise RuntimeError(f"distributed {label} differs across ranks")
        return value


__all__ = ["DistributedGateContext"]
