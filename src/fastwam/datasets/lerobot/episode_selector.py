"""Deterministic episode selectors for read-only LeRobot dataset subsets."""

from collections.abc import Sequence

import numpy as np


class GroupedStratifiedEpisodeSelector:
    """Select the same fraction from fixed strata inside every episode group.

    The FastWAM RoboTwin 2.0 conversion used by this repository stores each of
    its 50 task groups as 550 consecutive episodes: 50 clean episodes followed
    by 500 randomized episodes. Selecting each stratum independently keeps both
    the task distribution and the clean/randomized ratio unchanged without
    copying or modifying the source dataset.
    """

    def __init__(
        self,
        fraction: float,
        group_size: int,
        strata_sizes: Sequence[int],
        seed: int = 42,
        expected_total_episodes: int | None = None,
        expected_selected_episodes: int | None = None,
    ) -> None:
        self.fraction = float(fraction)
        self.group_size = int(group_size)
        self.strata_sizes = tuple(int(size) for size in strata_sizes)
        self.seed = int(seed)
        self.expected_total_episodes = (
            None if expected_total_episodes is None else int(expected_total_episodes)
        )
        self.expected_selected_episodes = (
            None if expected_selected_episodes is None else int(expected_selected_episodes)
        )
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")
        if not self.strata_sizes or any(size <= 0 for size in self.strata_sizes):
            raise ValueError(f"strata_sizes must contain positive integers, got {self.strata_sizes}")
        if sum(self.strata_sizes) != self.group_size:
            raise ValueError(
                "strata_sizes must sum to group_size, got "
                f"sum={sum(self.strata_sizes)} group_size={self.group_size}"
            )
        for size in self.strata_sizes:
            selected = size * self.fraction
            if not np.isclose(selected, round(selected)):
                raise ValueError(
                    "fraction must select an integer number of episodes from every stratum, "
                    f"got {size} * {self.fraction} = {selected}"
                )

    def __call__(self, total_episodes: int) -> list[int]:
        total_episodes = int(total_episodes)
        if self.expected_total_episodes is not None and total_episodes != self.expected_total_episodes:
            raise ValueError(
                "Unexpected source episode count: "
                f"expected {self.expected_total_episodes}, got {total_episodes}"
            )
        if total_episodes <= 0 or total_episodes % self.group_size != 0:
            raise ValueError(
                f"total_episodes must be a positive multiple of {self.group_size}, "
                f"got {total_episodes}"
            )

        rng = np.random.default_rng(self.seed)
        selected_indices: list[int] = []
        for group_start in range(0, total_episodes, self.group_size):
            stratum_start = group_start
            for stratum_size in self.strata_sizes:
                selected_count = int(round(stratum_size * self.fraction))
                candidates = np.arange(stratum_start, stratum_start + stratum_size)
                selected_indices.extend(
                    int(index)
                    for index in rng.choice(candidates, size=selected_count, replace=False)
                )
                stratum_start += stratum_size

        selected_indices.sort()
        if (
            self.expected_selected_episodes is not None
            and len(selected_indices) != self.expected_selected_episodes
        ):
            raise ValueError(
                "Unexpected selected episode count: "
                f"expected {self.expected_selected_episodes}, got {len(selected_indices)}"
            )
        return selected_indices
