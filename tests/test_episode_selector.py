import unittest

import numpy as np

from fastwam.datasets.lerobot.episode_selector import GroupedStratifiedEpisodeSelector


class GroupedStratifiedEpisodeSelectorTest(unittest.TestCase):
    def setUp(self):
        self.selector = GroupedStratifiedEpisodeSelector(
            fraction=0.2,
            group_size=550,
            strata_sizes=[50, 500],
            seed=42,
            expected_total_episodes=27_500,
            expected_selected_episodes=5_500,
        )

    def test_robotwin_subset_size_and_balance(self):
        selected = self.selector(27_500)

        self.assertEqual(len(selected), 5_500)
        self.assertEqual(len(selected), len(set(selected)))
        for group_start in range(0, 27_500, 550):
            clean = [index for index in selected if group_start <= index < group_start + 50]
            randomized = [
                index for index in selected if group_start + 50 <= index < group_start + 550
            ]
            self.assertEqual(len(clean), 10)
            self.assertEqual(len(randomized), 100)

    def test_train_val_partition_is_disjoint_and_complete(self):
        selected = self.selector(27_500)
        shuffled = selected.copy()
        np.random.default_rng(42).shuffle(shuffled)
        split_index = int(len(shuffled) * 0.99)
        train = set(shuffled[:split_index])
        val = set(shuffled[split_index:])

        self.assertEqual(len(train), 5_445)
        self.assertEqual(len(val), 55)
        self.assertFalse(train & val)
        self.assertEqual(train | val, set(selected))

    def test_selection_is_deterministic(self):
        self.assertEqual(self.selector(27_500), self.selector(27_500))

    def test_unexpected_source_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unexpected source episode count"):
            self.selector(27_499)


if __name__ == "__main__":
    unittest.main()
