import torch
import numpy as np
import math
from pathlib import Path
from typing import List, Literal, Dict, Optional, Any, DefaultDict, Callable
from tqdm import tqdm
from .lerobot.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset

from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from fastwam.utils.logging_config import get_logger
from .processors.base_processor import BaseProcessor

logger = get_logger(__name__)

MAX_GETITEM_ATTEMPT = 5

class BaseLerobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs: List[str],

        # shapes
        shape_meta: Dict[str, Any],
        action_size: int = 1, 
        past_action_size: int = 0, # Excludes the current frame
        obs_size: int = 1, # should be 
        past_obs_size: int = 0,

        # train vs val
        val_set_proportion: float = 0.05, 
        is_training_set: bool = False,
        seed: int = 42,

        # sampling
        global_sample_stride: int = 1,
        episode_selector: Optional[Callable[[int], List[int]]] = None,
        strict_getitem: bool = False,
        return_metadata: bool = False,
    ):
        assert len(dataset_dirs) > 0, "At least one dataset directory is required"
        assert past_action_size == 0
        assert past_obs_size == 0
        assert action_size == obs_size - 1, "In this dataset, action_size should be obs_size - 1"
        
        self.dataset_dirs = dataset_dirs
        self.shape_meta = shape_meta
        self.action_size = action_size
        self.past_action_size = past_action_size
        self.obs_size = obs_size
        self.processor = None  # Will be set externally
        self.strict_getitem = bool(strict_getitem)
        self.return_metadata = bool(return_metadata)
        metas = []
        for ds_dir in dataset_dirs:
            ds_root = Path(ds_dir)
            repo_id = ds_dir
            meta = LeRobotDatasetMetadata(repo_id=repo_id, root=ds_root)
            metas.append(meta)

        fps_list = [m.fps for m in metas]
        assert len(set(fps_list)) == 1, f"All dataset_dirs must have the same fps, got {fps_list}"
        fps = fps_list[0]
        
        self.global_sample_stride = global_sample_stride

        self.val_set_proportion = val_set_proportion
        self.is_training_set = is_training_set

        self.image_meta = shape_meta["images"]
        self.state_meta = shape_meta["state"]
        self.action_meta = shape_meta["action"]

        delta_timestamps = {}
        for meta in self.image_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"observation.images.{key}" if key != "default" else "observation.images"
            delta_timestamps[meta["lerobot_key"]] = [
                (t * global_sample_stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)
            ]
        
        for meta in self.state_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"observation.state.{key}" if key != "default" else "observation.state"
            delta_timestamps[meta["lerobot_key"]] = [
                (t * global_sample_stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)
            ]
        
        for meta in self.action_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"action.{key}" if key != "default" else "action"
            delta_timestamps[meta["lerobot_key"]] = [(t * global_sample_stride) / fps for t in range(-past_action_size, -past_action_size + action_size)]

        episodes = {}
        if episode_selector is None:
            # Preserve the original full-data train/val path exactly.
            if val_set_proportion < 1e-6:
                for meta in metas:
                    episodes.update({meta.repo_id: list(range(meta.total_episodes))})
            else:
                for meta in metas:
                    split_idx = int(meta.total_episodes * (1 - val_set_proportion))
                    # random shuffle episode indices before splitting
                    episode_indices = list(range(meta.total_episodes))
                    rng = np.random.default_rng(seed)
                    rng.shuffle(episode_indices)
                    if self.is_training_set:
                        episodes.update({meta.repo_id: [episode_indices[i] for i in range(split_idx)]})
                    else:
                        episodes.update({meta.repo_id: [episode_indices[i] for i in range(split_idx, meta.total_episodes)]})
        else:
            for meta in metas:
                episode_indices = list(episode_selector(meta.total_episodes))
                if not episode_indices:
                    raise ValueError(f"Episode selector returned no episodes for {meta.repo_id}")
                if len(episode_indices) != len(set(episode_indices)):
                    raise ValueError(f"Episode selector returned duplicate indices for {meta.repo_id}")
                invalid_indices = [
                    index for index in episode_indices if index < 0 or index >= meta.total_episodes
                ]
                if invalid_indices:
                    raise ValueError(
                        f"Episode selector returned out-of-range indices for {meta.repo_id}: "
                        f"{invalid_indices[:10]}"
                    )
                logger.info(
                    "Episode selector chose %d/%d source episodes for %s",
                    len(episode_indices),
                    meta.total_episodes,
                    meta.repo_id,
                )

                if val_set_proportion >= 1e-6:
                    split_idx = int(len(episode_indices) * (1 - val_set_proportion))
                    # Shuffle after subset selection so train and val partition the same subset.
                    rng = np.random.default_rng(seed)
                    rng.shuffle(episode_indices)
                    if self.is_training_set:
                        episode_indices = episode_indices[:split_idx]
                    else:
                        episode_indices = episode_indices[split_idx:]

                episodes[meta.repo_id] = episode_indices
                logger.info(
                    "Using %d episodes for %s split of %s",
                    len(episode_indices),
                    "train" if self.is_training_set else "validation",
                    meta.repo_id,
                )

        self.multi_dataset = MultiLeRobotDataset(
            dataset_dirs=self.dataset_dirs,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
        )
        
        # HACK: lerobot 3.0 will fix this
        episode_data_index = []
        end_index = 0
        for dataset in self.multi_dataset._datasets:
            multi_episode_data_index = {
                "from": dataset.episode_data_index["from"] + end_index,
                "to": dataset.episode_data_index["to"] + end_index,
            }
            episode_data_index.append(multi_episode_data_index)
            end_index = multi_episode_data_index["to"][-1]

        self.episode_data_index = {
            "from": torch.cat([dataset["from"] for dataset in episode_data_index]),
            "to": torch.cat([dataset["to"] for dataset in episode_data_index]),
        }

    def _get_action(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        action: torch.Tensor = lerobot_sample[lerobot_key] # [T, action_dim]
        if action.ndim == 1: # for shape of 1, like gripper
            action = action.unsqueeze(-1)
        assert action.shape[-1] == raw_shape, f"Action '{key}' shape {action.shape[-1]} mismatch with meta {raw_shape}."
        return action

    def _get_state(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        state: torch.Tensor = lerobot_sample[lerobot_key]
        if state.ndim == 1: # for shape of 1, like gripper
            state = state.unsqueeze(-1)
        # state = state[..., :-1, :]  # use state_{t} as observation_t
        assert state.shape[-1] == raw_shape, f"State '{key}' shape {state.shape[-1]} mismatch with meta {raw_shape}."
        return state
    
    def _get_image(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        image: torch.Tensor = lerobot_sample[lerobot_key]
        if image.ndim == 3: # time dim will lost when obs_size is 1
            image = image.unsqueeze(0)        
        image = (image * 255).to(torch.uint8) # (1, 3, H, W)
        # For config simplication
        # assert image.shape[1:] == raw_shape, f"Image '{key}' shape {image.shape[1:]} mismatch with {raw_shape}."
        return image
    
    def _split_lerobot_sample(self, lerobot_sample) -> Dict[str, Any]:
        return lerobot_sample
    
    def _get_episode_data(self, episode_idx):
        lerobot_sample = self.multi_dataset.get_episode_data(episode_idx)
        lerobot_sample = self._split_lerobot_sample(lerobot_sample)
        state, action = {}, {}
        for meta in self.state_meta:
            s = self._get_state(meta, lerobot_sample)
            state[meta["key"]] = s.unsqueeze(1).float()
        for meta in self.action_meta:
            a = self._get_action(meta, lerobot_sample)
            a = sliding_window_with_replication(a, self.action_size)
            action[meta["key"]] = a.float()
        return {"action": action, "state": state}

    def _set_return_images(self, flag: bool):
        self.return_images = flag
        self.multi_dataset.set_during_training(flag)

    def __len__(self):
        return self.multi_dataset.num_frames

    @staticmethod
    def _json_scalar(value: Any, field_name: str, *, allow_none: bool = False) -> Any:
        """Convert a scalar source value to a strict JSON primitive."""
        if value is None:
            if allow_none:
                return None
            raise ValueError(f"Required metadata field '{field_name}' is None")

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"Metadata field '{field_name}' must be scalar, got tensor shape {tuple(value.shape)}"
                )
            value = value.detach().cpu().item()
        elif isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError(
                    f"Metadata field '{field_name}' must be scalar, got array shape {value.shape}"
                )
            value = value.reshape(()).item()
        elif isinstance(value, np.generic):
            value = value.item()

        if not isinstance(value, (str, bool, int, float)):
            raise TypeError(
                f"Metadata field '{field_name}' must be a JSON scalar, got {type(value).__name__}"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Metadata field '{field_name}' must be finite, got {value}")
        return value

    @classmethod
    def _json_int(cls, value: Any, field_name: str, *, allow_none: bool = False) -> Optional[int]:
        value = cls._json_scalar(value, field_name, allow_none=allow_none)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Metadata field '{field_name}' must be an integer, got {value!r}")
        return value

    @classmethod
    def _json_string(cls, value: Any, field_name: str) -> str:
        value = cls._json_scalar(value, field_name)
        if not isinstance(value, str) or not value:
            raise TypeError(f"Metadata field '{field_name}' must be a non-empty string, got {value!r}")
        return value

    def _dataset_identity(self, dataset_index: int) -> tuple[str, str]:
        dataset_index = self._json_int(dataset_index, "dataset_index")
        if dataset_index is None or dataset_index < 0:
            raise IndexError(f"dataset_index must be non-negative, got {dataset_index}")

        # MultiLeRobotDataset owns the ordered names used to build _datasets.
        # Prefer that reliable structure: some bundled LeRobot versions have a
        # broken repo_index_to_id property implementation.
        ds_names = getattr(self.multi_dataset, "ds_names", None)
        if ds_names is not None:
            if not isinstance(ds_names, (list, tuple)):
                raise TypeError(
                    f"MultiLeRobotDataset.ds_names must be a sequence, got {type(ds_names).__name__}"
                )
            if len(ds_names) != len(self.multi_dataset._datasets):
                raise ValueError(
                    "MultiLeRobotDataset.ds_names/_datasets length mismatch: "
                    f"{len(ds_names)} != {len(self.multi_dataset._datasets)}"
                )
            if dataset_index >= len(ds_names):
                raise IndexError(
                    f"dataset_index {dataset_index} is out of bounds for {len(ds_names)} datasets"
                )
            raw_dataset_id = ds_names[dataset_index]
        else:
            # Compatibility path for lightweight fakes and older implementations.
            try:
                repo_index_to_id = self.multi_dataset.repo_index_to_id
            except Exception as exc:
                raise RuntimeError(
                    "Could not resolve dataset identity from ds_names or repo_index_to_id"
                ) from exc
            if dataset_index not in repo_index_to_id:
                raise KeyError(
                    f"dataset_index {dataset_index} is absent from repo_index_to_id"
                )
            raw_dataset_id = repo_index_to_id[dataset_index]

        dataset_id = self._json_string(raw_dataset_id, "dataset_id")
        dataset_name = Path(dataset_id).name
        if not dataset_name:
            raise ValueError(f"Could not derive dataset_name from dataset_id {dataset_id!r}")
        return dataset_id, dataset_name

    def dataset_index_ranges(self) -> List[Dict[str, Any]]:
        """Return JSON-safe global index ranges without materializing any samples."""
        ranges = []
        start = 0
        for dataset_index, dataset in enumerate(self.multi_dataset._datasets):
            population = self._json_int(dataset.num_frames, "population")
            if population is None or population < 0:
                raise ValueError(f"Dataset population must be non-negative, got {population}")
            dataset_id, dataset_name = self._dataset_identity(dataset_index)
            stop = start + population
            ranges.append(
                {
                    "dataset_index": dataset_index,
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "start": start,
                    "stop": stop,
                    "population": population,
                }
            )
            start = stop
        if start != len(self):
            raise RuntimeError(
                f"Dataset ranges cover {start} samples, but MultiLeRobotDataset reports {len(self)}"
            )
        return ranges

    def dataset_task_table(self, dataset_index: int) -> Dict[int, str]:
        """Return a copied local task-index table for one source dataset.

        LeRobot task indices are local to each member of MultiLeRobotDataset,
        so callers must select this table using the sample's dataset_index.
        This reads in-memory metadata only and never materializes a sample.
        """
        dataset_index = self._json_int(dataset_index, "dataset_index")
        if dataset_index is None or dataset_index < 0:
            raise IndexError(f"dataset_index must be non-negative, got {dataset_index}")
        datasets = self.multi_dataset._datasets
        if dataset_index >= len(datasets):
            raise IndexError(
                f"dataset_index {dataset_index} is out of bounds for {len(datasets)} datasets"
            )
        source_tasks = getattr(getattr(datasets[dataset_index], "meta", None), "tasks", None)
        if not isinstance(source_tasks, dict):
            raise TypeError(
                f"Source dataset {dataset_index} meta.tasks must be a dict, "
                f"got {type(source_tasks).__name__}"
            )

        task_table: Dict[int, str] = {}
        for raw_task_index, raw_task in source_tasks.items():
            task_index = self._json_int(raw_task_index, "task_index")
            if task_index is None or task_index < 0:
                raise ValueError(f"task_index must be non-negative, got {task_index}")
            task = self._json_string(raw_task, "task")
            if task_index in task_table:
                raise ValueError(f"Duplicate normalized task_index {task_index}")
            task_table[task_index] = task
        if not task_table:
            raise ValueError(f"Source dataset {dataset_index} has an empty task table")
        return task_table

    def _build_metadata(
        self,
        requested_sample_idx: int,
        source_sample_idx: int,
        lerobot_sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        def required(field_name: str) -> Any:
            if field_name not in lerobot_sample:
                raise KeyError(f"Raw LeRobot sample is missing required metadata field '{field_name}'")
            return lerobot_sample[field_name]

        requested_sample_idx = self._json_int(requested_sample_idx, "requested_sample_idx")
        source_sample_idx = self._json_int(source_sample_idx, "source_sample_idx")
        dataset_index = self._json_int(required("dataset_index"), "dataset_index")
        dataset_id, dataset_name = self._dataset_identity(dataset_index)
        task = self._json_string(required("task"), "task")

        timestamp = self._json_scalar(
            lerobot_sample.get("timestamp"), "timestamp", allow_none=True
        )
        if timestamp is not None and (
            isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
        ):
            raise TypeError(f"Metadata field 'timestamp' must be numeric, got {timestamp!r}")

        metadata = {
            "requested_sample_idx": requested_sample_idx,
            "source_sample_idx": source_sample_idx,
            "dataset_index": dataset_index,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "episode_index": self._json_int(required("episode_index"), "episode_index"),
            "frame_index": self._json_int(required("frame_index"), "frame_index"),
            "task_index": self._json_int(required("task_index"), "task_index"),
            "task": task,
            "timestamp": timestamp,
            "source_index": self._json_int(
                lerobot_sample.get("index"), "source_index", allow_none=True
            ),
        }
        if self.strict_getitem and requested_sample_idx != source_sample_idx:
            raise AssertionError(
                "Strict dataset access changed the requested sample index: "
                f"requested={requested_sample_idx}, source={source_sample_idx}"
            )
        return metadata

    def _get_additional_data(self, sample, lerobot_sample):
        return sample

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds {len(self)}.")
        if self.strict_getitem and idx < 0:
            raise IndexError(
                f"Strict dataset access requires a non-negative index, got {idx}."
            )

        if self.strict_getitem:
            # Collector mode: preserve the requested identity and original exception.
            sample_idx = idx
            lerobot_sample = self.multi_dataset[sample_idx]
            lerobot_sample = self._split_lerobot_sample(lerobot_sample)
            if sample_idx != idx:
                raise AssertionError(
                    "Strict dataset access changed the requested sample index: "
                    f"requested={idx}, source={sample_idx}"
                )
        else:
            # Preserve the legacy random-retry training behavior exactly.
            sample_idx = idx
            attempt = 0
            last_exception: Optional[Exception] = None
            while attempt < MAX_GETITEM_ATTEMPT:
                try:
                    lerobot_sample = self.multi_dataset[sample_idx]
                    lerobot_sample = self._split_lerobot_sample(lerobot_sample)
                    break
                except Exception as err:
                    attempt += 1
                    last_exception = err
                    logger.warning(
                        f"Error loading sample {sample_idx} (attempt {attempt}). "
                        "Retrying with a random index. "
                        f"Error: {err}"
                    )
                    sample_idx = np.random.randint(len(self))
                    print(traceback.format_exc())
            else:
                raise RuntimeError(
                    f"Failed to load a valid sample after {MAX_GETITEM_ATTEMPT} attempts "
                    f"for index {idx}."
                ) from last_exception

        metadata = None
        if self.return_metadata:
            # Capture provenance from the raw LeRobot record before any processor can
            # drop or transform those source fields.
            metadata = self._build_metadata(idx, sample_idx, lerobot_sample)

        # Get data from lerobot, organized in nested dict
        sample = {
            "idx": sample_idx,
            "task": lerobot_sample["task"],
            "action": {},
            "state": {},
            "images": {},
        }
        for meta in self.state_meta:
            sample["state"][meta["key"]] = self._get_state(meta, lerobot_sample)

        for meta in self.action_meta:
            sample["action"][meta["key"]] = self._get_action(meta, lerobot_sample)

        for meta in self.image_meta:
            sample["images"][meta["key"]] = self._get_image(meta, lerobot_sample)

        sample["action_is_pad"] = lerobot_sample[f"{self.action_meta[0]['lerobot_key']}_is_pad"]
        sample["state_is_pad"] = lerobot_sample[f"{self.state_meta[0]['lerobot_key']}_is_pad"]
        sample["image_is_pad"] = lerobot_sample[f"{self.image_meta[0]['lerobot_key']}_is_pad"]

        sample = self._get_additional_data(sample, lerobot_sample)

        for key in lerobot_sample:
            if key not in sample and "observation" not in key and "action" not in key:
                sample[key] = lerobot_sample[key]

        # Preprocess the sample using the processor
        # for quick data loading
        if self.processor is not None:
            sample = self.processor.preprocess(sample)

        if metadata is not None:
            sample["metadata"] = metadata

        return sample

    def set_processor(self, processor: BaseProcessor):
        """Set processor instance from external initialization."""
        self.processor = processor
        if self.is_training_set:
            self.processor.train()
        else:
            self.processor.eval()
        return self

    def get_dataset_stats(self, preprocessor: BaseProcessor):
        state_min = DefaultDict(list)
        state_max = DefaultDict(list)
        state_mean = DefaultDict(list)
        state_var = DefaultDict(list)
        state_q01 = DefaultDict(list)
        state_q99 = DefaultDict(list)

        action_min = DefaultDict(list)
        action_max = DefaultDict(list)
        action_mean = DefaultDict(list)
        action_var = DefaultDict(list)
        action_q01 = DefaultDict(list)
        action_q99 = DefaultDict(list)

        episodes_num = self.multi_dataset.num_episodes
        
        def process_episode(episode_idx):
            batch = self._get_episode_data(episode_idx) 
            batch = preprocessor.action_state_transform(batch)
            return batch
        
        multi_thread = True
        if not multi_thread:
            for episode_idx in tqdm(range(episodes_num), desc="Iterating dataset to get normalization"):
                batch = process_episode(episode_idx)
                for meta in self.state_meta:
                    key = meta["key"]
                    cur_state: torch.Tensor = batch["state"][key] # (B, T, dim)
                    state_min[key].append(cur_state.amin(0))
                    state_max[key].append(cur_state.amax(0))
                    state_mean[key].append(cur_state.mean(0))
                    state_var[key].append(cur_state.var(0))
                    state_q01[key].append(torch.quantile(cur_state, 0.01, dim=0, keepdim=False))
                    state_q99[key].append(torch.quantile(cur_state, 0.99, dim=0, keepdim=False))
                for meta in self.action_meta:
                    key = meta["key"]
                    cur_action: torch.Tensor = batch["action"][key] # (B, T, dim)
                    action_min[key].append(cur_action.amin(0))
                    action_max[key].append(cur_action.amax(0))
                    action_mean[key].append(cur_action.mean(0))
                    action_var[key].append(cur_action.var(0))
                    action_q01[key].append(torch.quantile(cur_action, 0.01, dim=0, keepdim=False))
                    action_q99[key].append(torch.quantile(cur_action, 0.99, dim=0, keepdim=False))
        
        else:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_episode, num) for num in range(episodes_num)]
                
                for future in tqdm(as_completed(futures), total=episodes_num, desc="Iterating dataset to get normalization"):
                    try:
                        batch = future.result()
                        for meta in self.state_meta:
                            key = meta["key"]
                            cur_state: torch.Tensor = batch["state"][key] # (B, T, dim)
                            state_min[key].append(cur_state.amin(0))
                            state_max[key].append(cur_state.amax(0))
                            state_mean[key].append(cur_state.mean(0))
                            state_var[key].append(cur_state.var(0))
                            state_q01[key].append(torch.quantile(cur_state, 0.01, dim=0, keepdim=False))
                            state_q99[key].append(torch.quantile(cur_state, 0.99, dim=0, keepdim=False))

                        for meta in self.action_meta:
                            key = meta["key"]
                            cur_action: torch.Tensor = batch["action"][key] # (B, T, dim)
                            action_min[key].append(cur_action.amin(0))
                            action_max[key].append(cur_action.amax(0))
                            action_mean[key].append(cur_action.mean(0))
                            action_var[key].append(cur_action.var(0))
                            action_q01[key].append(torch.quantile(cur_action, 0.01, dim=0, keepdim=False))
                            action_q99[key].append(torch.quantile(cur_action, 0.99, dim=0, keepdim=False))

                    except Exception as e:
                        logger.error(f"Error processing episode: {e}")
                        print(traceback.format_exc())
                        raise e

        # assume that each minibatch has equal number of samples
        def get_mean_std(means, vars):
            means = torch.stack(means)
            vars = torch.stack(vars)
            stepwise_mean = means.mean(0)
            stepwise_std = (vars + (means - stepwise_mean) ** 2).mean(0).sqrt()
            global_mean = means.mean((0, 1))
            global_std = (vars + (means - global_mean) ** 2).mean((0, 1)).sqrt()
            return stepwise_mean, stepwise_std, global_mean, global_std

        stats = {"state": DefaultDict(dict), "action": DefaultDict(dict), "num_episodes": episodes_num, "num_transition": self.multi_dataset.num_frames}
        for meta in self.state_meta:
            key = meta["key"]
            stats["state"][key]["stepwise_min"] = torch.stack(state_min[key]).amin(0)
            stats["state"][key]["stepwise_max"] = torch.stack(state_max[key]).amax(0)
            stats["state"][key]["global_min"] = stats["state"][key]["stepwise_min"].amin(0)
            stats["state"][key]["global_max"] = stats["state"][key]["stepwise_max"].amax(0)
            stats["state"][key]["stepwise_q01"] = torch.stack(state_q01[key]).amin(0)
            stats["state"][key]["stepwise_q99"] = torch.stack(state_q99[key]).amax(0)
            stats["state"][key]["global_q01"] = stats["state"][key]["stepwise_q01"].amin(0)
            stats["state"][key]["global_q99"] = stats["state"][key]["stepwise_q99"].amax(0)
            (
                stats["state"][key]["stepwise_mean"],
                stats["state"][key]["stepwise_std"],
                stats["state"][key]["global_mean"],
                stats["state"][key]["global_std"],
            ) = get_mean_std(state_mean[key], state_var[key])

        for meta in self.action_meta:
            key = meta["key"]
            stats["action"][key]["stepwise_min"] = torch.stack(action_min[key]).amin(0)
            stats["action"][key]["stepwise_max"] = torch.stack(action_max[key]).amax(0)
            stats["action"][key]["global_min"] = stats["action"][key]["stepwise_min"].amin(0)
            stats["action"][key]["global_max"] = stats["action"][key]["stepwise_max"].amax(0)
            stats["action"][key]["stepwise_q01"] = torch.stack(action_q01[key]).amin(0)
            stats["action"][key]["stepwise_q99"] = torch.stack(action_q99[key]).amax(0)
            stats["action"][key]["global_q01"] = stats["action"][key]["stepwise_q01"].amin(0)
            stats["action"][key]["global_q99"] = stats["action"][key]["stepwise_q99"].amax(0)
            (
                stats["action"][key]["stepwise_mean"], 
                stats["action"][key]["stepwise_std"], 
                stats["action"][key]["global_mean"], 
                stats["action"][key]["global_std"],
            ) = get_mean_std(action_mean[key], action_var[key])

        return stats


def sliding_window_with_replication(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Construct a sliding-window tensor from the input tensor x (shape: [N, D]).
    The output shape is [N, window_size, D].
    
    For each starting index i:
        out[i, j, :] =
            x[i + j, :]      if i + j < N
            x[-1, :]         otherwise (replicate the last row when out of bounds)
    
    Args:
        x (torch.Tensor): Input tensor of shape [N, D]
        window_size (int): Size of the sliding window
    
    Returns:
        torch.Tensor: Tensor of shape [N, window_size, D]
    """
    assert x.dim() == 2
    assert window_size > 0
    
    N, D = x.shape
    
    # shape [N, window_size]
    # indices[i, j] = i + j
    i_indices = torch.arange(N).unsqueeze(1)            # [N, 1]
    j_indices = torch.arange(window_size).unsqueeze(0)  # [1, window_size]
    indices = i_indices + j_indices                     # [N, window_size]

    # N-1
    # torch.clamp  [0, N-1]
    clamped_indices = torch.clamp(indices, min=0, max=N - 1)

    # clamped_indices [N, window_size]，x [N, D]
    # out[i, j, :] = x[clamped_indices[i, j], :]
    out = x[clamped_indices]  # [N, window_size, D]

    return out
