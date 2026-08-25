from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler

from fastwam.utils.samplers import ResumableEpochSampler


def _rank_batches(*, dataset_length: int, resume_batches: int):
    batches = []
    for rank in range(2):
        sampler = ResumableEpochSampler(
            dataset=range(dataset_length),
            seed=42,
            batch_size=1,
            num_processes=2,
        )
        sampler.set_resume_batch_offset(resume_batches)
        batch_sampler = BatchSampler(
            sampler,
            batch_size=1,
            drop_last=True,
        )
        shard = BatchSamplerShard(
            batch_sampler,
            num_processes=2,
            process_index=rank,
            split_batches=False,
            even_batches=True,
        )
        batches.append(list(shard))
    return batches


def test_two_rank_drop_last_resume_replays_tail_without_padding_drift():
    # Five samples are deliberately not divisible by the global batch of two.
    full = _rank_batches(dataset_length=5, resume_batches=0)
    resumed = _rank_batches(dataset_length=5, resume_batches=1)

    assert all(len(rank_batches) == 2 for rank_batches in full)
    assert resumed == [rank_batches[1:] for rank_batches in full]
