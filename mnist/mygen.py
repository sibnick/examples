import torch
from torch.utils.data import Sampler
from typing import Optional


class MyRandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    data_source_len: int
    replacement: bool

    def __init__(self, data_source_len: int, num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source_len = data_source_len
        self._num_samples = num_samples
        self.generator = generator
        self.stat_cumsum = None
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return self.data_source_len
        return self._num_samples

    def __iter__(self):
        n = self.data_source_len
        if self.stat_cumsum is None:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=self.generator).tolist()
            yield from torch.randperm(n, generator=self.generator).tolist()[:self.num_samples % n]
        else:
            values = torch.rand(n).cuda() * self.stat_cumsum[n - 1]
            values -= 1
            for _ in range(self.num_samples // n):
                yield from torch.searchsorted(self.stat_cumsum, values).tolist()
            yield from torch.searchsorted(self.stat_cumsum, values).tolist()[:self.num_samples % n]

    def update(self, stat_cumsum):
        self.stat_cumsum = stat_cumsum

    def __len__(self) -> int:
        return self.num_samples


class MyBatchSampler:
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, ds_len: int, batch_size: int, dev, mode=True, alpha=0.9) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        self.alpha = alpha
        self.epoch = 0
        self.ds_len = ds_len
        self.mode = mode
        self.statistics = torch.ones(ds_len, requires_grad=False).to(dev)
        self.stat_cumsum = torch.cumsum(self.statistics, 0).to(dev)

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))

        self.sampler = MyRandomSampler(ds_len)
        self.batch_size = batch_size

    def update_stats(self, loss):
        if self.mode:
            self.statistics[self.batch] = loss * self.alpha + (1 - self.alpha) * self.statistics[self.batch]
            mean_val = torch.mean(loss)
            self.statistics[self.statistics < mean_val / 10] = mean_val / 10

            self.stat_cumsum = torch.cumsum(self.statistics, 0)
            self.sum = self.stat_cumsum[self.ds_len - 1]
            self.sampler.update(self.stat_cumsum)

    def __iter__(self):
        sampler_iter = iter(self.sampler)
        while True:
            try:
                self.batch = [next(sampler_iter) for _ in range(self.batch_size)]
                yield self.batch
            except StopIteration:
                break

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        return len(self.sampler) // self.batch_size  # type: ignore[arg-type]

    def next_epoch(self):
        self.epoch += 1