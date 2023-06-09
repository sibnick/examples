import numpy
import torch
from torch.utils.data import Sampler
from typing import Optional



class WeightRandomSampler(Sampler[int]):
    data_source_len: int
    replacement: bool

    def __init__(self, seed, data_source_len: int) -> None:
        self.data_source_len = data_source_len
        self.stat_cumsum = None
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    @property
    def num_samples(self) -> int:
        return self.data_source_len

    def __iter__(self):
        for idx in range(self.num_samples // self.values.shape[0]):
            yield from torch.searchsorted(self.stat_cumsum, self.values).tolist()
        yield from torch.searchsorted(self.stat_cumsum, self.values).tolist()[:self.num_samples % self.values.shape[0]]

    def update(self, stat_cumsum):
        self.stat_cumsum = stat_cumsum

    def __len__(self) -> int:
        return self.data_source_len

    def next(self, n):
        self.values = (torch.rand(n) * self.stat_cumsum[-1])
        self.n = n
        return self

class DynamicWeightBatchSampler:
    def __init__(self, ds_len: int, batch_size: int, seed: int, initial_coef: float = 1000.0) -> None:
        self.ds_len = ds_len
        self.batch = None
        self.statistics = initial_coef * torch.ones(ds_len, requires_grad=False).cpu()
        self.count_statistics = torch.zeros(ds_len, dtype=torch.int32, requires_grad=False).cpu()
        self.stat_cumsum = torch.cumsum(self.statistics, 0)

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))

        self.sampler = WeightRandomSampler(seed, ds_len)
        self.batch_size = batch_size
        self.sampler.update(self.stat_cumsum)

    def update_stats(self, loss):
        self.count_statistics[self.batch] += 1
        counts = self.count_statistics[self.batch]
        self.statistics[self.batch] = loss.cpu() #/ torch.sqrt(counts)
        self.stat_cumsum = torch.cumsum(self.statistics, 0)
        self.sampler.update(self.stat_cumsum)

    def __iter__(self):
      sampler_iter = iter(self.sampler)
      while True:
          try:
              self.sampler.next(self.batch_size)
              self.batch = numpy.array([next(sampler_iter) for _ in range(self.batch_size)])
              yield self.batch
          except StopIteration:
              break

    def __len__(self) -> int:
      return self.ds_len // self.batch_size