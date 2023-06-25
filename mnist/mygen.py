import numpy
import numpy as np
import torch
from torch.utils.data import Sampler
from typing import Optional



class WeightRandomSampler(Sampler[int]):
    data_source_len: int
    replacement: bool

    def __init__(self, seed, data_source_len: int, excluded: int = 0) -> None:
        self.data_source_len = data_source_len
        self.excluded = excluded
        self.stat_cumsum = None
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    @property
    def num_samples(self) -> int:
        return self.data_source_len

    def __iter__(self):
        for idx in range((self.num_samples - self.excluded)// self.values.shape[0]):
            yield from torch.searchsorted(self.stat_cumsum, self.values).tolist()
        yield from torch.searchsorted(self.stat_cumsum, self.values).tolist()[:self.num_samples % self.values.shape[0]]

    def update(self, stat_cumsum):
        self.stat_cumsum = stat_cumsum

    def __len__(self) -> int:
        return self.data_source_len - self.excluded

    def next(self, n):
        self.values = (torch.rand(n) * self.stat_cumsum[-1])
        self.n = n
        return self

class DynamicWeightBatchSampler:
    def __init__(self, ds_len: int, batch_size: int, seed: int, exclude_ids = [], initial_coef: float = 1000.0) -> None:
        self.ds_len = ds_len
        self.batch = None
        self.target_ids = None
        self.statistics = initial_coef * torch.ones(ds_len, requires_grad=False).cpu()
        self.exclude_ids = numpy.array(exclude_ids)
        if len(exclude_ids) > 0:
            print("Exclude len(ids): ", len(exclude_ids))
            self.statistics[exclude_ids] = -0.0
        self.count_statistics = torch.zeros(ds_len, dtype=torch.int32, requires_grad=False).cpu()
        self.stat_cumsum = torch.cumsum(self.statistics, 0)

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))

        self.sampler = WeightRandomSampler(seed, ds_len, len(exclude_ids))
        self.batch_size = batch_size
        self.sampler.update(self.stat_cumsum)
        self.window = 5
        self.prev_batch = None
        self.prev_loss = None
        self.prev_output = None
        self.pearson_statistics = torch.zeros((ds_len, 6), requires_grad=False).cpu()
        self.pearson_statistics[:, 0] = 1

    def update_stats(self, loss, output):
        self.count_statistics[self.batch] += 1
        counts = self.count_statistics[self.batch]
        loss = loss.cpu()
        self.statistics[self.batch] = loss #/ torch.sqrt(counts)
        self.norm_weights()
        self.stat_cumsum = torch.cumsum(self.statistics, 0)
        self.sampler.update(self.stat_cumsum)
        output_max = output.max(dim=1)[0]
        if self.prev_batch is not None:
            self.pearson_corr(loss/(1e-3 + self.prev_loss), self.prev_batch, self.prev_output)
        self.prev_loss = loss
        self.prev_output = output_max.detach()
        self.prev_batch = self.batch.copy()

    def pearson_corr(self, loss, batch_idx, xn1):
        n = self.window
        yn1 = loss
        n1 = 1.0 / (1 + n)
        data = self.pearson_statistics[batch_idx]
        mean_xn1 = data[:, 1] + n1 * (xn1[:] - data[:, 1])
        mean_yn1 = data[:, 2] + n1 * (yn1 - data[:, 2])
        nn1 = data[:, 3] + (xn1 - data[:, 1]) * (yn1 - mean_yn1)
        dn1 = data[:, 4] + (xn1 - data[:, 1]) * (xn1 - mean_xn1)
        en1 = data[:, 5] + (yn1 - data[:, 2]) * (yn1 - mean_yn1)
        r = nn1 / (1e-6 + torch.sqrt(1e-6 + dn1 * en1))
        if (torch.sum(torch.isnan(r)) > 0):
            r = 1
            print("!")
        data = torch.zeros((batch_idx.shape[0], 6), device=self.pearson_statistics.device, requires_grad=False)
        data[:, 0] = r
        data[:, 1] = mean_xn1
        data[:, 2] = mean_yn1
        data[:, 3] = nn1
        data[:, 4] = dn1
        data[:, 5] = en1
        self.pearson_statistics[batch_idx] = data
        return r

    def __iter__(self):
      sampler_iter = iter(self.sampler)
      while True:
          try:
              self.sampler.next(self.batch_size)
              self.batch = numpy.array([next(sampler_iter) for _ in range(self.batch_size)])
              # if len(self.exclude_ids) > 0:
              #   intersect = numpy.intersect1d(self.batch, self.exclude_ids)
              #   if intersect:
              #         print("Intersection found! ", self.batch, " in ", self.exclude_ids)
              yield self.batch
          except StopIteration:
              break

    def __len__(self) -> int:
      return (self.ds_len - len(self.exclude_ids)) // self.batch_size

    def norm_weights(self):
        sum = torch.zeros((10), device=self.statistics.device, requires_grad=False)
        i = 0
        for cls in self.target_ids:
            sum[i] = self.statistics[cls].sum()
            i += 1
        #print(sum.cpu())
        avg = sum / sum.mean()
        i = 0
        for cls in self.target_ids:
            self.statistics[cls] /= avg[i]
            i += 1
