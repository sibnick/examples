import numpy
import torch
from torch.utils.data import Sampler
from typing import Optional


class MyRandomSampler(Sampler[int]):
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
        for idx in range(self.num_samples // self.values.shape[0]):
            yield from torch.searchsorted(self.stat_cumsum, self.values).tolist()
        yield from torch.searchsorted(self.stat_cumsum, self.values).tolist()[:self.num_samples % self.values.shape[0]]

    def update(self, stat_cumsum):
        self.stat_cumsum = stat_cumsum

    def __len__(self) -> int:
        return self.num_samples

    def next(self, n):
        self.values = torch.rand(n).cuda() * self.stat_cumsum[self.data_source_len - 1]
        self.n = n
        return self


def calc_coef(epoch, coef):
    coef = 10 * coef / epoch
    coef = max(10, coef)
    coef2 = 0.5 - 1 / coef
    return coef, coef2

class MyBatchSampler:
    def __init__(self, ds_len: int, batch_size: int, bptt: int, dev, seq_len=35, window=5, coef=1000, calc_coef_func = calc_coef) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        self.ds_len = ds_len
        self.window = window
        self.coef = coef
        self.prev_batch = None
        self.seq_len = seq_len
        self.calc_coef_func = calc_coef_func
        self.statistics = coef*torch.ones(ds_len, requires_grad=False).to(dev)
        self.pearson_statistics = torch.zeros((ds_len, 6), requires_grad=False).to(dev)
        self.pearson_statistics[:, 0] = 1
        self.count_pearson = torch.zeros((ds_len), dtype=torch.int32, requires_grad=False).to(dev)
        self.count_statistics = torch.zeros(ds_len, dtype=torch.int32, requires_grad=False).to(dev)
        self.stat_cumsum = torch.cumsum(self.statistics, 0).to(dev)

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))

        self.sampler = MyRandomSampler(ds_len - bptt)
        self.batch_size = batch_size
        self.sampler.update(self.stat_cumsum)

    def update_stats(self, epoch, loss, output, target):
        coef, coef2 = self.calc_coef_func(epoch, self.coef)
        output = torch.softmax(output, dim=1).detach()
        # max_idx = torch.argmax(output, dim=1, keepdim=True)
        # output.scatter_(1, max_idx, 0)
        output.scatter_(1, target[:, None], 0)
        output_max = output.max(dim=1)[0]
        tmp_corr = (torch.sigmoid(output_max) - coef2) * coef
        tmp_corr = tmp_corr.view(self.batch_size, -1)
        for i in range(self.batch.shape[0]):
            idx = self.batch[i]
            corr = 1 + self.pearson_statistics[idx: idx + self.seq_len, 0]
            self.statistics[idx: idx + self.seq_len] = corr * tmp_corr[i]

        if self.prev_batch is not None:
            for i in range(self.batch.shape[0]):
                self.pearson_corr(loss/(1e-6 + self.prev_loss), self.prev_batch[i], self.prev_output[i * self.seq_len : (i+1)*self.seq_len])
        self.prev_loss = loss
        self.prev_batch = self.batch.copy()
        self.prev_output = output_max.detach()

        self.stat_cumsum = torch.cumsum(self.statistics, 0)
        self.count_statistics[self.batch] += 1

        self.sum = self.stat_cumsum[self.ds_len - 1]
        self.sampler.update(self.stat_cumsum)


    def pearson_corr(self, loss, batch_idx, xn1):
        n = self.window
        yn1 = loss
        n1 = 1.0 / (1 + n)
        data = self.pearson_statistics[batch_idx : batch_idx + self.seq_len]
        mean_xn1 = data[:, 1] + n1 * (xn1[:] - data[:, 1])
        mean_yn1 = data[:, 2] + n1 * (yn1 - data[:, 2])
        nn1 = data[:, 3] + (xn1 - data[:, 1]) * (yn1 - mean_yn1)
        dn1 = data[:, 4] + (xn1 - data[:, 1]) * (xn1 - mean_xn1)
        en1 = data[:, 5] + (yn1 - data[:, 2]) * (yn1 - mean_yn1)
        r = nn1 / (1e-6 + torch.sqrt(1e-6 + dn1 * en1))
        if (torch.sum(torch.isnan(r))>0):
            r = 1
            print("!")
        data = torch.zeros((self.seq_len, 6), device=self.pearson_statistics.device, requires_grad=False)
        data[:, 0] = r
        data[:, 1] = mean_xn1
        data[:, 2] = mean_yn1
        data[:, 3] = nn1
        data[:, 4] = dn1
        data[:, 5] = en1
        self.pearson_statistics[batch_idx : batch_idx + self.seq_len] = data
        return r
    def __iter__(self):
        sampler_iter = iter(self.sampler)
        while True:
            try:
                self.sampler.next(self.batch_size)
                self.batch = numpy.array([next(sampler_iter) for _ in range(self.batch_size)])
                yield self.batch
            except StopIteration:
                break

    def next(self):
        sampler_iter = iter(self.sampler)
        self.sampler.next(self.batch_size)
        self.batch = numpy.array([next(sampler_iter) for _ in range(self.batch_size)])
        return self.batch

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        return len(self.sampler) // self.batch_size  # type: ignore[arg-type]

    def next_epoch(self):
        self.epoch += 1