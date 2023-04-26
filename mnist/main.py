from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import Sampler
from typing import Optional
import collections


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
        for _ in range(self.num_samples // n):
            yield from torch.randperm(n, generator=self.generator).tolist()
        yield from torch.randperm(n, generator=self.generator).tolist()[:self.num_samples % n]

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

    def __init__(self, ds_len: int, batch_size: int, mode=True) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        self.ds_len = ds_len
        self.mode = mode
        #self.statistics = torch.zeros(ds_len, requires_grad=False).to(device)

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))

        self.sampler = MyRandomSampler(ds_len)
        self.batch_size = batch_size

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class LinearRegression(torch.nn.Module):
    def __init__(self, n):
        super(LinearRegression, self).__init__()
        self.n = n #batch size
        self.e = 8 #embeding
        self.w = nn.Parameter(torch.rand((n, self.e), requires_grad=True))
    def forward(self, x):
        x = torch.triu(x, diagonal=1)
        x = x[:, :, None]
        x = x.repeat(1, 1, self.e)
        ww = self.w[:, None, :].repeat(1, self.n, 1)
        w2 = self.w[None, :, :].repeat(self.n, 1, 1)
        w2 = torch.triu(w2, diagonal=1)
        c1 = x * ww - w2
        #c1_loss = torch.sum(c1) + torch.abs(self.n - torch.sum(torch.abs(self.w)))
        return c1


def process_losses(losses, batches):
    diff = losses[1] - losses[0]
    samples = torch.cartesian_prod(diff, diff)
    n = losses[0].shape[0]
    samples = samples.view(n, n, 2)
    L = samples[:, :, 0]/samples[:, :, 1]
    sampler_reg = LinearRegression(n)
    from torch.autograd import Variable
    optimizer = torch.optim.SGD(sampler_reg.parameters(), lr=0.01)
    inputs = Variable(L)
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = sampler_reg(inputs)
        loss = torch.sum(torch.abs(outputs))/(n*n-n)/2 + torch.abs(n - torch.sum(torch.abs(sampler_reg.w)))
        loss.backward()
        optimizer.step()
        print('epoch {}, loss {}'.format(epoch, loss.item()))
    print(L)
    pass


def train(args, model, device, train_loader, optimizer, epoch, sampler):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        losses = {}
        data, target = data.to(device), target.to(device)
        for i in range(0, 2):
            optimizer.zero_grad()
            output = model(data)
            tmp = F.nll_loss(output, target, reduction='none')
            losses[i] = tmp.clone().detach().cpu()
            loss = torch.mean(tmp)
            loss.backward()
            optimizer.step()
        process_losses(losses, sampler.batch)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    return losses


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    sampler = MyBatchSampler(len(dataset1), batch_size=args.batch_size)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_sampler=sampler)#, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, sampler)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
