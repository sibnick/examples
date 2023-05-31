from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock

from mnist import mygen
import MyMnist

class EpochData:
    def __init__(self):
        self.epoch = 0
        self.sampler = None

    def __call__(self, idx: int) -> bool:
        if self.sampler.count_statistics[idx] < 3:
            return False
        if self.sampler.statistics[idx] > self.sampler.threshold_max:
            return False
        return True


class ConditionalTransform(torch.nn.Module):

    def __init__(self, impl:torch.nn.Module) -> None:
        super(ConditionalTransform, self).__init__()
        self.impl = impl

    def __call__(self, pic):
        return self.impl(pic)

class ConditionalTransformWithParams(ConditionalTransform):

    def __init__(self, impl, params_transform) -> None:
        super(ConditionalTransformWithParams, self).__init__(impl)
        orig_params = impl.get_params
        self.current_idx = None
        impl.get_params = lambda *args : self.get_params(self.get_current_idx, orig_params, params_transform, *args)

    def get_current_idx(self):
        return self.current_idx

    def set_current_idx(self, idx):
        self.current_idx = idx

    @staticmethod
    def get_params(get_idx, callable, params_transform, *args):
        params = callable(*args)
        return params_transform(get_idx(), *params)

class ConditionalCompose:
    def __init__(self, checker, transforms):
        super(ConditionalCompose, self).__init__()
        self.checker = checker
        self.transforms = transforms

    def __call__(self, idx, img):
        for t in self.transforms:
            if issubclass(type(t), ConditionalTransform):
                if self.checker(idx):
                    if type(t) is ConditionalTransformWithParams:
                        t.set_current_idx(idx)
                    img = t(img)
            else:
                img = t(img)
        return img


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        # self.dropout2 = nn.Dropout(0.37)
        # self.dropout3 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(9216, 1024)
        # self.fc2 = nn.Linear(1024, 1024)
        # self.fc3 = nn.Linear(1024, 10)

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

class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self._forward_impl(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if isinstance(train_loader.batch_sampler, mygen.MyBatchSampler):
            train_loader.batch_sampler.update_stats(epoch, loss.detach().item(), output, target)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


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
    torch.use_deterministic_algorithms(True, warn_only=True)
    import random
    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)

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
    # train_transform = transform
    check_transform = EpochData()
    def affine_params_transform(idx, angle, translations, scale, shear):
        if check_transform.sampler.threshold_min is None:
            return angle, translations, scale, shear
        coef = check_transform.sampler.statistics[idx]
        if coef < check_transform.sampler.threshold_min:
            return angle, translations, scale, shear
        if coef > check_transform.sampler.threshold_max:
            return (0, (0, 0), 1, (0, 0))
        coef = (check_transform.sampler.threshold_max*1.01 - coef) / check_transform.sampler.threshold_max
        coef = coef.detach().item()
        return coef * angle, (coef * translations[0], coef * translations[1]), 1 - coef * (1-scale), (coef * shear[0], coef * shear[1])

    train_transform = ConditionalCompose(check_transform, [
        #                     transforms.RandomRotation(30),
        ConditionalTransform(transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1))),
        # ConditionalTransformWithParams(transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)), affine_params_transform),
        ConditionalTransform(transforms.ColorJitter(brightness=0.2, contrast=0.2)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = MyMnist.MyMnist('../data', train=True, download=True,
                       transform=train_transform, checker=check_transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    sampler = mygen.MyBatchSampler(len(dataset1), dev=device, batch_size=args.batch_size, coef=1000, window=3)
    check_transform.sampler = sampler
    train_loader = torch.utils.data.DataLoader(dataset1, batch_sampler=sampler)
    # train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # model = Net().to(device)
    model = MnistResNet().to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(dataset1), epochs=14)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr) #, weight_decay=1e-6)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        check_transform.epoch = epoch
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
        print("last LR: ", scheduler.get_last_lr())

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()

# import PIL.Image as pil
# idxs = (sampler.count_pearson>4).nonzero().cpu().numpy()
# p = sampler.pearson_statistics[(sampler.count_pearson>4).nonzero().cpu().numpy(), 0].cpu().detach().numpy()
# for idx in  idxs[np.where(p<0.5)]: print(idx, " ", sampler.count_pearson[idx].item(), " ", sampler.count_statistics[idx].item(), " ", sampler.statistics[idx].item(), " ", sampler.pearson_statistics[idx, 0].item());
# idx=1666; img = pil.fromarray(dataset1.data[idx].numpy(), 'L'); img.show(title=dataset1.targets[idx].item()); print(dataset1.targets[idx].item());
# sampler.pearson_statistics[:, 0][sampler.pearson_statistics[:, 0].nonzero()[:, 0]].topk(1000, largest=True)
#
