from __future__ import print_function
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import Sampler
import mygen

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
    def __init__(self, n, dev):
        super(LinearRegression, self).__init__()
        self.n = n #batch size
        self.e = 1 #embeding
        self.w = nn.Parameter(torch.rand((n, self.e), requires_grad=True, device=dev))
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


def process_losses(losses):
    dev = losses[0].device
    diff = losses[1] - losses[0]
    samples = torch.cartesian_prod(diff, diff)
    n = losses[0].shape[0]
    samples = samples.view(n, n, 2)
    L = samples[:, :, 0]/(samples[:, :, 1] + 1e-6)
    sampler_reg = LinearRegression(n, dev)
    from torch.autograd import Variable
    optimizer = torch.optim.SGD(sampler_reg.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    inputs = Variable(L.to(dev))
    prev = 1e+10
    initL = 0
    weight = sampler_reg.w.clone()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = sampler_reg(inputs)
        loss = torch.sum(torch.abs(outputs))/(n*n-n)/2 + torch.abs(n - torch.sum(torch.abs(sampler_reg.w)))
        if math.isnan(loss.item()):
            print("!!!!")
        if epoch == 0:
            initL = loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        cur = loss.item()
        if (cur > prev and epoch > 25) or math.isnan(cur):
            break
        if (cur < prev):
            prev = cur
            weight = sampler_reg.w.clone()
    #print("Stop on losses: {} -> {} with LR {} after {}".format(initL, cur, scheduler.get_lr(), epoch))
    k = 1/torch.sum(torch.abs(weight), axis=1)
    return k


def train(args, model, device, train_loader, optimizer, epoch, sampler):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        losses = {}
        data, target = data.to(device), target.to(device)
        for i in range(0, 2):
            optimizer.zero_grad()
            output = model(data)
            output = F.log_softmax(output, dim=1)
            tmp = F.nll_loss(output, target, reduction='none')
            losses[i] = tmp.clone()
            loss = torch.mean(tmp)
            loss.backward()
            optimizer.step()
        knowledge = process_losses(losses)
        sampler.update_stats(knowledge)
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
            output = F.log_softmax(output, dim=1)
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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=30),
        transforms.RandomPerspective(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    # transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    #     ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    alpha = 0.1
    sampler = mygen.MyBatchSampler(len(dataset1), dev = device, batch_size=args.batch_size, alpha= alpha)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_sampler=sampler)#, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    #model = Net().to(device)
    from torchvision import models
    model_ft = models.resnet18(weights=None)
    # change input layer
    # the default number of input channel in the resnet is 3, but our images are 1 channel. So we have to change 3 to 1.
    # nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) <- default
    model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # change fc layer
    # the number of classes in our dataset is 10. default is 1000.
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 10)
    model = model_ft.to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, sampler)
        test(model, device, test_loader)
        scheduler.step()
        print("LR ", scheduler.get_last_lr())

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
