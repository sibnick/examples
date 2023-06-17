from __future__ import print_function
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.modules
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn.init

class InnerNet(nn.Module):
    def __init__(self, len, emb_size, hidden:int = 128):
        super(InnerNet, self).__init__()
        self.emb_size = emb_size
        self.embedding = nn.Embedding(len, emb_size, max_norm=1, norm_type=2)
        #self.embedding.weight.data = self.embedding.weight.data / 100
        self.fc1 = nn.Linear(emb_size * 2, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, 1, bias=False)
        self.fc2b = nn.Linear(hidden, int(hidden/2), bias=False)
        self.bn = nn.BatchNorm1d(int(hidden/2))
        self.fc3 = nn.Linear(int(hidden/2), 1, bias=False)

    def forward(self, x:tuple):
        prev_idx, cur_idx, prev_loss = x
        e = self.embedding(torch.concat((prev_idx, cur_idx)))
        x0 = e[0 : prev_idx.shape[0]]
        x1 = e[prev_idx.shape[0] : ]

        x0 = x0 * prev_loss[:, None].repeat((1, self.emb_size))
        mean_x0 = torch.mean(x0, axis=0)
        mean_x0 = mean_x0[None, :].repeat((x1.shape[0], 1))
        x = torch.concat((mean_x0, x1), dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        # x = self.fc2b(x)
        # x = F.relu(x)
        # x = self.fc3(x)
        # x = F.relu(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.prev_loss = None
        self.prev_mean_loss = None
        self.prev_idx = None

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



def train(args, model, imodel, device, train_loader, optimizer, optimizer2, epoch):
    model.train()
    imodel.train()
    imin = 1e+10
    imax = 0
    iavg = 0
    count = 1
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        tmp_loss = F.nll_loss(output, target, reduction='none')
        # topk = torch.argmax(tmp_loss)
        # tmp_loss[topk] = 0
        # topk = torch.topk(tmp_loss, k=int(tmp_loss.shape[0]/2), largest=False)
        # tmp_loss[topk[1]] = 0
        loss = torch.mean(tmp_loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loader.batch_sampler.update_stats(tmp_loss, output.detach().cpu())

        prev_loss = model.prev_loss
        prev_mean_loss = model.prev_mean_loss
        prev_idx = model.prev_idx
        model.prev_loss = tmp_loss.detach()
        model.prev_mean_loss = loss.detach()
        model.prev_idx = torch.tensor(train_loader.batch_sampler.batch).detach().to(loss.device)

        if prev_loss is not None:
            for n in range(1):
                iy = imodel((prev_idx, model.prev_idx, prev_loss))
                yy = model.prev_loss
                loss2 = F.mse_loss(iy.view(-1), yy)
                loss2.backward()
                optimizer2.step()
                tmp = loss2.detach().item()
                optimizer2.zero_grad()
            imin = min(imin, tmp)
            imax = max(imax, tmp)
            iavg += tmp
            count += 1
            if batch_idx % args.log_interval == 0:
                print("intersect: ", np.intersect1d(yy.topk(k=8, largest=True)[1].cpu(), iy[:, 0].topk(k=8, largest=True)[1].cpu()))



        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss))
            # if args.dry_run:
            #     break
            print("imin: ", imin, " imax: ", imax, " avg: ", iavg/count)
    print("imin: ", imin, " imax: ", imax, " avg: ", iavg/count)


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
    import mygen
    sampler = mygen.DynamicWeightBatchSampler(len(dataset1), batch_size=args.batch_size, seed=1)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_sampler = sampler)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    emb_size = 8
    imodel = InnerNet(len(dataset1), emb_size, hidden=1024).to(device)
    optimizer2 = optim.Adadelta(imodel.parameters(), lr=0.1)
    #scheduler2 = StepLR(optimizer2, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, imodel, device, train_loader, optimizer, optimizer2, epoch)
        test(model, device, test_loader)
        scheduler.step()
        #scheduler2.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
