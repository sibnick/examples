from __future__ import print_function
import argparse
import copy
from collections import OrderedDict
from functools import partial

from torch.utils.data import ConcatDataset, TensorDataset

import mygen
import numpy as np
import torch
import torch.nn as nn
import torch.nn.modules
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn.init
from sklearn.model_selection import KFold



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


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    count = 0
    stat = np.zeros((10), dtype=int)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        count += target.shape[0]
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
        if isinstance(train_loader.batch_sampler, mygen.DynamicWeightBatchSampler):
            for i in target:
                stat[i] += 1
            train_loader.batch_sampler.update_stats(tmp_loss, output.detach().cpu())

        if batch_idx % args.log_interval == 0:
            # if isinstance(train_loader.batch_sampler, mygen.DynamicWeightBatchSampler) and  epoch > 1:
            #     print("batch stat: ", 100.0 * stat / np.sum(stat))

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss))
            # if args.dry_run:
            #     break
    print("Finish epoch. Train items count: ", count)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            count += target.shape[0]

    test_loss /= count

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
                .format(test_loss, correct, count,  100. * correct / count))
    return test_loss, 100. * correct / count

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

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
    parser.add_argument('--new-sampler',  action='store_true', default=False,
                        help='Use new sampler')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    print(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device: ", device)
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True
                       } # , 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    dataset = ConcatDataset([dataset1, dataset2])
    # Define the K-fold Cross Validator
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    # For fold results
    results = {}
    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        if args.new_sampler:
            sampler = mygen.DynamicWeightBatchSampler(len(dataset), batch_size=args.batch_size, seed=1, exclude_ids=test_ids)
            train_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
            print("Use new sampler")
        else:
            train_loader = torch.utils.data.DataLoader(dataset, sampler=train_subsampler, **train_kwargs)
            print("Use old sampler")

        train_stat = np.zeros((10), dtype=int)
        target_ids = []
        for i in range(10):
            target_ids.append([])
        for i in train_ids:
            target = dataset[i][1]
            train_stat[target] += 1
            target_ids[target].append(i)
        for i in range(10):
            target_ids[i] = np.array(target_ids[i])
        if args.new_sampler:
            sampler.target_ids = target_ids

        test_stat = np.zeros((10), dtype=int)
        for i in test_ids:
            test_stat[dataset[i][1]] += 1
        print("train stat: ", 100.0*train_stat/len(train_ids))
        print("test stat: ", 100.0*test_stat/len(test_ids))
        test_loader = torch.utils.data.DataLoader(dataset, sampler=test_subsampler)
        model = Net().to(device)
        model.apply(reset_weights)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            # train(args, model, imodel, device, train_loader, optimizer, optimizer2, epoch)
            results[fold] = test(model, device, test_loader)
            scheduler.step()
            #scheduler2.step()

        if args.save_model:
            if isinstance(train_loader.batch_sampler, mygen.DynamicWeightBatchSampler):
                torch.save(model.state_dict(), "mnist_cnn_weighted_" + str(fold) + ".pt")
            else: torch.save(model.state_dict(), "mnist_cnn" + str(fold) + ".pt")
        # break

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum_loss = 0.0
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum_loss += value[0]
        sum += value[1]
    print(f'Average:  {sum_loss/len(results.items())}, {sum/len(results.items())} %')

if __name__ == '__main__':
    main()
