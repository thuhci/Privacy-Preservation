import os
import math
import torch
import shutil
import argparse
import torchvision
import collections
import pandas as pd
from tqdm import tqdm
from torch import nn
from d2l import torch as d2l

parser = argparse.ArgumentParser()
parser.add_argument('--directory',
                    default='../data/',
                    type=str)
parser.add_argument('--size',
                    default=224,
                    type=int)
parser.add_argument('--batch',
                    default=4,
                    type=int)
parser.add_argument('--epochs',
                    default=50,
                    type=int)
parser.add_argument('--device_id',
                    default=1,
                    type=int)
parser.add_argument('--model_name',
                    default='vit',
                    type=str)
parser.add_argument('--learning_rate',
                    default=5e-5,
                    type=float)


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def train_batch(net, X, y, loss, trainer, device):
    if isinstance(X, list):
        X = [x.to(device) for x in X]
    else:
        X = X.to(device)
    y = y.to(device)
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum


def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train(net, train_iter, test_iter, valid_iter, loss, trainer, num_epochs, devices=d2l.try_all_gpus()):
    timer, num_batches = d2l.Timer(), len(train_iter)
    valid_acc_previous = 0
    valid_acc_fall = 0
    valid_threshold = 5
    legend = ['train loss', 'train acc', 'test acc']
    if valid_iter is not None:
        legend.append('valid acc')
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        for i, (features, labels) in tqdm(enumerate(train_iter), desc="iter"):
            timer.start()
            l, acc = train_batch(net, features, labels,
                                 loss, trainer, devices[0])
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
        train_acc = metric[1] / metric[3]
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        valid_acc = evaluate_accuracy_gpu(net, valid_iter)
        if valid_acc <= valid_acc_previous:
            valid_acc_fall += 1
            if valid_acc_fall > valid_threshold:
                break
        elif max(valid_acc, valid_acc_previous) == valid_acc:
            valid_acc_fall = 0
        print(f'epoch:{epoch} train:{train_acc:.3f} test:{test_acc:.3f} valid:{valid_acc:.3f} '
              f'valid_pre:{valid_acc_previous:.3f} valid_fall:{valid_acc_fall}')
        valid_acc_previous = max(valid_acc, valid_acc_previous)
    print(
        f'loss {metric[0] / metric[2]:.3f}, train acc 'f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(
        f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on 'f'{str(devices[0])}')


def main():
    args = parser.parse_args()
    print(args)
    data_directory = args.directory
    test_size = args.size
    batch_size = args.batch
    num_epochs = args.epochs
    model_name = args.model_name
    device_id = args.device_id
    learning_rate = args.learning_rate
    resize = 512
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize((test_size, test_size)),
        torchvision.transforms.Resize((resize, resize)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010])])
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize((test_size, test_size)),
        torchvision.transforms.Resize((resize, resize)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010])])
    finetune_net = nn.Sequential()

    if model_name == 'resnet':
        finetune_net = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 6)
        nn.init.xavier_uniform_(finetune_net.fc.weight)
    elif model_name == 'vit':
        finetune_net.features = torchvision.models.vit_l_16(
            weights=torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        finetune_net.output = nn.Sequential(
            nn.Linear(1000, 256), nn.ReLU(), nn.Linear(256, 6))
    elif model_name == 'efficient':
        finetune_net.features = torchvision.models.efficientnet_v2_l(
            weights=torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        finetune_net.output = nn.Sequential(
            nn.Linear(1000, 256), nn.ReLU(), nn.Linear(256, 6))

    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_directory, 'train_resized', str(224)), transform=transform_train), batch_size=batch_size,
        shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_directory, 'test_resized', str(224)), transform=transform_test), batch_size=batch_size)
    valid_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_directory, 'valid_resized'), transform=transform_test), batch_size=batch_size)
    devices = [try_gpu(device_id)]
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.SGD(
        finetune_net.parameters(), lr=learning_rate, weight_decay=0.001)
    train(finetune_net, train_iter, test_iter,
          valid_iter, loss, trainer, num_epochs, devices)


if __name__ == '__main__':
    main()
