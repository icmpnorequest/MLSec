# coding=utf8
"""
@author: Yantong Lai
@date: 09/11/2019
@code description: It is a Python3 file to implement neural network models in Paper
`Membership Inference Attack against Machine Learning Models` by Shokri et al. in 17 S&P
"""

import numpy as np
import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn


# Dataset path
dataset_path = "../data"


# Hyper-parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001
num_workers = 2


def new_size_conv(size, kernel, stride=1, padding=0):
    return np.floor((size + 2 * padding - (kernel - 1) - 1) / stride + 1)


def new_size_max_pool(size, kernel, stride=None, padding=0):
    if stride == None:
        stride = kernel
    print("MaxPool kernel = ", kernel)
    print("MaxPool stride = ", stride)
    print("MaxPool padding = ", padding)
    return np.floor((size + 2 * padding - (kernel - 1) - 1) / stride + 1)


def calc_mlleaks_cnn_size(size):
    x = new_size_conv(size, 5, 1, 2)
    print("After Conv1, x = ", x)
    print("\n")

    x = new_size_max_pool(x, 2, 2)
    print("After MaxPool1, x = ", x)
    print("\n")

    x = new_size_conv(x, 5, 1, 2)
    print("After Conv2, x = ", x)
    print("\n")

    out = new_size_max_pool(x, 2, 2)
    print("After MaxPool2, x = ", out)

    return out


def cifar10_loader(data_path, batch_size, num_workers):
    """
    It is a function to load CIFAR-10 Data Loader.
    :param data_path: path of the dataset
    :return: train_loader, test_loader
    """
    # Transform
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load Dataset
    train_dataset = torchvision.datasets.CIFAR10(root=data_path,
                                                 download=True,
                                                 train=True,
                                                 transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_path,
                                                download=True,
                                                train=False,
                                                transform=transform)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, test_loader


train_loader, test_loader = cifar10_loader(dataset_path, batch_size, num_workers)
