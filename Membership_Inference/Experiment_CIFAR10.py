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
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim

from Membership_Inference.nn_model import *


# Dataset path
dataset_path = "../data"


# Hyper-parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001
num_workers = 2


# Device configuaration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


# train_loader, test_loader = cifar10_loader(dataset_path, batch_size, num_workers)

def rm_array_comma(len, ndarray):
    """
    It is a function to remove comma in an array
    :param ndarray: np.ndarray
    :return: array without comma
    """
    for _ in range(len):
        for i in ndarray:
            if "," in i:
                i = i.replace(",", "")
    return ndarray





def ndarray_to_tensor(ndarray):
    """
    It is a function to transform <np.ndarray> to <tensor>
    :param ndarray: np.ndarray
    :return: tensor object
    """
    out = torch.from_numpy(ndarray)
    return out


def form_dataset(X_tensor, y_tensor):
    """
    It is a function to form dataset.
    :param X_tensor: tensor x
    :param y_tensor: tensor y
    :return: dataset
    """
    dataset = TensorDataset(X_tensor, y_tensor)
    return dataset


def form_dataloader(tensor_dataset, batch_size, num_workers):
    """
    It is a function to form data loader from tensor dataset.
    :param tensor_dataset: tensor dataset
    :return: data loader
    """
    data_loader = DataLoader(tensor_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return data_loader


def train_target_model(ndarray_path, trainset_size, net, num_epochs, criterion, optimizer, device):
    """
    It is a function to train the target model
    :param net: neural network
    :param train_loader: train data loader
    :param num_epochs: number of epochs
    :param criterion: loss function
    :param optimizer: optimizer
    :param device: device (cuda or cpu)
    """
    # 1. Load norm_all_batch_data array
    norm_all_data_array = (np.load(ndarray_path, allow_pickle=True))    # (5, 10000, 3)
    concatenate_norm_array = np.concatenate((norm_all_data_array))      # (50000, 3)

    # 2. Extract train set array
    trainset_array = concatenate_norm_array[:trainset_size, :]      # (trainset_size, 3)
    # print(trainset_array)

    X = trainset_array[:, 0]            # Data, shape = (trainset_size, )
    y = trainset_array[:, 1]            # Label, shape = (trainset_size, )


    test = X


    # test1 = rm_array_comma(len(test), test)



    test1 = np.frompyfunc(lambda x: x.replace(',', ''), 1, 1)(test).astype(float)
    print(test1)


    # test1 = torch.Tensor(list(test.values))

    # print(test1)


    # print(X)
    # test = np.array([v.replace(',', '') for v in X], dtype=np.float32)
    # print(test.dtype)

    # test = np.core.defchararray.replace(X.astype(np.unicode_), ',', '').astype(np.float)
    # print(test)

    # 3. ndarray -> tensor
    # X_tensor = ndarray_to_tensor(ndarray=X)
    # y_tensor = ndarray_to_tensor(ndarray=y)

    # X_tensor = torch.from_numpy(test).to(float)


    # print(X_tensor)
    print("\n")
    # print(y_tensor)
    """
    # 4. tensor -> dataset
    target_train_dataset = form_dataset(X_tensor=X_tensor, y_tensor=y_tensor)

    # 5. dataset -> data loader
    target_train_loader = form_dataloader(tensor_dataset=target_train_dataset,
                                          batch_size=batch_size,
                                          num_workers=num_workers)

    # 6. train
    train(net=net,
          train_loader=target_train_loader,
          num_epochs=num_epochs,
          criterion=criterion,
          optimizer=optimizer,
          device=device)
    """



def main():

    # 1. Define target neural network
    target_net = CNN_CIFAR10()

    # 2. Define loss function
    criterion = nn.CrossEntropyLoss()

    # 3. Define optimizer
    optimizer = optim.Adam(target_net.parameters(), lr=learning_rate)

    # 4. Train target model
    train_target_model(ndarray_path="cifar10/norm_all_batch_data.npy",
                       trainset_size=2500,
                       net=target_net,
                       num_epochs=num_epochs,
                       criterion=criterion,
                       optimizer=optimizer,
                       device=device)


if __name__ == '__main__':

    main()
