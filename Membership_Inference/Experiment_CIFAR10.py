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
from torch.utils.data import Subset

from Membership_Inference.nn_model import *
from Membership_Inference.utils import *


# Dataset path
dataset_path = "../data"

# Hyper-parameters
num_epochs = 10
batch_size = 100
learning_rate = 0.001
num_workers = 2

# Device configuaration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_target_model(ndarray_path, trainset_size, net, num_epochs, criterion,
                       optimizer, device, dataset_path, in_channels, img_size):
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

    # 3. Create an Ndarray2DataLoader instance
    nd2loader = Ndarray2DataLoader(X_array=X,
                                   y_array=y,
                                   trainset_size=trainset_size,
                                   in_channels=in_channels,
                                   img_size=img_size,
                                   batch_size=batch_size,
                                   num_workers=num_workers)

    # 4. Generate target train dataset
    target_train_dataset = nd2loader.form_dataset(X_array=X,
                                                  y_array=y,
                                                  trainset_size=trainset_size,
                                                  in_channels=in_channels,
                                                  img_size=img_size)

    # 5. Generate target train data loader
    target_train_loader = nd2loader.form_dataloader(dataset=target_train_dataset)

    # 5.1 Use API: torch.utils.data import Subset
    # 1) Create an instance using official provided API.
    cifar10_loader_api = Load_CIFAR10_API(dataset_path=dataset_path)
    # 2) Load CIFAR-10 Dataset
    api_train_dataset, api_test_dataset = cifar10_loader_api.cifar10_dataset()
    # 3) Load CIFAR-10 DataLoader
    subset_indices = list(range(trainset_size))
    api_train_subset = Subset(api_train_dataset, indices=subset_indices)
    api_train_loader = DataLoader(api_train_subset, batch_size=batch_size, shuffle=True)

    # 6. train
    train(net=net,
          train_loader=target_train_loader,     # if according to 5.1, change train_loader = api_train_loader
          num_epochs=num_epochs,
          criterion=criterion,
          optimizer=optimizer,
          device=device)

    # 7. test
    test(net=net,
         test_loader=target_train_loader,
         test_dataset=target_train_dataset,
         criterion=criterion,
         device=device)

    probability = nn_predict_proba(net=net,
                                   test_loader=target_train_loader,
                                   device=device)
    print("probability = ", probability)
    print("probability.size() = ", probability.size())
    print("probability.dtype = ", probability.dtype)


def cifar10_data_synthesize():
    """
    :return:
    """
    pass





def main():

    # 1. Define target neural network
    target_net = CNN_CIFAR10()

    # 2. Define loss function
    criterion = nn.CrossEntropyLoss()

    # 3. Define optimizer
    optimizer = optim.Adam(target_net.parameters(), lr=learning_rate)

    # 4. Train target model
    train_target_model(ndarray_path="cifar10/norm_all_batch_data.npy",
                       trainset_size=2500,      # (2500, 5000, 10000, 15000)
                       net=target_net,
                       num_epochs=num_epochs,
                       criterion=criterion,
                       optimizer=optimizer,
                       device=device,
                       dataset_path=dataset_path,
                       in_channels=3,
                       img_size=32)


if __name__ == '__main__':

    main()