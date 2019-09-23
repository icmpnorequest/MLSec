# coding=utf8
"""
@author: Yantong Lai
@date: 09/11/2019
@code description: It is a Python3 file to implement neural network models in Paper
`Membership Inference Attack against Machine Learning Models` by Shokri et al. in 17 S&P
"""

import numpy as np
import pandas as pd
import os
import argparse
from skorch import NeuralNetClassifier

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
ndarray_path="cifar10/norm_all_batch_data.npy"

# Hyper-parameters
num_epochs = 10
batch_size = 100
learning_rate = 0.001
num_workers = 2

# Device configuaration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--trainset_size", type=int, default=100, help="size of trainset (2500, 5000, 10000, 15000)")
parser.add_argument("-c", "--in_channels", type=int, default=3, help="number of in_channels")
parser.add_argument("-im", "--image_size", type=int, default=32, help="size of image")

args = parser.parse_args()


def train_target_model(target_train_dataset, target_train_loader, trainset_size, net, num_epochs, criterion,
                       optimizer, device, dataset_path):
    """
    It is a function to train the target model
    :param net: neural network
    :param train_loader: train data loader
    :param num_epochs: number of epochs
    :param criterion: loss function
    :param optimizer: optimizer
    :param device: device (cuda or cpu)
    """

    '''
    # 1.1 Use API: torch.utils.data import Subset
    # 1) Create an instance using official provided API.
    cifar10_loader_api = Load_CIFAR10_API(dataset_path=dataset_path)
    # 2) Load CIFAR-10 Dataset
    api_train_dataset, api_test_dataset = cifar10_loader_api.cifar10_dataset()
    # 3) Load CIFAR-10 DataLoader
    subset_indices = list(range(trainset_size))
    api_train_subset = Subset(api_train_dataset, indices=subset_indices)
    api_train_loader = DataLoader(api_train_subset, batch_size=batch_size, shuffle=True)

    # 2. train
    train(net=net,
          train_loader=target_train_loader,     # if according to 5.1, change train_loader = api_train_loader
          num_epochs=num_epochs,
          criterion=criterion,
          optimizer=optimizer,
          device=device)

    # 3. test
    test(net=net,
         test_loader=target_train_loader,
         test_dataset=target_train_dataset,
         criterion=criterion,
         device=device)

    # 4. prediction probability
    probability = nn_predict_proba(net=net,
                                   test_loader=target_train_loader,
                                   device=device)
    '''


def data_synthesize(net, trainset_size, fix_class, initial_record, k_max,
                    in_channels, img_size, batch_size, num_workers, device):
    """
    It is a function to synthesize data
    """
    # Initialize X_tensor with an initial_record, with size of (1, in_channels, img_size, img_size)
    X_tensor = initial_record
    print("X_tensor.size() = ", X_tensor.size())
    # Generate y_tensor with the size equivalent to X_tensor's
    y_tensor = gen_class_tensor(trainset_size, fix_class)
    print("y_tensor.size() = ", y_tensor.size())

    y_c_current = 0         # target models probability of fixed class
    j = 0                   # consecutive rejections counter
    k = k_max               # search radius
    max_iter = 100          # max iter number
    conf_min = 0.1          # min probability cutoff to consider a record member of the class
    rej_max = 5             # max number of consecutive rejections
    k_min = 1               # min radius of feature perturbation

    for _ in range(max_iter):

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

        y_c = nn_predict_proba(net, dataloader, device, fix_class)

        # Phase 1: Search
        if y_c >= y_c_current:
            # Phase 2: Sample
            if y_c > conf_min and fix_class == torch.argmax(nn_predict(net, dataloader, device), dim=1):
                return X_tensor

            X_new_tensor = X_tensor
            y_c_current = y_c  # renew variables
            j = 0
        else:
            j += 1
            if j > rej_max:  # many consecutive rejects
                k = max(k_min, int(np.ceil(k / 2)))
                j = 0
        X_tensor = rand_tensor(X_new_tensor, k, in_channels, img_size, trainset_size)

    return X_tensor, y_c


def shadow_dataset(trainset_size, net, fix_class, initial_record, k_max, in_channels,
                   img_size, batch_size, num_workers, device, num_label):
    """
    It is a function to form training and testing dataset for shadow models
    :param trainset_size: size of training dataset
    :param X_tensor: X_tensor
    :return: training_set / testing_set
    """
    trainset_list = []
    testset_list = []

    total_trainset_list = []
    total_testset_list = []


    for i in range(trainset_size //  num_label):
        # Synthesize data
        X_tensor, y_c = data_synthesize(net=net,
                                        trainset_size=1,
                                        fix_class=fix_class,
                                        initial_record=initial_record,
                                        k_max=k_max,
                                        in_channels=in_channels,
                                        img_size=img_size,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        device=device)
        trainset_list.append(X_tensor)

    return torch.stack(trainset_list)


def main():

    ####################### 1. Data Preprocessing ######################
    # 1. Load norm_all_batch_data array
    norm_all_data_array = (np.load(ndarray_path, allow_pickle=True))  # (5, 10000, 3)
    concatenate_norm_array = np.concatenate((norm_all_data_array))  # (50000, 3)

    # 2. Extract train set array
    trainset_array = concatenate_norm_array[:args.trainset_size, :]  # (trainset_size, 3)

    # 3. Extract data and label from trainset_array
    X = trainset_array[:, 0]  # Data, shape = (trainset_size, ). X[0].shape = (32, 32, 3)
    y = trainset_array[:, 1]  # Label, shape = (trainset_size, )

    # 4. Create an Ndarray2DataLoader instance
    nd2loader = Ndarray2DataLoader(X_array=X,
                                   y_array=y,
                                   trainset_size=args.trainset_size,
                                   in_channels=args.in_channels,
                                   img_size=args.image_size,
                                   batch_size=batch_size,
                                   num_workers=num_workers)

    # 5. Generate target train dataset
    target_train_dataset = nd2loader.form_dataset(X_array=X,
                                                  y_array=y,
                                                  trainset_size=args.trainset_size,
                                                  in_channels=args.in_channels,
                                                  img_size=args.image_size)

    # 6. Generate target train data loader
    target_train_loader = nd2loader.form_dataloader(dataset=target_train_dataset)


    ##################### 2. Train Target Model #######################
    # 1. Define target neural network
    target_net = CNN_CIFAR10()

    # 2. Define loss function
    criterion = nn.CrossEntropyLoss()

    # 3. Define optimizer
    optimizer = optim.Adam(target_net.parameters(), lr=learning_rate)

    # 4. Train target model
    train_target_model(target_train_dataset=target_train_dataset,
                       target_train_loader=target_train_loader,
                       trainset_size=args.trainset_size,    # (2500, 5000, 10000, 15000)
                       net=target_net,
                       num_epochs=num_epochs,
                       criterion=criterion,
                       optimizer=optimizer,
                       device=device,
                       dataset_path=dataset_path)


    ################## 3. CIFAR-10 Data Synthesis ####################
    # 1. Initialize a data record
    initial_record = torch.rand([1, args.in_channels, args.image_size, args.image_size])

    # 2. Data Synthesis
    X_tensor, y_c = data_synthesize(net=target_net,
                                    trainset_size=1,
                                    fix_class=8,
                                    initial_record=initial_record,
                                    k_max=8,
                                    in_channels=3,
                                    img_size=32,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    device=device)

    print("X_tensor = ", X_tensor)
    print("X_tensor.size() = ", X_tensor.size())
    print("y_c = ", y_c)

    # 3. Get shadow model training set and testing set

    # shadow_training_set

    # trainset_size, net, fix_class, initial_record, k_max, in_channels,
    #                    img_size, batch_size, num_workers, device
    shadow_training_set = shadow_dataset(trainset_size=args.trainset_size,
                                         net=target_net,
                                         fix_class=7,
                                         initial_record=initial_record,
                                         k_max=8,
                                         in_channels=args.in_channels,
                                         img_size=args.image_size,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         device=device,
                                         num_label=len(np.unique(y)))

    print("len(shadow_training_set) = ", len(shadow_training_set))
    print("shadow_training_set.size() = ", shadow_training_set.size())


if __name__ == '__main__':



    main()
