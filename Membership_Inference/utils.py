# coding=utf8
"""
@author: Yantong Lai
@date: 09/18/2019
@code description: It is a Python3 file to implement utils in Paper
`Membership Inference Attack against Machine Learning Models` by Shokri et al. in 17 S&P
"""

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
from torch.utils.data import Subset
import torchvision.transforms as transforms

import numpy as np


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


class Ndarray2DataLoader:
    """
    It is a custom class to transform <np.ndarray> to <torch.DataLoader>
    """

    def __init__(self, X_array, y_array, trainset_size, in_channels, img_size, batch_size, num_workers):
        self.X_array = X_array
        self.y_array = y_array
        self.trainset_size = trainset_size
        self.in_channels = in_channels
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def ndarray_to_tensor(self, ndarray):
        """
        It is a function to transform <np.ndarray> to <tensor>
        :param ndarray: np.ndarray
        :return: tensor object
        """
        out = torch.Tensor(list(ndarray))
        return out

    def form_dataset(self, X_array, y_array, trainset_size, in_channels, img_size):
        """
        It is a function to form dataset.
        :param X_tensor: tensor x
        :param y_tensor: tensor y
        :return: dataset
        """
        X_tensor = self.ndarray_to_tensor(X_array).reshape([trainset_size, in_channels, img_size, img_size])
        y_tensor = self.ndarray_to_tensor(y_array).long()
        dataset = TensorDataset(X_tensor, y_tensor)
        return dataset

    def form_dataloader(self, dataset):
        """
        It is a function to form data loader from tensor dataset.
        :param dataset: tensor dataset
        :return: data loader
        """
        data_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return data_loader


class Load_CIFAR10_API():
    """
    It is a class to load cifar-10 dataset using official provided API.
    """

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def cifar10_transform(self):
        """
        It is a function to transform picture of CIFAR-10.
        :return: transform
        """
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        return transform

    def cifar10_dataset(self):
        """
        It is a function to load CIFAR-10 dataset
        :param dataset_path: path of dataset
        :return: train_dataset, test_dataset
        """
        transform = self.cifar10_transform()
        train_dataset = torchvision.datasets.CIFAR10(root=self.dataset_path,
                                                     download=True,
                                                     train=True,
                                                     transform=transform)

        test_dataset = torchvision.datasets.CIFAR10(root=self.dataset_path,
                                                    download=True,
                                                    train=False,
                                                    transform=transform)
        return train_dataset, test_dataset

    def cifar10_loader(self, batch_size, num_workers, sampler):
        """
        It is a function to load CIFAR-10 Data Loader.
        :param data_path: path of the dataset
        :return: train_loader, test_loader
        """
        # Load Dataset
        train_dataset, test_dataset = self.cifar10_dataset()

        # Data Loader
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                                  num_workers=num_workers, sampler=sampler)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 num_workers=num_workers)

        return train_loader, test_loader


def bug_recorder():
    """
    It is a function to record bugs.
    """
    bug = "RuntimeError: Given groups=1, weight of size 6 3 5 5, " \
          "expected input[100, 32, 32, 3] to have 3 channels, " \
          "but got 32 channels instead"

    return None


def result_recorder():
    """
    It is a function to record results.
    """
    result = """ Custom Dataset, 
    1) trainset_size = 2500
    Epoch [1/3], Step [5/25], Loss: 2.3024
    Epoch [1/3], Step [10/25], Loss: 2.3009
    Epoch [1/3], Step [15/25], Loss: 2.2994
    Epoch [1/3], Step [20/25], Loss: 2.2982
    Epoch [1/3], Step [25/25], Loss: 2.2977
    Epoch [2/3], Step [5/25], Loss: 2.2943
    Epoch [2/3], Step [10/25], Loss: 2.2945
    Epoch [2/3], Step [15/25], Loss: 2.2930
    Epoch [2/3], Step [20/25], Loss: 2.2910
    Epoch [2/3], Step [25/25], Loss: 2.2916
    Epoch [3/3], Step [5/25], Loss: 2.2867
    Epoch [3/3], Step [10/25], Loss: 2.2842
    Epoch [3/3], Step [15/25], Loss: 2.2886
    Epoch [3/3], Step [20/25], Loss: 2.2848
    Epoch [3/3], Step [25/25], Loss: 2.2887

    2) trainset_size = 5000
    Epoch [1/3], Step [5/50], Loss: 2.3022
    Epoch [1/3], Step [10/50], Loss: 2.3018
    Epoch [1/3], Step [15/50], Loss: 2.3003
    Epoch [1/3], Step [20/50], Loss: 2.2998
    Epoch [1/3], Step [25/50], Loss: 2.2981
    Epoch [1/3], Step [30/50], Loss: 2.2963
    Epoch [1/3], Step [35/50], Loss: 2.2975
    Epoch [1/3], Step [40/50], Loss: 2.2951
    Epoch [1/3], Step [45/50], Loss: 2.2928
    Epoch [1/3], Step [50/50], Loss: 2.2911
    Epoch [2/3], Step [5/50], Loss: 2.2943
    Epoch [2/3], Step [10/50], Loss: 2.2856
    Epoch [2/3], Step [15/50], Loss: 2.2865
    Epoch [2/3], Step [20/50], Loss: 2.2909
    Epoch [2/3], Step [25/50], Loss: 2.2896
    Epoch [2/3], Step [30/50], Loss: 2.2866
    Epoch [2/3], Step [35/50], Loss: 2.2907
    Epoch [2/3], Step [40/50], Loss: 2.2947
    Epoch [2/3], Step [45/50], Loss: 2.2885
    Epoch [2/3], Step [50/50], Loss: 2.2902
    Epoch [3/3], Step [5/50], Loss: 2.2896
    Epoch [3/3], Step [10/50], Loss: 2.2883
    Epoch [3/3], Step [15/50], Loss: 2.2861
    Epoch [3/3], Step [20/50], Loss: 2.2693
    Epoch [3/3], Step [25/50], Loss: 2.2808
    Epoch [3/3], Step [30/50], Loss: 2.2880
    Epoch [3/3], Step [35/50], Loss: 2.2882
    Epoch [3/3], Step [40/50], Loss: 2.2841
    Epoch [3/3], Step [45/50], Loss: 2.2897
    Epoch [3/3], Step [50/50], Loss: 2.2961

    3) trainset_size = 10000
    Epoch [1/3], Step [10/100], Loss: 2.3009
    Epoch [1/3], Step [20/100], Loss: 2.3011
    Epoch [1/3], Step [30/100], Loss: 2.2963
    Epoch [1/3], Step [40/100], Loss: 2.2964
    Epoch [1/3], Step [50/100], Loss: 2.3005
    Epoch [1/3], Step [60/100], Loss: 2.2993
    Epoch [1/3], Step [70/100], Loss: 2.2883
    Epoch [1/3], Step [80/100], Loss: 2.2923
    Epoch [1/3], Step [90/100], Loss: 2.2905
    Epoch [1/3], Step [100/100], Loss: 2.2956
    Epoch [2/3], Step [10/100], Loss: 2.2887
    Epoch [2/3], Step [20/100], Loss: 2.2877
    Epoch [2/3], Step [30/100], Loss: 2.2850
    Epoch [2/3], Step [40/100], Loss: 2.2877
    Epoch [2/3], Step [50/100], Loss: 2.2826
    Epoch [2/3], Step [60/100], Loss: 2.2803
    Epoch [2/3], Step [70/100], Loss: 2.2881
    Epoch [2/3], Step [80/100], Loss: 2.2900
    Epoch [2/3], Step [90/100], Loss: 2.2791
    Epoch [2/3], Step [100/100], Loss: 2.2776
    Epoch [3/3], Step [10/100], Loss: 2.2846
    Epoch [3/3], Step [20/100], Loss: 2.2774
    Epoch [3/3], Step [30/100], Loss: 2.2863
    Epoch [3/3], Step [40/100], Loss: 2.2957
    Epoch [3/3], Step [50/100], Loss: 2.2879
    Epoch [3/3], Step [60/100], Loss: 2.2913
    Epoch [3/3], Step [70/100], Loss: 2.2676
    Epoch [3/3], Step [80/100], Loss: 2.2955
    Epoch [3/3], Step [90/100], Loss: 2.2682
    Epoch [3/3], Step [100/100], Loss: 2.2882
    """

    return None


