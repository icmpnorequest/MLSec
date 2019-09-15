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
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn


def cal_conv_size(input_size, kernel_size, padding, stride=1):
    """
    It is a function to calculate tensor after convolution layer.
    :param input_size: input tensor size
    :param kernel_size: kernel size
    :param padding: padding
    :param stride: stride
    :return: tensor size after convolution layer
    """
    output = np.floor((input_size - kernel_size + 2 * padding) / stride + 1)
    return output


class CNN_CIFAR10(nn.Module):
    """
    Local Target Model: A standard convolutional neural network.
    - 2 Conv layer
    - 2 Max pooling layer
    - 1 FC layer
    - 1 Softmax layer
    - Tanh as the activation function
    """
    def __init__(self):
        super(CNN_CIFAR10, self).__init__()

        self.convnet = nn.Sequential(
            # Conv1
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.Tanh(),

            # MaxPool1
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),

            # MaxPool2
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            # FC1
            nn.Linear(in_features=16 * 5 * 5, out_features=128),
            nn.Tanh(),

            # FC2
            nn.Linear(in_features=128, out_features=10),
            nn.Softmax())

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size[0], -1)
        output = self.fc(output)

        return output

