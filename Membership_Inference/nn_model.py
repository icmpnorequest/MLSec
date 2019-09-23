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
            nn.Softmax(dim=0))

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(x.size(0), -1)
        output = self.fc(output)

        return output


def train(net, train_loader, num_epochs, criterion, optimizer, device):
    """
    It is a function to train the NN model
    :param net: neural network
    :param train_loader: train data loader
    :param num_epochs: number of epochs
    :param criterion: loss function
    :param optimizer: optimizer
    :param device: device (cuda or cpu)
    """
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            output = net(images)
            loss = criterion(output, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    print("\n")


def test(net, test_loader, test_dataset, criterion, device):
    """
    It is a function to test the NN model
    :param net: NN model
    :param test_loader: test data loader
    :param test_dataset: test dataset
    :param criterion: loss function
    :param device: device (cpu or cuda)
    """
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        output = net(images)
        avg_loss += criterion(output, labels)
        probability, pred = torch.max(output.data, dim=1)
        # pred = torch.argmax(output.data, dim=1)
        # print("Probability = ", probability)
        # print("Probability.size() = ", probability.size())
        total_correct += (pred == labels).sum().item()

    avg_loss = avg_loss / len(test_dataset)
    print("Test Avg. Loss: {}, Accuracy: {}%"
          .format(avg_loss, 100 * total_correct / len(test_dataset)))


def nn_predict_proba(net, test_loader, device, fix_class):
    """
    It is a function to calculate probability.
    :param net: neural network model
    :param test_loader: test data loader
    :param device: device (cpu or cuda)
    :return: probability
    """
    net.eval()
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        output = net(images)
        sm = torch.nn.Softmax(dim=1)
        probabilities = sm(output)
        y_c = probabilities.data[0, fix_class - 1].item()

        return y_c


def nn_predict(net, test_loader, device):
    """
    It is a function to predict label.
    :param net: neural network model
    :param test_loader: test data loader
    :param device: device (cpu or cuda)
    :return: prediction label
    """
    net.eval()
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        output = net(images)

    return output.data