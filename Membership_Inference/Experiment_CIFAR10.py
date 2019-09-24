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
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import Subset
from torch.utils.data import Sampler
from torch.utils.data import RandomSampler

from Membership_Inference.nn_model import *
from Membership_Inference.utils import *
from Membership_Inference.shadow_models import SplitTrainTest


# Dataset path
dataset_path = "../data"
ndarray_path = "cifar10/norm_all_batch_data.npy"
shadowset_path = "shadowset/"

# Hyper-parameters
num_epochs = 10
batch_size = 100
learning_rate = 0.001
num_workers = 2

# Device configuaration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--trainset_size", type=int, default=250, help="size of trainset (2500, 5000, 10000, 15000)")
parser.add_argument("-c", "--in_channels", type=int, default=3, help="number of in_channels")
parser.add_argument("-im", "--image_size", type=int, default=32, help="size of image")
parser.add_argument("-ns", "--num_shadow_models", type=int, default=5, help="number of shadow models")

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


def data_synthesize(net, trainset_size, fix_class, initial_record, k_max,
                    in_channels, img_size, batch_size, num_workers, device):
    """
    It is a function to synthesize data
    :param net: net model
    :param trainset_size: size of trainset
    :param fix_class: fixed class
    :param initial_record: initial record
    :param k_max: max value of k
    :param in_channels: number of in_channels
    :param img_size: size of image
    :param batch_size: size of batch
    :param num_workers: number of workers
    :param device: device
    :return: synthetic data and prediction vector
    """
    # Initialize X_tensor with an initial_record, with size of (1, in_channels, img_size, img_size)
    X_tensor = initial_record
    # print("X_tensor.size() = ", X_tensor.size())
    # X_tensor.size() = torch.Size([1, 3, 32, 32])

    # Generate y_tensor with the size equivalent to X_tensor's
    y_tensor = gen_class_tensor(trainset_size, fix_class)
    # print("y_tensor.size() = ", y_tensor.size())
    # y_tensor.size() = torch.Size([1])

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
    :param trainset_size: trainset size
    :param net: net model
    :param fix_class: fixed class
    :param initial_record: initial record
    :param k_max: max value of k
    :param in_channels: number of in_channels
    :param img_size: size of image
    :param batch_size: size of batch
    :param num_workers: number of workers
    :param device: device to train
    :param num_label: number of labels
    :return: shadow dataset
    """
    trainset_list = []

    for i in range(trainset_size // num_label):
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


def train_shadow_model(num_shadow_model, all_synth_dataset, trainset_size, net,
                       num_epochs, criterion, optimizer, device):
    """
    It is a function to train shadow model
    :param num_shadow_model: number of shadow models
    :param all_synth_dataset: all synthetic datasets
    :param trainset_size: trainset size
    :param net: net model
    :param num_epochs: number of epochs
    :param criterion: loss function
    :param optimizer: optimizer
    :param device: device
    :return: train_labels_list, train_proba_list, test_labels_list, test_proba_list
    """
    train_proba_list = []
    test_proba_list = []

    train_labels_list = []
    test_labels_list = []

    for i in range(num_shadow_model):
        train_indices = np.random.randint(low=0, high=trainset_size - 1, size=trainset_size // num_shadow_model).tolist()
        test_indices = np.random.randint(low=0, high=trainset_size - 1, size=trainset_size // num_shadow_model).tolist()

        train_dataset = Subset(all_synth_dataset, train_indices)
        test_dataset = Subset(all_synth_dataset, test_indices)

        train_dataloader = DataLoader(train_dataset)        # 50
        test_dataloader = DataLoader(test_dataset)          # 50
        assert len(train_dataloader) == len(test_dataloader)

        # 1. Train
        print("########## Train ##########")
        total_step = len(train_dataloader)
        for epoch in range(num_epochs):
            net.train()
            for i, (images, labels) in enumerate(train_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                train_labels_list.append(labels)

                # Forward pass
                output = net(images)
                loss = criterion(output, labels)

                # Get train prediction probability
                sm = torch.nn.Softmax(dim=1)
                probabilities = sm(output)
                train_probability = probabilities.data[0, labels].item()
                train_proba_list.append(train_probability)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print training results
                if (i + 1) % 10 == 0:
                    print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # 2. Test
        print("########## Test ##########")
        net.eval()
        total_correct = 0
        avg_loss = 0.0
        for i, (images, labels) in enumerate(test_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            test_labels_list.append(labels)

            output = net(images)
            sm = torch.nn.Softmax(dim=1)
            probabilities = sm(output)
            test_probability = probabilities.data[0, labels].item()
            test_proba_list.append(test_probability)

            avg_loss += criterion(output, labels)
            pred = torch.argmax(output.data, dim=1)
            total_correct += (pred == labels).sum().item()

        avg_loss = avg_loss / len(test_dataset)
        print("Test Avg. Loss: {}, Accuracy: {}%\n"
              .format(avg_loss, 100 * total_correct / len(test_dataset)))

    return train_labels_list, train_proba_list, test_labels_list, test_proba_list


def form_attack_trainset(train_labels_list, train_proba_list, test_labels_list, test_proba_list):
    """
    It is a function to for attack training set.
    :param train_labels_list: shadow trainset labels
    :param train_proba_list: shadow trainset probabilities
    :param test_labels_list: shadow testset labels
    :param test_proba_list: shadow testset probabilities
    :return: attack_trainset
    """
    IN_list = np.ones(len(train_labels_list))
    assert len(train_labels_list) == len(train_proba_list) == len(IN_list)

    OUT_list = np.zeros(len(test_labels_list))
    assert len(test_labels_list) == len(test_proba_list) == len(OUT_list)

    # shadow trainset: (label, proba, 1)
    attack_in_array = np.array((train_labels_list, train_proba_list, IN_list)).T

    # shadow testset: (label, y_proba, 0)
    attack_out_array = np.array((test_labels_list, test_proba_list, OUT_list)).T

    return attack_in_array, attack_out_array


def split_attackset_by_label(attack_array, num_labels):
    """
    It is a function to split attack_array into k partitions by labels
    :param attack_array: attackset array
    :param num_labels: number of labels
    :return:
    """
    split_label_list = []
    for label in range(num_labels):
        # print("label = ", label)
        temp = attack_array[np.where(attack_array[:, 0] == label)]
        # print("len(temp) = ", len(temp))
        split_label_list.append(temp)

    return split_label_list


def train_attack_model(attack_model, split_label_list):
    """
    It is a function to train attack model with split label
    :param attack_model: attack model
    :param split_label_list: split attackset by labels
    """
    for item in np.array((split_label_list)):
        (X_train, y_train), (X_test, y_test) = SplitTrainTest(item)
        attack_model.fit(X_train, y_train)
        y_pred = attack_model.predict(X_test)

        print("Class label = {}".format(item[0][0]))  # print prediction results
        print("Membership Inference precision score: {}".format(precision_score(y_test, y_pred)))
        print("Mermbership Inference recall score: {}".format(recall_score(y_test, y_pred)))
        print("Membership Inference F1 score: {}".format(f1_score(y_test, y_pred)))
        print("########################\n")


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
    num_labels = len(np.unique(y))

    # 2. Data Synthesis Example
    X_tensor, y_c = data_synthesize(net=target_net,
                                    trainset_size=1,
                                    fix_class=5,
                                    initial_record=initial_record,
                                    k_max=8,
                                    in_channels=3,
                                    img_size=32,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    device=device)
    print("########## Example ##########")
    print("X_tensor = ", X_tensor)
    print("X_tensor.size() = ", X_tensor.size())
    print("y_c = ", y_c)

    # 3. Check shadowset directory to save shadow model training set and testing set
    if not os.path.exists(shadowset_path):
        os.mkdir(shadowset_path)

    # 4. Get shadow model training set and testing set
    for cls in range(num_labels - 1):
        # shadow_training_set class label from 0-8
        shadow_training_set = shadow_dataset(trainset_size=args.trainset_size,
                                             net=target_net,
                                             fix_class=cls,
                                             initial_record=initial_record,
                                             k_max=8,
                                             in_channels=args.in_channels,
                                             img_size=args.image_size,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             device=device,
                                             num_label=num_labels)

        torch.save(shadow_training_set, os.path.join(shadowset_path, "Shadowset_" + str(cls) + ".pt"))
        print("Save {} successfully!\n".format("Shadowset_" + str(cls) + ".pt"))

    # save shadow_training_set label 9
    shadow_training_set = shadow_dataset(trainset_size=args.trainset_size,
                                         net=target_net,
                                         fix_class=10,
                                         initial_record=initial_record,
                                         k_max=8,
                                         in_channels=args.in_channels,
                                         img_size=args.image_size,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         device=device,
                                         num_label=num_labels)
    torch.save(shadow_training_set, os.path.join(shadowset_path, "Shadowset_" + str(9) + ".pt"))


    ################ 4. CIFAR-10 Shadow Model Training #############
    # 1. Form Synthetic Dataset and DataLoader
    all_synth_dataset, all_synth_dataloader = gen_shadow_trainset(path=shadowset_path,
                                                                  each_trainset_size=args.trainset_size // num_labels,
                                                                  num_labels=num_labels)
    # 2. Train shadow model
    train_labels_list, train_proba_list, test_labels_list, test_proba_list = \
        train_shadow_model(num_shadow_model=args.num_shadow_models,
                           all_synth_dataset=all_synth_dataset,
                           trainset_size=args.trainset_size,
                           net=target_net,
                           num_epochs=num_epochs,
                           criterion=criterion,
                           optimizer=optimizer,
                           device=device)


    ################ 5. CIFAR-10 Shadow Model Training #############
    # 1. Form attack_in_array and attack_out_array
    attack_in_array, attack_out_array = form_attack_trainset(train_labels_list=train_labels_list,
                                                             train_proba_list=train_proba_list,
                                                             test_labels_list=test_labels_list,
                                                             test_proba_list=test_proba_list)
    # 2. For attack_array
    attack_array = np.vstack((attack_in_array, attack_out_array))
    assert attack_in_array.shape[1] == attack_out_array.shape[1] == attack_array.shape[1]
    attack_label_array = attack_array[:, 0]
    label_count_dict = count_in_ndarray(attack_label_array)
    print("Labels count in attackset = ", label_count_dict)

    # 3. Split into k partitions with each label
    split_label_list = split_attackset_by_label(attack_array=attack_array,
                                                num_labels=num_labels)
    # !Be aware: the length of item in split_label_list is not the same!

    # 4. Train k attack models
    rf = RandomForestClassifier(n_estimators=100)   # Create a RandomForest Classifier
    train_attack_model(attack_model=rf,
                       split_label_list=split_label_list)


if __name__ == '__main__':

    main()