# coding=utf8
"""
@author: Yantong Lai
@date: 08/30/2019
@code description: It is a Python3 file to implement shadow models in Paper
`Membership Inference Attack against Machine Learning Models` by Shokri et al. in 17 S&P
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=np.inf)

pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_colwidth', 10000)
pd.set_option('display.width',1000)

filename = "../Membership_Inference/synthetic_data.csv"
attack_file = "../Membership_Inference/attack_data"


def SplitShadowSet(filename, k):
    """
    It is a function to split generated synthetic dataset into k parts.
    :param filename: the whole generated synthetic dataset (csv file)
    :param k: number of shadow models
    :return: k child datasets
    """
    df = pd.read_csv(filename)
    dataset = df.values[:, 1:]                      # get all data records, dataset.shape = (300, 14)
    dataset_reshape = dataset.reshape(1, -1)        # dataset_reshape.shape = (1, 4200), it's a 2D array

    num_child_record = (dataset.shape[0]) // k      # data record number of each child dataset
    num_child_features = dataset.shape[1]
    print("num_child_record = ", num_child_record)
    print("num_child_features = ", num_child_features)
    print("#############################\n")

    child_array = np.zeros((k, num_child_record, num_child_features))       # define a array to store child dataset
    dataset_idx = np.arange(len(dataset))

    for i in range(k):
        sample_idx = np.random.choice(dataset_idx, size=num_child_record, replace=False)    # sample_idx.shape = (60, )
        temp_array = np.zeros((num_child_record, num_child_features))       # Initialize an array to save child dataset
        for idx in range(len(sample_idx)):
            temp_array[idx] = dataset[sample_idx[idx]]
        child_array[i] = temp_array

    return child_array


def SplitTrainTest(child_dataset):
    """
    It is a function to split each child dataset into child train dataset and test dataset.
    :param child_dataset: <np.ndarray> child synthetic dataset.
    :return: X_train, y_train, X_test, y_test
    """
    X = child_dataset[:, :-1]
    y = child_dataset[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return (X_train, y_train), (X_test, y_test)


def ShadowModel(X_train, y_train, X, shadow_model):
    """
    It is a function to feed child shadow model with training dataset and predict with test dataset.
    :param X_train: training data record to feed in shadow model
    :param y_train: training class label
    :param X: testing data record (according to paper, it could be X_train OR X_test)
    :param shadow_model: shadow model
    :return: y_pred: prediction vector of shadow model
    """
    shadow_model.fit(X_train, y_train)
    y_pred = shadow_model.predict_proba(X)

    return y_pred


def FormAttackTrainSet(y, y_pred, label):
    """
    It is a function to form attack training set for attack model
    :param y: class label (y_train OR y_test)
    :param y_pred: prediction vector of shadow model
    :param label: "IN"(label = 1) / "OUT"(label = 0)
    :return: attack training dataset (y, y_pred, label)
    """
    if label == 1:
        label_array = np.ones(y.shape[0])
    else:
        label_array = np.zeros(y.shape[0])
    data_record = np.hstack((y.reshape(-1, 1), y_pred, label_array.reshape(-1, 1)))

    return data_record


def SaveAttackSet(attack_set, filename):
    """
    It is a function to save attack training dataset
    :param attack_set: <List> a list with attack training data
    :param filename: file to save attack_set
    """
    np.save(filename, attack_set)
    print("Save attack training dataset successfully!\n")


def LoadAttackSet(filename):
    """
    It is a function to load saved attack training dataset
    :param filename: the file saved attack training dataset
    :return: <List> attack_set
    """
    attack_set = np.load(filename + ".npy")
    print("Load attack training dataset successfully!\n")
    return attack_set


if __name__ == '__main__':

    ####################### Split Synthetic Data ####################
    num_shadow_models = 5                                                       # Definition: number of shadow models
    split_array = SplitShadowSet(filename, k=num_shadow_models)                 # split_array.shape =  (5, 60, 14)


    ####################### Create a Classifier ####################
    shadow_model = RandomForestClassifier(n_estimators=100)


    ###################### Train Shadow Model ######################
    y_pred = 0                                                                  # Definition: initialize y_pred
    attackSet = []

    for i in range(num_shadow_models):
        (X_train, y_train), (X_test, y_test) = SplitTrainTest(split_array[0])   # Split train set and test set

        # Prediction vector y_pred_(train/test) of shadow train dataset and test dataset
        y_pred_train = ShadowModel(X_train, y_train, X=X_train, shadow_model=shadow_model)  # y_pred_train.shape=(48, 3)
        y_pred_test = ShadowModel(X_train, y_train, X=X_test, shadow_model=shadow_model)    # y_pred_test.shape=(12, 3)

        ################ Form Attack Training Set #################
        train_record = FormAttackTrainSet(y=y_train, y_pred=y_pred_train, label=1)          # IN, label = 1
        test_record = FormAttackTrainSet(y=y_test, y_pred=y_pred_test, label=0)             # OUT, label = 0
        attack_record = np.vstack((train_record, test_record))
        attackSet.append(attack_record)


    ################ Save Attack Training Dataset #################
    SaveAttackSet(attackSet, attack_file)


    ################# Load Attack Training Dataset ################
    loadAttack = LoadAttackSet(attack_file)
    print(loadAttack)