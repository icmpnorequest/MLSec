# coding=utf8
"""
@author: Yantong Lai
@date: 08/29/2019
@code description: It is a Python3 file to implement Algorithm 1 Data synthesis using the target model in Paper
`Membership Inference Attack against Machine Learning Models` by Shokri et al. in 17 S&P
"""

import numpy as np
import pandas as pd
import random

from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

np.set_printoptions(threshold=np.inf)

pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_colwidth', 10000)
pd.set_option('display.width',1000)

filename = "../Membership_Inference/synthetic_data.csv"
# Membership_Inference


'''
############### REFERENCE CODE by BielStela ####################
link: https://github.com/BielStela/membership_inference/blob/master/mblearn/data_synthesis.py

def generate_record(len_features, dtype, rang):
    """
    It is a function to generate a data record with length is the same as original data features.
    :param len_features: length of original data features
    :param dtype: data type of the original data features
    :param rang: data range
    :return: np.ndarray data record
    """
    try:
        if dtype not in ('int', 'float', 'bool'):
            raise ValueError("Parameter `dtype` must be 'bool', 'int' or 'float'")
        elif dtype is 'int':
            x = np.random.randint(low=rang[0], high=rang[1], size=len_features)
        elif dtype is 'float':
            x = np.random.uniform(low=rang[0], high=rang[1], size=len_features)
        elif dtype is 'bool':
            x = np.random.randint(low=0, high=2, size=len_features)
    except Exception as e:
        print("Exception: ", e)

    return x.reshape(1, -1)


def modify_record(record, k, dtype, rang):
    """
    It is a function to randomly modify k features
    :param record: generated data record
    :param k: number of features to be modified
    :param dtype: data type
    :param rang: data range
    :return: np.ndarray modified data record
    """
    idx_to_modify = np.random.randint(low=0, high=record.shape[1], size=k)
    print("index to be modified = ", idx_to_modify)
    # record.shape[1] = len_features
    new_gen_feats = generate_record(k, dtype=dtype, rang=rang)
    record[0, idx_to_modify] = new_gen_feats

    return record
'''


def RandRecord(data, k):
    """
    It is a function to randomize k features
    :param data: <np.ndarray> data record
    :param k: number of features to modify
    :return: <np.ndarray> modified data record
    """
    if k == 0:
        # Initialize a data record
        data = np.random.uniform(low=0, high=1, size=data.shape[1])
    elif k < 0:
        raise ValueError("k < 0!")
    else:
        idx_to_modify = np.random.randint(low=0, high=data.shape[1], size=k)
        new_feats = np.random.uniform(low=0, high=1, size=k)
        data[0, idx_to_modify] = new_feats

    return data.reshape(1, -1)


def Synthesize(data, target_model, fixed_class, k_max):
    """
    It is a function to implement Algorithm 1 in paper
    :param data: original scaled training dataset
    :param target_model: target model
    :param fixed_class: the class you want to change
    :param k_max: max radius of feature perturbation
    :param len_features: number of features
    :return: <np.ndarray> synthesis feature vector
    """
    ############### Initialization ##############
    x = RandRecord(data, k=0)       # initialize a record randomly
    y_c_current = 0                 # target models probability of fixed class
    j = 0                           # consecutive rejections counter
    k = k_max                       # search radius
    max_iter = 1000                 # max iter number
    conf_min = 0.8                  # min probability cutoff to consider a record member of the class
    rej_max = 5                     # max number of consecutive rejections
    k_min = 1                       # min radius of feature perturbation

    ################# Iteration #################
    for _ in range(max_iter):
        y = target_model.predict_proba(x)       # query the target model
        y_c = y.flat[fixed_class]               # get the index of fixed class in y

        # Phase 1: Search
        if y_c >= y_c_current:
            # Phase 2: Sample
            if y_c > conf_min and fixed_class == np.argmax(y):
                return x                        # synthetic data
            x_new = x
            y_c_current = y_c                   # renew variables
            j = 0
        else:
            j += 1
            if j > rej_max:                     # many consecutive rejects
                k = max(k_min, int(np.ceil(k / 2)))
                j = 0
        x = RandRecord(x_new, k)

    return False


def GenerateTrainSet(num_synth, data, target, target_model, k_max):
    """
    It is a function to generate training dataset for shadow models using synthetic data
    :param num_synth: number of synthetic data for each class in target
    :param data: original scaled training dataset
    :param target: original target
    :param target_model: target model
    :param num_class: number of classes in target
    :param k_max: max value of k
    :return: <np.ndarray> training dataset for shadow models
    """
    num_class = target_model.n_classes_
    n_record = num_synth * num_class
    n_features = data.shape[1]
    uniq_target = np.unique(target)
    # Initialize res <np.ndarray> with [n_record, n_features]

    gen_data = np.zeros((n_record, n_features))
    gen_label = np.zeros((n_record, 1))

    all_zeros = np.zeros((1, n_features))

    cls = 0
    step = 0
    for i in range(n_record):
        print("i = ", i)
        step = i // num_synth
        print("step = ", step)
        cls = uniq_target[step]
        print("cls = ", cls)
        gen_label[i] = uniq_target[step]

        try:
            while True:
                x = Synthesize(data=data, target_model=target_model, fixed_class=cls, k_max=k_max)
                if x.all() == all_zeros.all():
                    print("x = ", x)
                    print("Repeat generating x!\n")
                    continue
                else:
                    print("x = ", x)
                    break
        except Exception as e:
            print(e)

        gen_data[i] = x
        print("gen_data[{}] = {}".format(i, gen_data[i]))
        print("gen_label[{}] = {}".format(i, gen_label[i]))
        print("############################\n")

    return np.hstack((gen_data, gen_label))


def Save2CSV(gen_array, filename):
    """
    It is a function to save generated array to csv
    :param gen_array: generate array of GenerateTrainSet
    :param filename: filename of file to be saved
    :return: csv file
    """
    df = pd.DataFrame(data=gen_array)
    df.to_csv(filename, index_label=False, index=False)


def FormatCSV(filename, data, target_model, k_max):
    """
    It is a function to format csv file.
    :param filename:
    :param data:
    :param target_model:
    :param k_max:
    :return:
    """
    df = pd.read_csv(filename)
    cls = 0

    for i in range(len(df)):
        if df.iloc[i, :-1].all() == 0.0:
            cls = int(df.iloc[i, -1])
            print("i = ", i)
            x = Synthesize(data=data, target_model=target_model, fixed_class=cls, k_max=k_max)
            while x is False:
                x = Synthesize(data=data, target_model=target_model, fixed_class=cls, k_max=k_max)
            print("x = ", x)
            df.iloc[i, :-1] = x
            print("df.iloc[{}] = \n{}".format(i, df.iloc[i]))
            print("######################################\n")
    df.to_csv(filename)


if __name__ == '__main__':

    ################# Load Data and Classifier ##############
    data, target = load_wine(return_X_y=True)
    # data.shape = (178, 13)

    # Count how many types of classes are in target
    n_classes = len(np.unique(target))
    print("n_classes = ", n_classes)

    # Count how many features are in data
    n_features = data.shape[1]
    print("n_features = ", n_features)


    ##################### Regularization ####################
    scaler = MinMaxScaler()
    data_std = scaler.fit_transform(data)


    ################### Create a classifier #################
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(data_std, target)


    ###################### Synthesize #######################
    # Generate k modified feature vector
    x = RandRecord(data_std, k=0)
    print("x = ", x)
    print("\n")

    # class we want the record to belong to
    fixed_class = 1

    x_synth = Synthesize(data_std, rf, fixed_class, k_max=3)
    print("x_synth = ", x_synth)
    print("x_synth.shape = ", x_synth.shape)
    # (1, 13)
    print('####################################\n')


    ############### Generate Training dataset ##############
    gen_array = GenerateTrainSet(num_synth=100, data=data_std, target=target, target_model=rf, k_max=3)
    # print("gen_array = ", gen_array)
    print("gen_array.shape = ", gen_array.shape)
    print("####################################\n")


    ####################### Save to CSV ###################
    Save2CSV(gen_array, filename)


    #################### Format CSV #################
    FormatCSV(filename, data=data_std, target_model=rf, k_max=3)