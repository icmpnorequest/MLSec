# coding=utf8
"""
@author: Yantong Lai
@date: 09/04/2019
@code description: It is a Python3 file to implement attack models in Paper
`Membership Inference Attack against Machine Learning Models` by Shokri et al. in 17 S&P
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from Membership_Inference.shadow_models import SplitTrainTest

np.set_printoptions(threshold=np.inf)

pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_colwidth', 10000)
pd.set_option('display.width',1000)

attack_file = "../Membership_Inference/attack_data.npy"
split_labels_file = "../Membership_Inference/attack_split_labels_data"


def SplitClassData(filename):
    """
    It is a file to split training data by class label.
    :param filename: attack training dataset
    :return: <List> split attack training data
    """
    attack_set = np.load(filename)                      # <List>
    attack_array = np.concatenate(attack_set)           # ndarray concatenation, attack_array.shape = (300, 5)

    class_labels = attack_array[:, 0]
    uniq_class = np.unique(class_labels)

    split_by_labels = []
    for i in uniq_class:
        temp = []
        for item in attack_array:
            if item[0] == i:
                temp.append(item)
        split_by_labels.append(np.array(temp))

    np.save(split_labels_file, split_by_labels)         # save as .npy file
    print("Save successfully!\n")

    return split_by_labels


if __name__ == '__main__':

    #################### Split class label ###################
    attack_by_labels = SplitClassData(attack_file)        # <List> object

    ##################### Train and Predict ##################
    for item in attack_by_labels:
        (X_train, y_train), (X_test, y_test) = SplitTrainTest(item)
        model = RandomForestClassifier(n_estimators=100)        # create classifier RandomForest
        model.fit(X_train, y_train)
        y_pred= model.predict(X_test)

        print("Class label = {}".format(item[0][0]))            # print prediction results
        print("Membership Inference precision score: {}".format(precision_score(y_test, y_pred)))
        print("Mermbership Inference recall score: {}".format(recall_score(y_test, y_pred)))
        print("Membership Inference F1 score: {}".format(f1_score(y_test, y_pred)))
        print("########################\n")