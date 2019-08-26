# coding=utf8
"""
@author: Yantong Lai
@date: 08/22/2019
@code discription: It aims to complete data poisoning attack using SVM.
"""

from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


########## 1. Generate random datasets ##########
# Create 500000 separable points
X, y = make_blobs(n_samples=500000, centers=2, random_state=6)

# Spfrom sklearn.model_selection import train_test_splitlit train dataset and test dataset manually
X_train = X[:80]
y_train = y[:80]
X_test = X[80:]
y_test = y[80:]

# Using sklearn given API train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6)


def plot_dataset(X, y):
    """
    It is a function to plot dataset
    :param X: input data
    :param y: label
    """
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    plt.show()

plot_dataset(X, y)


########## 2. Normal SVM Classification ##########
normal_clf = svm.SVC(kernel='linear')
normal_clf.fit(X_train, y_train)
normal_pred = normal_clf.predict(X_test)
print("normal_pred = ", normal_pred)
print("#######################\n")


########## 3. Data Poisoning Attack ##########
def check_binary_classification(labels):
    """
    It is a function to check if it is binary classification.
    :param labels: Label of the dataset
    :return: True/False
    """
    label_counter = Counter(labels)
    label_counter_2list = list(label_counter.keys())
    if label_counter_2list is not None:
        if len(label_counter_2list) is 2:
            return True
        else:
            return False
    else:
        return None

def poison_label(labels, portion):
    """
    It is a function to poison label
    :param label: <np.ndarray>
    :return: poisoned label  <np.ndarray>
    """
    labels_index = np.arange(labels.size)
    # shuffle labels_index
    np.random.shuffle(labels_index)
    # get some shuffled label's index according to the portion
    select_labels_index = labels_index[:round(len(labels_index) * portion)]
    # print("select_labels_index =  ", select_labels_index)
    select_labels = labels[select_labels_index]
    # print("select_labels = ", select_labels)

    # binary classification
    if check_binary_classification(labels) is True:
        # print("labels_selected = ", labels[[item for item in select_labels_index]])
        reverse_labels_list = []
        for item in select_labels:
            reverse_labels_list.append(1 if item == 0 else 0)
        # print("reverse_labels_list = ", reverse_labels_list)
        for idx in range(len(select_labels_index)):
            labels[select_labels_index[idx]] = reverse_labels_list[idx]
        return labels
    else:
        return None

# Initialize y_poisoned
y_poisoned = y_train      # poison y_train
y_poisoned = poison_label(y_poisoned, portion=0.4)
# print("y_poisoned = ", y_poisoned)

poisoned_clf = svm.SVC(kernel='linear')
poisoned_clf.fit(X_train, y_poisoned)
poisoned_pred = poisoned_clf.predict(X_test)
print("poisoned_pred = ", poisoned_pred)
print("#######################\n")


########## 4. Compare Accuracy ##########
print("y_test = ", y_test)
print("#######################\n")

true_norm_acc = accuracy_score(y_test, normal_pred)
true_poisoned_acc = accuracy_score(y_test, poisoned_pred)
norm_poisoned_acc = accuracy_score(normal_pred, poisoned_pred)
print("true_norm_acc = ", true_norm_acc)
print("true_poisoned_acc = ", true_poisoned_acc)
print("norm_poisoned_acc = ", norm_poisoned_acc)