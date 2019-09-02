# coding=utf8
"""
@author: Yantong Lai
@date: 09/02/2019
@code description: It is a Python3 file to implement utils in Paper
`Membership Inference Attack against Machine Learning Models` by Shokri et al. in 17 S&P
"""

import pandas as pd
import numpy as np


pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_colwidth', 10000)
pd.set_option('display.width',1000)


filename = "../Membership_Inference/synthetic_data.csv"


def formatCSV(filename):
    """
    It is a function to format csv file
    :param filename: file name
    :return: a formatted csv file
    """
    df = pd.read_csv(filename)
    # print(df.nonzero())


df = pd.read_csv(filename)
row = df.iloc[285, :13]
print("row = \n", row)

a = np.array((1,2,3,4,5,6,7,8,9,10,11,12,13), dtype=float)
print("a = ", a)

df.iloc[285, :13] = a
print("df.iloc = \n", df.iloc[285, :13])