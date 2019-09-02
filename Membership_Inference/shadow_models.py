# coding=utf8
"""
@author: Yantong Lai
@date: 08/30/2019
@code description: It is a Python3 file to implement shadow models in Paper
`Membership Inference Attack against Machine Learning Models` by Shokri et al. in 17 S&P
"""

import numpy as np
import pandas as pd
import random

from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

np.set_printoptions(threshold=np.inf)