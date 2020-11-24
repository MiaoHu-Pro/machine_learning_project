# -*- coding: utf-8 -*-
# @Time    : 2018/4/12 10:54
# @Author  : HuMiao
# @Email   : humiao001@163.com
# @File    : kd_tree_test.py
# @Software: PyCharm Community Edition

import numpy as np

import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from kd_tree import find_split ,split

from kd_tree import KDTree

if __name__ == "__main__":

    # X = [[1,2,3,1],
    #      [3,4,6,1],
    #      [3,6,6,1],
    #      [3,5,9,2],
    #      [3,5,12,2],
    #      [3,5,13,2],]

    X= [[2, 3,1], [5, 4,1], [9, 6,1],[8.5, 6,1],[4, 7,1], [8, 1,1], [7, 2,1]]

    X = np.array(X)

    # data_train = pd.read_csv('./data_set/iris_1.csv', header=0)
    # train_data = np.array(data_train)

    # X = train_data[:, :-1]
    # y = train_data[:, -1]

    # X_train, X_test, y_train, y_true = train_test_split(X, y,test_size=1 / 3., random_state=6)
    #
    # train_set = np.column_stack((X_train, y_train))

    kd = KDTree()
    kd.build_tree(X)

    x = [[7,6, 1],
         [3,4.5,1]]
    test_x = np.array(x)
    # print(test_x[:,:-1])
    nearest = kd.search_neighbour(test_x)
    for i in range(len(test_x)):
        print(test_x[i] ,'--->',nearest[i])



