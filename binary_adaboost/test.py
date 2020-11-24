# -*- coding: utf-8 -*-
# @Time    : 2018/7/1 10:28
# @Author  : HuMiao
# @Email   : humiao001@163.com
# @File    : test.py
# @Software: PyCharm Community Edition

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from adaboost import TwoAdaboostClassifier


def load_data():

    """
    加载数据，
    X加一列全1 项
    y 标签 0 -1 化
    最后 X ， y 矩阵化，然后返回
    :return:
    """

    # 读取数据
    data_train = pd.read_csv('./data_set/cancer_2.csv', header=None)
    data = np.array(data_train)
    X = data[:, :-1]
    y = data[:, -1]  #

    ## 把标签转为0 和 1
    lab_y = list(set(y))
    if len(lab_y) == 2:
        for i in range(len(y)):
            if y[i] == lab_y[0]:
                y[i] = -1
            else:
                y[i] = 1
    else:
        print('logistic regression 是二分类器，无法处理你的数据！')
        exit()

    # X添加1列
    x_1 = np.ones((len(y), 1))
    X = np.column_stack((x_1, X))

    # 矩阵化
    y = np.mat(y).transpose()
    X = np.mat(X)
    # X = np.array(X)
    # y = np.array(y)
    return X, y
if __name__ == "__main__":

    X , y = load_data()

    train_X ,test_X , train_y , y_true = train_test_split(X,y,train_size= 1/3.)

    tac = TwoAdaboostClassifier(base_estimator= DecisionTreeClassifier(max_depth= 5),
                          n_estimators= 10)
    tac.fit(train_X,train_y)

    pre_y = tac.predict(test_X)


    accuracy = accuracy_score(y_true,pre_y)

    print("accuracy : %0.4f"%(accuracy * 100))









