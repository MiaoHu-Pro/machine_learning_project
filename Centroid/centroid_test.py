# -*- coding: utf-8 -*-
# @Time    : 2018/4/4 22:30
# @Author  : HuMiao
# @Email   : humiao001@163.com
# @File    : test.py
# @Software: PyCharm Community Edition

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from centroid_model import Centroid
import numpy as np
import pandas as pd
import pickle

#存储模型
def save_model(rf_model):
    with open('./model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)  # 保存模型


def load_model():
    # 重新加载模型进行预测
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)  # 加载模型
    return model




if __name__ == "__main__":

    data_train = pd.read_csv('./data_set/iris.csv', header=0)
    train_data = np.array(data_train)

    X = train_data[:, :-1]
    y = train_data[:, -1]

    X_train, X_test, y_train, y_true = train_test_split(X, y,test_size=1 / 3. , random_state=6)

    centroid_model = Centroid()

    centroid_model.fit(X_train, y_train)

    # 存储模型
    save_model(centroid_model)
    pre_lab = centroid_model.predict(X_test)
    print('y_true : ', y_true.tolist())
    print('pre_lab :', pre_lab)

    print('The accuracy was ', 100 * accuracy_score(y_true, pre_lab), '% on the test ')

