# -*- coding: utf-8 -*-
# @Time    : 2018/4/5 9:36
# @Author  : HuMiao
# @Email   : humiao001@163.com
# @File    : naive_bayes_test.py
# @Software: PyCharm Community Edition



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from naive_bayes_model import  NaiveBayes
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

    data_train = pd.read_csv('./data_set/train_data.csv', header=0)
    train_data = np.array(data_train)

    X = train_data[:, :-1]
    y = train_data[:, -1]
    # print(X)

    X_train, X_test, y_train, y_true = train_test_split(X, y,test_size=1 / 10., )

    naivebayes = NaiveBayes()

    naivebayes.fit(X,y)

    # 存储模型
    save_model(naivebayes)

    print('测试样本 :\n',np.column_stack((X_test,y_true)))
    pre_lab = naivebayes.predict(X_test)

    print('pre_lab :', pre_lab)


