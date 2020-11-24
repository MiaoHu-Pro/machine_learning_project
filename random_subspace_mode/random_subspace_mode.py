# -*- coding: utf-8 -*-
# @Time    : 2018/6/25 21:09
# @Author  : HuMiao
# @Email   : humiao001@163.com
# @File    : random_subspace_mode.py
# @Software: PyCharm Community Edition
from collections import Counter
import random
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from utilities import random_sample_feature
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

#backtracking tree


class RandomSubspaceMode(object):

    def __init__(self,mode="dt",max_feature=0.5,n_estimators=100):

        self.base_mode = mode #基分类器
        self.max_feature = max_feature #训练特征数
        self.n_estimators = n_estimators #基分类器个数
        self.selected_feature = []
        self.subspace_mode = []

    def fit(self, train_X, train_y):

        row, column = train_X.shape
        selected_feature = []
        subspace_mode = []
        for i in range(self.n_estimators):
            #样本选择和特征选择，选择的特征被记录
            X_subset ,y_subset, random_feature = random_sample_feature(train_X, train_y,column,self.max_feature)

            selected_feature.append(random_feature) #记录这个分类器选择的特征，测试时，一一对应

            #供选择的基分类器,基分类器调用sklearn机器学习库
            if self.base_mode == "dt":
                base_mode = DecisionTreeClassifier()
            elif self.base_mode == "lr":
                base_mode = LogisticRegression()
            elif self.base_mode == "svm":
                base_mode = svm.SVC()
            elif self.base_mode == "knn":
                base_mode = KNeighborsClassifier()

            base_mode.fit(X_subset, y_subset)

            subspace_mode.append(base_mode) #树的集合

        self.selected_feature = selected_feature #保存各个分类器选择的特征
        self.subspace_mode = subspace_mode #保存基分类模型

    def __predict_(self,test_X):

        """ Predict the class of each sample in X.  private function"""
        n_samples = test_X.shape[0]

        predictions = np.empty([self.n_estimators, n_samples]) #返回一个初始化为随机值的二维数组
        for i in range(self.n_estimators):

            X = pd.DataFrame(test_X)
            X = X[self.selected_feature[i]]#每一个基分类器对应训练时的特征

            predictions[i] = self.subspace_mode[i].predict(X)
        return predictions
    def predict(self, X):
        """ 预测样本X的类别 """
        predictions = self.__predict_(X)
        return mode(predictions)[0][0] #取众数

if __name__ == "__main__":

    dataset = pd.read_csv('./data_set/glass.csv',header=None)
    data = np.array(dataset)
    X = data[:, :-1] #
    y = data[:, -1]
    train_X, test_X, train_y, y_true = train_test_split(X, y,test_size=1 / 4.)




    #随机子空间模型
    rsm = RandomSubspaceMode(mode="dt",max_feature=0.5,n_estimators=100)
    rsm.fit(train_X,train_y)
    y_pred_rsm = rsm.predict(test_X)
    accuracy = accuracy_score(y_true, y_pred_rsm)
    print("RSM classification results：%.4f" % (accuracy *100.0))


    #Random Forest 模型
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(train_X,train_y)
    y_pred_rf = rf_model.predict(test_X)
    accuracy = accuracy_score(y_true, y_pred_rf)
    print("RF classification results：%.4f" % (accuracy *100.0))

    #GBDT
    model = XGBClassifier(n_estimators=100)
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    accuracy = accuracy_score(y_true, y_pred)
    print("XGboost classification results：%.4f" % (accuracy *100.0)) #：%%.4f%



