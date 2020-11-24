# -*- coding: utf-8 -*-
# @Time    : 2018/7/1 10:59
# @Author  : HuMiao
# @Email   : humiao001@163.com
# @File    : adaboost.py
# @Software: PyCharm Community Edition

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from copy import deepcopy
from utilities import sample_bootstrap

class TwoAdaboostClassifier(object):

    def __init__(self, *args, **kwargs):
        if kwargs and args:
            raise ValueError(
                '''TwoAdaboostClassifier can only be called with keyword
                   arguments for the following keywords: base_estimator ,n_estimators,
                    learning_rate,algorithm,random_state''')
        allowed_keys = ['base_estimator', 'n_estimators', 'learning_rate', 'algorithm', 'random_state']
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise ValueError(keyword + ":  Wrong keyword used --- check spelling")

        n_estimators = 50 #基分类器的个数
        learning_rate = 1 #学习速率
        # algorithm = 'SAMME' #使用分类误差率更新分类器的权重
        random_state = None #随机数种子
        base_estimator = DecisionTreeClassifier() #基分类器

        if kwargs and not args: #（kwargs） 和 （not args）都是真
            if 'base_estimator' in kwargs:
                base_estimator = kwargs.pop('base_estimator')
            else:
                raise ValueError('''base_estimator can not be None''')

            if 'n_estimators' in kwargs:
                n_estimators = kwargs.pop('n_estimators')
            if 'learning_rate' in kwargs:
                learning_rate = kwargs.pop('learning_rate')
            # if 'algorithm' in kwargs:
            #     algorithm = kwargs.pop('algorithm')
            if 'random_state' in kwargs:
                random_state = kwargs.pop('random_state')

        self.base_estimator_ = base_estimator #基分类器
        self.n_estimators_ = n_estimators #基分类器个数
        self.learning_rate_ = learning_rate #学习率
        # self.algorithm_ = algorithm #更新基分类器权重的方式
        self.random_state_ = random_state #随机种子
        self.estimators_ = list() #基分类器集合

        # self.estimator_weights_ = np.zeros(self.n_estimators_) #分类器权重(am) (有em计算得到)
        self.estimator_weights_ = list()
        self.estimator_errors_ = np.ones(self.n_estimators_) #分类器错误率(em)

    def fit(self,X,y):

        self.n_samples = X.shape[0]

        # There is hidden trouble for classes, here the classes will be sorted.
        # So in boost we have to ensure that the predict results have the same classes sort

        # self.classes_ = sorted(list(set(y)))
        # self.n_classes_ = len(self.classes_)

        #初始化样本权重
        sample_weight = np.ones(self.n_samples) / self.n_samples

        for iboost in range(self.n_estimators_):
            # if iboost == 0:
            #     sample_weight = np.ones(self.n_samples) / self.n_samples

            sample_weight, estimator_weight, estimator_error = self.discrete_boost(X, y, sample_weight)

            # early stop
            if estimator_error is None: #当错误率为None，停止当前树的创建
                break

            # append error and weight
            self.estimator_errors_[iboost] = estimator_error #em

            # self.estimator_weights_[iboost] = estimator_weight #am
            self.estimator_weights_.append(estimator_weight)
            if estimator_error <= 0:
                break
            # 样本抽样，进行下一次的训练
            # X, y = sample_bootstrap(X,y,sample_weight)

        return self

    def discrete_boost(self,X,y,sample_weight):
        #训练，测试，更新样本权重和分类器权重
        sample_num = len(y)

        base_estimator = deepcopy(self.base_estimator_)#获取基分类器

        base_estimator.fit(X,y,sample_weight=sample_weight) #分类器训练

        pre_y = base_estimator.predict(X)#分类器预测

        incorrect = np.zeros((sample_num, 1))
        for i in range(sample_num):
            if y[i] != pre_y[i]:
                incorrect[i] = 1
        #分类误差率
        # print("-------------")
        # print(incorrect.T, sample_weight)

        em = np.dot(incorrect.T, sample_weight)
        # print("em",em)
        if np.isnan(em) or em == 0:
            return None, None, None
        #计算分类器的系数am

        am = 0.5 * (np.log((1-em)/em))
        # if np.isnan(am):
        #     return None, None, None

        # print("am",am)
        # if am <= 0:
        #     return None,None, None
        #根据am调整训练样本的权重sample_weight

        for i in range(sample_num):
            sample_weight[i] = sample_weight[i] * np.exp(-1*am*y[i]*pre_y[i])

        sample_weight = sample_weight/sum(sample_weight)

        self.estimators_.append(base_estimator)

        return sample_weight, am, em


    def predict(self,X):

        row,column = X.shape
        n_estimators = len(self.estimators_)
        print(n_estimators)
        pre_y = np.zeros((n_estimators,row))
        for i in range(n_estimators):
            pre_y[i] = self.estimators_[i].predict(X)

        y_hat = []
        # print(pre_y.shape)
        self.estimator_weights_ = np.array(self.estimator_weights_)
        for i in range(row):
            # print(self.estimator_weights_.T)
            # print(pre_y[:,i])

            lab = np.dot(self.estimator_weights_.T,pre_y[:,i])

            if lab > 0:
                y_hat.append(1)
            else:
                y_hat.append(-1)

        return y_hat































