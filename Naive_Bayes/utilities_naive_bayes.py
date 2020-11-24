# -*- coding: utf-8 -*-
# @Time    : 2018/4/5 9:37
# @Author  : HuMiao
# @Email   : humiao001@163.com
# @File    : utilities_naive_bayes.py
# @Software: PyCharm Community Edition

from collections import Counter
import random
import numpy as np

def comput_cls_pro(y_train,sample_num):
    '''
    计算每个类的概率
    :param y_train:
    :return:
    '''

    cls_pro = {}
    distribution = Counter(y_train)  # Counter类的目的是用来跟踪值出现的次数。它是一个无序的容器类型，以字典的键值对形式存储，
    total = sample_num
    for y, num_y in distribution.items():
        probability_y = (num_y / total)
        cls_pro[y] = probability_y

    return cls_pro


def comput_cls_nfeature_pro(X_train,y_train,n_cls,n_feature,sample_num):
    '''
    X_train, 训练数据x
    y_train, 训练数据y
    n_cls, 样本类别
    sample_num 样本个数

    '''
    cls_static = []
    for cls in n_cls:
        #处理每一类
        cls_sample = X_train[y_train == cls]
        total = cls_sample.shape[0]
        #该类的特征有多少取值，每个的概率
        feature_static = []
        for j in range(n_feature): #统计每一个特征
            feature_dic = {}
            feature_value = Counter(cls_sample[:,j]) #第j个特征
            for y, num_y in feature_value.items():
                probability_y = (num_y / total) #每个特征值在每个类中的概率
                feature_dic[y] = probability_y

            feature_static.append(feature_dic) #当前特征统计完成，放入list中

        cls_static.append(feature_static)

    return cls_static







