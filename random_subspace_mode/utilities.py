# -*- coding: utf-8 -*-
# @Time    : 2018/6/25 21:08
# @Author  : HuMiao
# @Email   : humiao001@163.com
# @File    : utilities.py
# @Software: PyCharm Community Edition

from collections import Counter
import random
import numpy as np
import pandas as pd


# 创建数据集的随机子样本,有放回的抽样
def random_sample_feature(X,y,column,max_feature=0.5):

    sample_X = list()
    sample_y = list()
    #样本的选择
    n_sample = len(y)  # round() 方法返回浮点数x的四舍五入值。
    while len(sample_X) < n_sample:
        index = random.randrange(len(y))  # 有放回的随机采样，有一些样本被重复采样，从而在训练集中多次出现，有的则从未在训练集中出现，此则自助采样法。从而保证每棵决策树训练集的差异性
        sample_X.append(X[index])
        sample_y.append(y[index])

    x_df = pd.DataFrame(sample_X)
    #特征的选择
    count  = round(column * max_feature)  # round() 方法返回浮点数x的四舍五入值。
    n_feature = int(count)#取整
    random_feature = random.sample(range(0,column),n_feature) # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素。

    sample_X = x_df[random_feature]#选择相应的特征（即去除没有被选择的特征）

    return np.array(sample_X),np.array(sample_y),random_feature#返回时数组类型


