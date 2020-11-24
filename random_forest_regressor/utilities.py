
from collections import Counter
import random
import numpy as np
import pandas as pd
import sys

# 创建数据集的随机子样本,有放回的抽样
def random_sample_feature(X,y,bootstrap):

    sample_X = list()
    sample_y = list()
    #样本的选择
    n_sample = round(len(y) * bootstrap )# round() 方法返回浮点数x的四舍五入值。
    n_sample = int(n_sample)
    while len(sample_X) < n_sample:
        index = random.randrange(len(y))  # 有放回的随机采样，有一些样本被重复采样，从而在训练集中多次出现，有的则从未在训练集中出现，此则自助采样法。从而保证每棵决策树训练集的差异性
        sample_X.append(X[index])
        sample_y.append(y[index])

    """
    x_df = pd.DataFrame(sample_X)
    #特征的选择
    count  = round(column * max_feature)  # round() 方法返回浮点数x的四舍五入值。
    n_feature = int(count)#取整
    random_feature = random.sample(range(0,column),n_feature) # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素。

    sample_X = x_df[random_feature]#选择相应的特征（即去除没有被选择的特征）
    """

    return np.array(sample_X),np.array(sample_y)#返回时数组类型




def find_split(X, y, criterion ,feature_indices):

    #使均方误差最小化
    if criterion != "mse":

        raise ("请使用均方误差作为分裂评价标准")

    mse = sys.maxsize  # 初始一个很大的数，作为mse的比较
    best_feature_index = 0  # 最好的分裂属性
    best_threshold = 0  # 最好的分裂值
    sample_num,culumn = X.shape

    X_data = np.column_stack((X, y))

    for feature_index in feature_indices:  # 遍历所有的候选特征

        data = X_data[X_data[:, feature_index].argsort()]  # 安装指定的列排序=

        values = list() #这个特征的候选分割点
        for i in range(sample_num):
            if i == sample_num - 1:
                break

            if data[i, -1] != data[i + 1, -1]:

                value = (data[i, feature_index] + data[i + 1, feature_index]) / 2

                values.append(value)
        #按照候选分割点，进行样本的划分
        for j in range(len(values)):  # 有n-1个分割点

            threshold = values[j]
            X_true, y_true, X_false, y_false = split(X, y, feature_index, threshold)  # 将样本分成两部分

            temp_mse = ((y_true - np.mean(y_true)) ** 2).sum() + ((y_false - np.mean(y_false)) ** 2).sum()

            if temp_mse < mse:

                mse = temp_mse
                best_feature_index = feature_index
                best_threshold = threshold

    return best_feature_index, best_threshold,mse  # 返回最佳分割属性和最佳分割属性值

def split(X, y, feature_index, threshold):

    """ 样本集划分为两部分，分别是大于threshold 和 小于 threshold. """

    X_true = []
    y_true = []
    X_false = []
    y_false = []

    #一个数据集分成两个部分
    for j in range(len(y)):
        if X[j][feature_index] <= threshold:
            X_true.append(X[j])
            y_true.append(y[j])
        else:
            X_false.append(X[j])
            y_false.append(y[j])

    X_true = np.array(X_true)
    y_true = np.array(y_true)
    X_false = np.array(X_false)
    y_false = np.array(y_false)

    return X_true, y_true, X_false, y_false

class Leaf(object):
    """叶子节点，记录上节点分裂特征f，以及该叶节点中f的取值范围"""

    def __init__(self,estimated_value,leaf_data_set,sample_size):

        self.estimated_value = estimated_value #该节点代表的标签
        self.leaf_data_set = leaf_data_set
        self.sample_size = sample_size



class Node(object):

    """ 决策树中的节点. """

    def __init__(self, feature_index, min_mes,threshold,node_data_set,sample_size, branch_true =None, branch_false =None):

        self.feature_index = feature_index #最优属性
        self.threshold = threshold #切割点
        self.min_mes = min_mes #最小均方误差

        self.node_data_set = node_data_set #样本集
        self.sample_size = sample_size #样本量

        self.branch_true = branch_true  # 左孩子
        self.branch_false = branch_false  # 右孩子





def mse(y_true, y_pre):
    mse = sum((y_true - y_pre) ** 2)/len(y_pre)
    return mse

def R2(y_true,y_pre):
    return 1 - (sum((y_true - y_pre)**2)/ sum((y_true - y_true.mean())**2))

