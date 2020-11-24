
from collections import Counter
import random
import numpy as np
import sys

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

