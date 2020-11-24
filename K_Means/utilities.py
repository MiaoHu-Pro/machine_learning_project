# -*- coding: utf-8 -*-
# @Time    : 2018/4/28 16:48
# @Author  : HuMiao
# @Email   : humiao001@163.com
# @File    : utilities.py
# @Software: PyCharm Community Edition



import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import codecs
import re
import numpy as np
import math

#坐标点类
class Sample(object):
    """
        样本点类
    """
    def __init__(self, coords):
        self.coords = coords    # 样本点包含的坐标
        self.n_dim = len(coords)    # 样本点维度

    def __repr__(self):
        """
            输出对象信息
        """
        return str(self.coords)

#欧式距离
def get_distance(a, b):
    """
        返回样本点a, b的欧式距离
        参考：https://en.wikipedia.org/wiki/Euclidean_distance#n_dimensions
    """
    if a.n_dim != b.n_dim:
        # 如果样本点维度不同
        raise Exception("错误: 样本点维度不同，无法计算距离！")

    acc_diff = 0.0
    for i in range(a.n_dim):
        square_diff = pow((a.coords[i]-b.coords[i]), 2)
        acc_diff += square_diff
    distance = math.sqrt(acc_diff)

    return distance


def loadDataSet(inFile):

    inDate = codecs.open(inFile, 'r', 'utf-8').readlines()
    dataSet = list()
    for line in inDate:

        line = line.strip()
        strList = re.split('[ ]+', line)  # 去除多余的空格

        numList = list()
        for item in strList:
            num = float(item)
            numList.append(num)

        dataSet.append(numList)

    return dataSet



def pca_fun(train_X, num):
    n_components = num  #
    pca = PCA(n_components=n_components, svd_solver='auto')
    X_pca = pca.fit_transform(train_X)

    # print '方差所占比例(贡献率)方差越大说明向量的权重越大：\n', pca.explained_variance_ratio_  # 返回各个成分各自的方差百分比(贡献率)
    # pca_ratio = pca.explained_variance_ratio_
    # print('pca_ratio :', pca_ratio, sum(pca_ratio))
    return X_pca

def show(data):
    # 可视化结果
    plt.subplot()
    x = []
    y = []
    color = 'm'
    for sample in range(len(data)):
        x.append(data[sample][2])
        y.append(data[sample][1])
    plt.scatter(x, y, c=color)
    plt.show()

def load_data():
    #
    inFile = "./data_set/out_gene_ncbi_ds1.txt"
    data = loadDataSet(inFile)

    data = pca_fun(np.array(data), 15) #原始数据降至15维
    show(data) #可视化

    sample_trans = [] #样本转为Sample类型
    for i in range(len(data)):
        sample_trans.append(Sample(data[i]))

    return sample_trans