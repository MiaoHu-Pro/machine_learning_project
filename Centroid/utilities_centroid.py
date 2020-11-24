# -*- coding: utf-8 -*-
# @Time    : 2018/4/4 22:31
# @Author  : HuMiao
# @Email   : humiao001@163.com
# @File    : test.py
# @Software: PyCharm Community Edition
import numpy as np

#数据对象
class Centre(object):
    def __init__(self,data,cls):

        '''
        :param number: 样本标号
        :param data: 样本数据域
        :param cls: 样本标签
        '''

        self.data = data #样本数据域
        self.cls = cls   #样本标签

#计算欧式距离 方法1 省时，高效
def euclidean_distance(vector1, vector2):
    d = 0
    for a, b in zip(vector1, vector2):
        d += (a - b) ** 2
    return np.sqrt(d)
#计算曼哈顿距离
def manhattan_distance(vector1,vector2):

    return sum(abs(np.array(vector1) - np.array( vector2)))

