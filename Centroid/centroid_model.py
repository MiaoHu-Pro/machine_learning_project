# -*- coding: utf-8 -*-
# @Time    : 2018/4/4 22:30
# @Author  : HuMiao
# @Email   : humiao001@163.com
# @File    : test.py
# @Software: PyCharm Community Edition
from utilities_centroid import Centre,euclidean_distance,manhattan_distance

import numpy as np


class Centroid(object):
    '''
    基于中心点原形分类器
    '''

    def __init__(self):

        self.centre = []
        self.cls_num = None


    def fit(self,X_train, y_train):


        """
            返回训练集中每个类的统计参数
        """
        # 获取类别
        unique_cls_list = list(set(y_train.tolist()))
        #获取每一类的数据
        for cls in unique_cls_list:
            # 获取属于该类的样本
            samples_in_cls = X_train[y_train == cls] #获取该类别的数据
            cls_mean = np.mean(samples_in_cls,axis = 0) #获取每一类的中心

            self.centre.append(Centre(data=cls_mean,cls=cls))
        self.cls_num = len(self.centre)

    def predict(self,x_test):

        data = x_test
        data_num = data.shape[0]  # 多少样本
        pre_lab = []

        #计算与每一类中心点的距离
        for i in range(data_num):
            dis = []
            for j in range(self.cls_num):
                dis.append(manhattan_distance(data[i],self.centre[j].data))

            index = dis.index(np.min(dis)) #最小值对应的下标

            pre_lab.append(self.centre[index].cls)#根据下标获取类别

        return pre_lab













