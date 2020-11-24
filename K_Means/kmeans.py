# -*- coding: utf-8 -*-

import math
from utilities import get_distance,Sample

class Cluster(object):
    """
        聚类
    """
    def __init__(self, samples):
        if len(samples) == 0:
            # 如果聚类中无样本点
            raise Exception("错误：一个空的聚类！")

        # print((samples))

        # 属于该聚类的样本点
        self.samples = samples

        # 该聚类中样本点的维度
        # print(samples[0].n_dim)
        self.n_dim = samples[0].n_dim

        # 判断该聚类中所有样本点的维度是否相同
        for sample in samples:
            if sample.n_dim != self.n_dim:
                raise Exception("错误： 聚类中样本点的维度不一致！")

        # 设置初始化的聚类中心
        self.centroid = self.cal_centroid()


    def __repr__(self):
        """
            输出对象信息
        """
        return str(self.samples)

    def update(self, samples): #
        """
            计算之前的聚类中心和更新后聚类中心的距离
        """

        old_centroid = self.centroid
        self.samples = samples
        self.centroid = self.cal_centroid()

        shift = get_distance(old_centroid, self.centroid)
        return shift

    def cal_centroid(self):
        """
           对于一组样本点计算其中心点
        """
        n_samples = len(self.samples)
        # 获取所有样本点的坐标（特征）
        coords = [sample.coords for sample in self.samples]

        unzipped = zip(*coords)
        # 计算每个维度的均值
        centroid_coords = [math.fsum(d_list)/n_samples for d_list in unzipped]

        return Sample(centroid_coords)



