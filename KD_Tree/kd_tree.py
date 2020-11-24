# -*- coding: utf-8 -*-
# @Time    : 2018/4/12 10:53
# @Author  : HuMiao
# @Email   : humiao001@163.com
# @File    : kd_tree.py
# @Software: PyCharm Community Edition

import random
import numpy as np
import math
import sys
import pandas as pd



class Leaf(object):

    def __init__(self,X,prior,flag = 0):

        self.X = X #当前叶子节点的样本集合
        self.flag = flag

        self.prior = prior #叶子节点只有前驱节点


class Node(object):
    def __init__(self,X,best_feature ,threshold,flag = 0,prior =None,branch_left=None,branch_right=None):

        self.X = X
        self.best_feature = best_feature#分割特征
        self.threshold = threshold #分割点
        self.flag = flag

        self.prior = prior #前驱节点
        self.branch_left = branch_left #左孩子
        self.branch_right = branch_right #右孩子


class KDTree(object):

    def __init__(self):
        self.trunk = None  # 保存树结构

    def build_tree(self,X):

        #根节点，寻找最优分割属性，和最优分割点
        #最优分割属性： 方差最大的属性 ，最优分割点是中位数

        self.trunk = self.__build_tree(X,None,0) #根节点没有父亲节点，所以father_node= None

        # print('v^v congratulate build tree ok .....')
        # dran_tree(self.trunk)
        print('\nv^v congratulate dran tree ok .....\n')


    def __build_tree(self,X,father_node,depth):

        if len(X) == 1:
            return Leaf(X=X, prior=father_node)

        #查找最优属性和分割点
        best_feature , best_threshold = find_split(X)
        #X_true(小于分割点), cut_off_threshold(在分割面上的点), X_false(大于分割面的点)
        X_true, cut_off_threshold, X_false =  split(X, best_feature, best_threshold)  # 分成左子树和右子树



        node = Node(X= cut_off_threshold,best_feature= best_feature,
                    threshold=best_threshold,prior=father_node)

        if len(X_true) == 0:
            node.branch_left = None
        else:
            node.branch_left = self.__build_tree(X_true,node,depth+1)
        if len(X_false) == 0:
            node.branch_right = None
        else:
            node.branch_right = self.__build_tree(X_false,node,depth+1)

        return node

    def search_k_neighbour(self,x_sample,k):
        '''
        :param x_sample:待查样本
        :param k: k个近邻
        :return:返回k个近邻样本和距离
        最近邻法和k-近邻法 KD树 （博客）
        '''
        pass

    def search_neighbour(self,test_x):
        """查找最近邻，返回最近邻点和距离"""

        # print('search neighbour of ' ,test_x ,'...')
        row,column = test_x.shape
        nearest = {}
        for i  in range(row):
            x = test_x[i,: -1]
            node = self.trunk
            current_node = None
            while isinstance(node, Node):  # 判断实例是否是这个类或者object是变量
                if x[node.best_feature] < node.threshold:

                    if node.branch_left != None: #左孩子不空
                        node = node.branch_left
                    else:
                        current_node = node
                        break
                elif x[node.best_feature] == node.threshold:
                    #等于，说明在分割平面上，则不用继续向下，当前就是终结点
                    current_node = node #记录当前节点，递归结束
                    break
                else:
                    if node.branch_right != None:

                        node = node.branch_right
                    else:
                        current_node = node
                        break

            if isinstance(node, Leaf): #到达叶子节点，叶子节点上只有一个实例
                current_node_x = node.X #当前叶子节点的实例

                # 当前最近邻节点
                current_nearest_sample = current_node_x
                # 当前最近邻节点与目标点的距离
                current_nearest_dis = euclidean_distance(x,current_nearest_sample[0,:-1])

                # print(current_nearest_sample)
                # print(current_nearest_dis)

                node.flag = 1
                #递归回溯
                current_nearest_sample, current_nearest_dis = backtracking_search(x,node,current_nearest_sample,current_nearest_dis)

                # print(sample, dis)

                 #搜索完之后，flag 赋予1

            else: #到达非叶子节点，，非叶子（内节点）至少一个实例点
                current_node_x = current_node.X #当前节点记录的样本集，是分割面上的点

                # 当前最近邻节点
                current_nearest_sample = None
                # 当前最近邻节点与目标点的距离
                current_nearest_dis = sys.maxsize
                for j in range(len(current_node_x)):
                    current_dis = euclidean_distance(x, current_node_x[j, :-1])
                    if current_dis < current_nearest_dis:
                        current_nearest_dis = current_dis
                        current_nearest_sample = current_node_x[j]

                # print(current_nearest_sample, ' ',current_nearest_dis)
            nearest[i] = current_nearest_sample

        return nearest

def backtracking_search(x,node,current_nearest_sample,current_nearest_dis):

    flag = 0 #记录node是哪个节点
    if node.prior.branch_left == node :
        flag = 0 #左
    else:
        flag = 1 #右边

    # _nearest_sample = current_nearest_sample
    # _nearest_dis = current_nearest_dis

    while node.prior != None:

        if node.prior.flag != 1:
            node = node.prior
            current_node_x = node.X  # 当前节点记录的样本集，是分割面上的点

            # 当前最近邻节点
            #current_nearest_sample = None
            # 当前最近邻节点与目标点的距离
            #current_nearest_dis = sys.maxsize
            for j in range(len(current_node_x)):
                current_dis = euclidean_distance(x, current_node_x[j, :-1])
                if current_dis < current_nearest_dis:
                    current_nearest_dis = current_dis
                    current_nearest_sample = current_node_x[j]

            node.flag = 1

            if current_nearest_dis > abs(node.threshold - x[node.best_feature]): #说明与切线相交，遍历另一个子节点

                if flag == 0 and node.branch_right != None:

                    #xinagyou

                    _dis = euclidean_distance(x, node.branch_right.X[0, :-1])
                    if _dis < current_nearest_dis:
                        current_nearest_dis = _dis
                        current_nearest_sample = node.branch_right.X

                    node.branch_right.flag = 1 #右节点访问了，置1
                    node = node.branch_right

                elif flag == 1 and node.branch_left != None:

                    _dis = euclidean_distance(x, node.branch_left.X[0, :-1])
                    if _dis < current_nearest_dis:
                        current_nearest_dis = _dis
                        current_nearest_sample = node.branch_left.X

                    node.branch_left.flag = 1  # 左节点节点访问了，置1
                    node = node.branch_left

            if node.prior != None and node.prior.branch_left == node: #记录这个节点时左节点还是右节点
                flag = 0  # node 是左节点
            else:
                flag = 1  # 右边

        else:
            node = node.prior

    # print(current_nearest_sample,current_nearest_dis)
    return  current_nearest_sample,current_nearest_dis

#计算欧式距离 方法1 省时，高效
def euclidean_distance(vector1, vector2):
    d = 0
    for a, b in zip(vector1, vector2):
        d += (a - b) ** 2
    return np.sqrt(d)
#计算曼哈顿距离
def manhattan_distance(vector1,vector2):

    return sum(abs(np.array(vector1) - np.array( vector2)))


def find_split(X):
    # print(X)
    row,column = X.shape
    mini_var = 0
    index = 0

    for i in range(column -1) :#最后一列是类别
        var = np.var(X[:,i])
        if var > mini_var:
            mini_var = var
            index = i

    best_feature = index #最优分割属性
    if len(X)%2 == 1:

        best_threshold = np.median(X[:,index])
    else:
        x_list = X[:, index].tolist()
        x_list = sorted(x_list)
        best_threshold = x_list[math.ceil(len(x_list)/2)]#下标向上取整，取之为中位数

    return best_feature , best_threshold #返回最优分割属性和最优分割点

def split(X,feature_index,threshold):

    """ 样本集划分为两部分，分别是大于threshold 和 小于 threshold. """

    X_true = []
    X_false = []

    cut_off_threshold = [] #分界面的样本

    row, column = X.shape
    #一个数据集分成两个部分
    for j in range(row):
        if X[j][feature_index] < threshold:
            X_true.append(X[j])

        elif X[j][feature_index] > threshold:
            X_false.append(X[j])
        else:
            cut_off_threshold.append(X[j])

    X_true = np.array(X_true)
    X_false = np.array(X_false)

    cut_off_threshold = np.array(cut_off_threshold)

    return X_true,cut_off_threshold ,X_false


#想打印树结构
def dran_tree(trunk):
    node = trunk

    if isinstance(node, Node):

        print('最优分割特征', node.best_feature, '分割点', node.threshold)
        print('当前节点记录的元素 \n',node.X)
        print('父亲节点', node.prior)
        print('当前节点', node)

    if isinstance(node.branch_left, Node):
        print('left:')
        dran_tree(node.branch_left)
    else:

        leaf = node.branch_left
        if leaf != None:
            print('左叶子节点')

            print('当前节点记录的元素 \n', leaf.X)

            print('父亲节点 ', leaf.prior)
            print('当前节点', leaf)

            print('------end-------')
        else:
            print('没有左叶子')
    if isinstance(node.branch_right, Node):
        print('right:')
        dran_tree(node.branch_right)
    else:
        leaf = node.branch_right
        if leaf != None:

            print('右叶子节点')
            print('当前节点记录的元素 \n',leaf.X)
            print('父亲节点',leaf.prior)
            print('当前节点', leaf)
            print('------end-------')
        else:
            print('没有右叶子')