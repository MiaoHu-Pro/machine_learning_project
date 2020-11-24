
from __future__ import division
from utilities import Node ,Leaf,find_split,split,R2,mse
import random
import numpy as np


class RegressorTree(object):

    def __init__(self,criterion="mse",max_features=None,max_depth=10,min_samples_split=3,):

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features

        self.trunk = None

    def fit(self,X,y):

        row , column = X.shape
        n_features = column #特征个数

        if self.max_features == "auto" or self.max_features == None:

           self.max_features = column

        elif self.max_features == "sqrt":

            self.max_features = np.sqrt(column)

        elif self.max_features == "log2":

            self.max_features = np.log2(column) + 1

        # 取整
        self.max_features = int(self.max_features)

        # 随机取特征
        feature_indices = random.sample(range(n_features), self.max_features) #候选特征

        self.trunk = self.__build_tree(X,y,n_features,feature_indices,0)

    def __build_tree(self,X,y,n_features,feature_indices,depth):

        node_data_set = np.column_stack((X, y))
        sample_size = len(y)

        if len(y) <= self.min_samples_split or (depth !=None and depth == self.max_depth) :


            estimated_value = np.mean(y) #

            leaf = Leaf(estimated_value= estimated_value,sample_size=sample_size,leaf_data_set= node_data_set)
            return leaf

        #寻找分裂属性和最优分裂点
        best_feature_index, threshold, min_mes = find_split(X, y, self.criterion, feature_indices)

        X_true, y_true, X_false, y_false = split(X, y, best_feature_index, threshold)  # 分成左子树和右子树



        node = Node(feature_index= best_feature_index,
                    threshold=threshold,
                    min_mes=min_mes,
                    sample_size=sample_size,
                    node_data_set = node_data_set)


        # # 随机的选特征
        feature_indices = random.sample(range(n_features), int(self.max_features))
        ## 递归的创建左子树
        node.branch_true = self.__build_tree(X_true, y_true,n_features, feature_indices,depth + 1)

        ## 随机的选特征
        feature_indices = random.sample(range(n_features), int(self.max_features))

        node.branch_false = self.__build_tree(X_false, y_false, n_features, feature_indices,depth + 1)

        return node


    def predict(self,X):


        """ 预测样本X的类别 """

        num_samples = X.shape[0]
        y = np.empty(num_samples)
        for j in range(num_samples):
            node = self.trunk

            while isinstance(node, Node): #判断实例是否是这个类或者object是变量
                if X[j][node.feature_index] <= node.threshold:
                    node = node.branch_true
                else:
                    node = node.branch_false

            if isinstance(node,Leaf):
                y[j] = node.estimated_value #不是node对象，就是一个预测值

        return y #返回样本点在这棵树上的预测标签

    def score(self,text_x,y_true):

        """
        :param text_x:
        :param y_true:
        :return: 返回评价指标 R2

        """
        y_pre = self.predict(text_x)

        return mse(y_true,y_pre),R2(y_true,y_pre)










