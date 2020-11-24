

from regressor_tree import RegressorTree
import random
import numpy as np
import pandas as pd
from utilities import random_sample_feature,R2,mse


class RandomForestRegressor(object):
    def __init__(self,n_estimators=10,criterion="mse",max_features="auto",max_depth=5,
                 min_samples_split=3,bootstrap=1):

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap #随机选择多少数据训练决策树
        self.forest = [] #



    def fit(self,X,y):


        self.n_calss_num = len(set(y)) #有几个类
        self.n_calss = list(set(y)) #类标签集合

        row , column = X.shape

        for i in range(self.n_estimators):

            #随机的取数据 self.bootstrap 比率 ，表示抽取样本集的比例
            X_subset,y_subset = random_sample_feature(X,y,self.bootstrap)
            ###########################################
            tree = RegressorTree(max_features = self.max_features,max_depth=self.max_depth,min_samples_split=self.min_samples_split)

            #打印树的信息

            tree.fit(X_subset, y_subset)

            self.forest.append(tree) #树的集合

    def __predict_(self,X):

        """ Predict the class of each sample in X.  private function"""
        n_samples = X.shape[0]
        n_trees = len(self.forest)
        predictions = np.empty([n_trees, n_samples]) #返回一个初始化为随机值的二维数组
        for i in range(n_trees):

            predictions[i] = self.forest[i].predict(X)

        return predictions

    def predict(self, X):
        """ 预测样本X的类别 """

        predictions = self.__predict_(X)

        return np.mean(predictions,axis=0)


    def score(self,text_x,y_true):

        """
        :param text_x:
        :param y_true:
        :return: 返回评价指标 R2

        """
        y_pre = self.predict(text_x)

        return mse(y_true,y_pre),R2(y_true,y_pre)
