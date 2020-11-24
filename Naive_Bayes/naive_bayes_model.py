# -*- coding: utf-8 -*-
# @Time    : 2018/4/5 9:36
# @Author  : HuMiao
# @Email   : humiao001@163.com
# @File    : naive_bayes_model.py
# @Software: PyCharm Community Edition
import numpy as np

from utilities_naive_bayes import comput_cls_pro ,comput_cls_nfeature_pro

class NaiveBayes(object):

    def __init__(self):

        self.P_Ck = {} #训练样本每个类的概率
        self.P_Ck_a = [] #每个类的每个维度值的概率 [{},{},{}]
        self.n_cls = []
        self.n_feature = None

        '''
        [
        [[{1:2/9},{2:3/9},{3:4/9}],
        [{S:},{M:},{L:}]], #每一个类的每个属性的每个值的概率

        [[{},{},{}],
        [{},{},{}]],

            ]
        '''


    def fit(self,X_train,y_train):
        '''
        计算先验概率和条件概率
        :param X_train:
        :param y_train:
        :return:
        '''

        #先验概率
        unique_cls_list = list(set(y_train.tolist()))
        self.n_cls = unique_cls_list
        sample_num = len(y_train) #训练样本数
        self.n_feature = len(X_train[0]) #特征数

        self.P_Ck = comput_cls_pro(y_train,sample_num) #每个类的概率
        # print(self.P_Ck)

        # 条件概率
        #统计每个类的每个属性的，每个值出现的概率

        self.P_Ck_a = comput_cls_nfeature_pro(X_train,y_train,self.n_cls,self.n_feature,sample_num)
        print('----------------------------')
        print('类别标签:\n',self.n_cls)
        print('统计每个类的概率：',self.P_Ck)
        print('类每个特征值的概率：',np.array(self.P_Ck_a))
        print('----------------------------')
    def predict(self,X_test):

        test_num = X_test.shape[0]
        pre = []
        for i in range(test_num):#每一个样本
            predict_cls_pro = []
            for c in range(len(self.n_cls)): #每一类别
                p = 1
                for j in range(self.n_feature):#每个特征

                    # print('----', X_test[i][j])
                    # print(self.P_Ck_a[c][j])

                    temp = self.P_Ck_a[c][j].get(X_test[i][j])
                    # print(temp)
                    if temp == None:
                        temp = 0
                    # print((temp))
                    p *= temp

                predict_cls_pro.append(self.P_Ck[self.n_cls[c]]*p)
            print('概率计算 ：',predict_cls_pro)
            index = predict_cls_pro.index(np.max(predict_cls_pro))

            pre.append(self.n_cls[index])

        return pre





