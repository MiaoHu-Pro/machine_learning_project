import numpy as np
import pandas as pd

from logistic_regression_utilities import sigmoid_fun

class LogisticRegression(object):

    def __init__(self, alpha=0.01, maxIter=1000, optimizeType='stocGradDescent'):
        '''
        :param alpha: 学习步长
        :param maxIter: 迭代次数
        :param optimizeType: 优化方式
        '''

        self.alpha = alpha
        self.maxIter = maxIter
        self.optimizeType = optimizeType

        self.weights = 0

    def fit(self,train_x,train_y):

        num_sample ,num_features = train_x.shape

        weights = np.ones((num_features, 1)) #权值矩阵

        for k in range(self.maxIter): #迭代次数

            if self.optimizeType == "stocGradDescent": #随机梯度下降 ，每个样本进行梯度下降
                for i in range(num_sample):
                    output = sigmoid_fun(train_x[i]*weights)

                    error = train_y[i] - output
                    print("error : ",error)

                    weights += self.alpha * train_x[i].transpose() * error

            elif self.optimizeType == 'gradDescent':  # gradient descent algorilthm(批量梯度下降)

                output = sigmoid_fun(train_x * weights)
                error = train_y - output
                weights += self.alpha * train_x.transpose() * error

            else:
                raise NameError('Not support optimize method type!')

        self.weights = weights



    def predict(self,test_x):

        num_samples = test_x.shape[0]

        per_list = []
        for i in range(num_samples):

            predict = sigmoid_fun(test_x[i] * self.weights) > 0.5

            if predict:
                per_list.append(1)
            else:
                per_list.append(0)

        return per_list




