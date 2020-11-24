import numpy as np
import matplotlib.pyplot as plt
import time


# calculate the sigmoid function
def sigmoid(seita_x):

    return 1.0 / (1 + np.exp(-seita_x))


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

    def fit(self,train_x, train_y):

        # calculate training time
        start_time = time.time()
        num_samples, num_features = np.shape(train_x)
        weights = np.ones((num_features, 1))
        # optimize through gradient descent algorilthm
        for k in range(self.maxIter):
            if self.optimizeType == 'gradDescent':  # gradient descent algorilthm(批量梯度下降)
                output = sigmoid(train_x * weights)
                error = train_y - output
                weights += self.alpha * train_x.transpose() * error

            elif self.optimizeType == 'stocGradDescent':  # stochastic gradient descent #(随机梯度下降)
                for i in range(num_samples):
                    output = sigmoid(train_x[i, :] * weights)
                    error = train_y[i, 0] - output
                    weights += self.alpha * train_x[i, :].transpose() * error

            elif self.optimizeType == 'smoothStocGradDescent':  # smooth stochastic gradient descent #批量随机梯度下降 mini-batch
                # randomly select samples to optimize for reducing cycle fluctuations
                dataIndex = range(num_samples)
                for i in range(num_samples):
                    alpha = 4.0 / (1.0 + k + i) + 0.01
                    randIndex = int(np.random.uniform(0, len(dataIndex)))
                    output = sigmoid(train_x[randIndex, :] * weights)
                    error = train_y[randIndex, 0] - output
                    weights += alpha * train_x[randIndex, :].transpose() * error
                    # del (dataIndex[randIndex])  # during one interation, delete the optimized sample
            else:
                raise NameError('Not support optimize method type!')

        print('Congratulations, training complete! Took %fs!' % (time.time() - start_time))
        self.weights = weights
        return weights

    # test your trained Logistic Regression model given test set
    def predict(self, test_x):

        numSamples, numFeatures = np.shape(test_x)
        per_list = []
        for i in range(numSamples):

            predict = sigmoid(test_x[i, :] * self.weights) > 0.5
            if predict:
                per_list.append(1)
            else:
                per_list.append(0)

        return per_list


    '''
    def testLogRegres(weights, test_x, test_y):
        numSamples, numFeatures = shape(test_x)
        matchCount = 0
        for i in range(numSamples):
            predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
            if predict == bool(test_y[i, 0]):

                 matchCount += 1
        accuracy = float(matchCount) / numSamples
        return accuracy
    '''

# show your trained logistic regression model only available with 2-D data
def showLogRegres(weights, train_x, train_y):
    # notice: train_x and train_y is mat datatype
    numSamples, numFeatures = np.shape(train_x)
    print(numFeatures)
    if numFeatures != 3:
        print("Sorry! I can not draw because the dimension of your data is not 2!")

        return 1

    # draw all samples
    for i in range(numSamples):
        if int(train_y[i, 0]) == 0:
            plt.plot(train_x[i, 1], train_x[i, 2], 'or')
        elif int(train_y[i, 0]) == 1:
            plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

    # draw the classify line
    min_x = min(train_x[:, 1])[0, 0]
    max_x = max(train_x[:, 1])[0, 0]
    weights = weights.getA()  # convert mat to array
    y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
    y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]

    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()