from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import sys
from logistic_regression import LogisticRegression

from logistic_regression_utilities import load_data,cal_acc


def test():
    num_samples = 100
    dataIndex = range(num_samples)
    print(dataIndex)

    randIndex = int(np.random.uniform(0, len(dataIndex)))

    print(randIndex)

    del (dataIndex[:randIndex])

    print(dataIndex)

    # print('===', sys._getframe().f_code.co_filename, sys._getframe().f_code.co_name, sys._getframe().f_lineno,"===")

if __name__ == "__main__":

    X, y = load_data() #数据转化

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=1 / 3.,random_state=0)

    ## step 2: training...
    print( "step 2: training...")
    #alpha :学习步长
    #maxIter：迭代次数
    #optimizeType ：优化方式

    lr_model = LogisticRegression(alpha= 0.01,maxIter= 200,optimizeType='stocGradDescent')
    lr_model.fit(train_x, train_y)

    ## step 3: testing
    print("step 3: testing...")
    per_list = lr_model.predict(test_x)
    print('The classify accuracy is: %.4f%%' % cal_acc(per_list,test_y))




