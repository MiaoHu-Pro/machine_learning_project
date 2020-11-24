
import numpy as np
import pandas as pd

def load_data():

    """
    加载数据，
    X加一列全1 项
    y 标签 0 -1 化
    最后 X ， y 矩阵化，然后返回
    :return:
    """

    # 读取数据
    data_train = pd.read_csv('./data_set/iris_1_3.csv', header=0)
    data = np.array(data_train)
    X = data[:, :-1]
    y = data[:, -1]  #

    ## 把标签转为0 和 1
    lab_y = list(set(y))
    if len(lab_y) == 2:
        for i in range(len(y)):
            if y[i] == lab_y[0]:
                y[i] = 0
            else:
                y[i] = 1
    else:
        print('logistic regression 是二分类器，无法处理你的数据！')
        exit()

    # X添加1列
    x_1 = np.ones((len(y), 1))
    X = np.column_stack((x_1, X))

    # 矩阵化
    y = np.mat(y).transpose()
    X = np.mat(X)
    return X, y

def sigmoid_fun(seitaX):

    return 1.0 / (1 + np.exp(-seitaX))

def cal_acc(true_labels, pred_labels):
    """
        计算准确率
    """
    n_total = len(true_labels)
    correct_list = [true_labels[i] == pred_labels[i] for i in range(n_total)]
    # correct_list = [ 1 for i in range(n_total) if true_labels[i] == pred_labels[i]]
    acc = sum(correct_list) / n_total
    return acc
