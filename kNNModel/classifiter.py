#coding=gbk

'''
    ����������
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm


def classifiter_test(data_set_train,data_set_test):


    train_X  = data_set_train[:,:-1]
    train_y  = data_set_train[:,-1]

    test_X = data_set_test[:,:-1]
    y_true = data_set_test[:,-1]


    # train_X, test_X, train_y, y_true = train_test_split(X, y,test_size=1 / 3., random_state=0)



    # �������� gamma  C ����ͨ��ѧϰ�õ��ģ�������
    # ���Բ�ָ������Ĭ��ֵ
    svm_model = svm.SVC(gamma=0.001, C=100.)
    svm_model.fit(train_X, train_y)

    # # ѡ��LRģ��
    lr_model = LogisticRegression(solver = 'lbfgs',multi_class = 'multinomial')
    # ѵ��ģ��
    lr_model.fit(train_X, train_y)

    #ѡ��RFģ��
    rf_model = RandomForestClassifier(n_estimators= 100)
    rf_model.fit(train_X,train_y)

    #ѡ��NNģ��
    nn_model = MLPClassifier()
    nn_model.fit(train_X,train_y)


    # KNNģ��

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(train_X,train_y)
    # * �ڲ��Լ��ϲ���ģ��

    y_pred_svm = svm_model.predict(test_X)
    print('y_pred_svm',y_pred_svm)
    y_pred_lr = lr_model.predict(test_X)
    print('y_pred_lr',y_pred_lr)
    y_pred_rf = rf_model.predict(test_X)
    print('y_pred_rf',y_pred_rf)
    y_pred_nn = nn_model.predict(test_X)
    print('y_pred_nn',y_pred_nn)
    y_pred_knn = knn_model.predict(test_X)
    print('y_pred_knn',y_pred_knn)

    # accuracy_score() ��һ����������ʵֵ���ڶ�����Ԥ��ֵ
    print('SVM classification results��', accuracy_score(y_true, y_pred_svm))
    print('LR classification results��', accuracy_score(y_true, y_pred_lr))
    print('RF classification results��', accuracy_score(y_true,y_pred_rf))
    print('NN classification results��', accuracy_score(y_true,y_pred_nn))
    print('KNN classification results��', accuracy_score(y_true, y_pred_knn))

if __name__ == '__main__':


    data_set_train = pd.read_csv('./data_set/data/P_train3.csv', header=0)
    data_train = np.array(data_set_train ,'float')

    data_set_test = pd.read_csv('./data_set/data/P_test3.csv', header=0)
    data_test = np.array(data_set_test, 'float')


    classifiter_test(data_train,data_test)






