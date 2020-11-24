
from sklearn.model_selection import train_test_split
from KNNModel import KNNModel
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


if __name__ == '__main__':

    data_train = pd.read_csv('./data_set/apple.csv', header=0)
    train_data = np.array(data_train ,'float')

    X = train_data[:,:-1]
    y = train_data[:, -1]

    X_train, X_test, y_train, y_true = train_test_split(X, y,test_size=1 / 3., random_state = 1)
    knnmodel = KNNModel()
    knnmodel.fit(X_train,y_train)

    print('y_true : ', y_true.tolist())
    pre_lab  = knnmodel.predict(X_test)
    print('pre_lab: ',pre_lab)

    print('The accuracy was ',100 * accuracy_score(y_true,pre_lab),'% on the test ')


    #异常检测
    #读入异常点
    # novelty_set = pd.read_csv('./data_set/balance_2.csv', header=0)
    # novelty_data = np.array(novelty_set, 'float')
    # novelty_data_test = novelty_data[:,:-1]
    # pre_lab = knnmodel.novelty_detection(novelty_data_test)
    # print('novelty_detetion : ',pre_lab)









