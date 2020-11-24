import numpy as np
import pandas as pd
from csv import reader
import  time


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
手写RF 与 sklearn库RF做比较
"""


def sklearn_test():


    train_set = pd.read_csv('./data_set/seeds.csv')
    train_data = np.array(train_set)

    X = train_data[:, :-1]
    y = train_data[:, -1]

    print(X.shape)
    train_X, test_X, train_y, y_true = train_test_split(X, y,
                                                        test_size=1 / 3., random_state= 8)

    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=10 , criterion='gini', max_features='sqrt')

    time_begin = time.time()
    rf_model.fit(train_X, train_y)
    time_end = time.time()

    print('fit time :',time_end - time_begin )

    pre_lab = rf_model.predict(test_X)

    score = accuracy_score(y_true, pre_lab)

    print(' sklearn.ensemble accuracy :', score)
    print("----------------------------")

def test():

    train_set = pd.read_csv('./data_set/seeds.csv')
    train_data = np.array(train_set)

    X = train_data[:, :-1]
    y = train_data[:, -1]

    print(X.shape)
    train_X, test_X, train_y, y_true = train_test_split(X, y,
                                                        test_size=1 / 3., random_state= 8)

    from random_forest import RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators= 10 , criterion = 'gini' , max_features='sqrt')

    time_begin = time.time()
    rf_model.fit(train_X, train_y)
    time_end = time.time()

    print('fit time :',time_end - time_begin)

    pre_lab = rf_model.predict(test_X)

    score = accuracy_score(y_true, pre_lab)
    print(' my random_forest accuracy :', score)
    print("----------------------------")


if __name__ == '__main__':


    sklearn_test()
    test()