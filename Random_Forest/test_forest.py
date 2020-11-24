
import numpy as np
import pandas as pd
from csv import reader
from random_forest import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def test_forest():

    #加载数据
    train_set = pd.read_csv('./data_set/seeds.csv')
    data_set = np.array(train_set)

    X = data_set[:,:-1]
    y = data_set[:,-1]

    train_X, test_X, train_y, y_true = train_test_split(X, y,
                                                        test_size=1 / 3., random_state=7)
    # 加载模型
    rf_model = RandomForestClassifier(n_estimators= 3 , criterion='gini', max_features='sqrt', max_depth=20)

    rf_model.fit(train_X, train_y) #创建决策树集合

    print('rf_model.predict...begin...')
    pre_result = rf_model.predict(test_X)
    print('训练数据的预测概率向量：')
    print(pre_result)
    print('真实标签 ：')
    print(y_true)
    print('训练数据的预测准确度：')
    print(accuracy_score(y_true, pre_result))

if __name__ == '__main__':
    test_forest()