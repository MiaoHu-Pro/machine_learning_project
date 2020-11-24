

import random
import numpy as np
import pandas as pd
import sys
from regressor_tree import RegressorTree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_score

#
def cross_vscore(mode ,X, y,cv):

    scores = cross_val_score(mode ,X, y,cv=cv)

    print(scores)
    print(np.mean(scores))
    print("------- cross_vscore ---ok---")

if __name__ == "__main__":

    dataset = pd.read_csv('./data_set/8.Advertising.csv',)
    data = np.array(dataset)
    X = data[:, :-1] #
    y = data[:, -1]

    #交叉验证
    print("sklearn DecisionTreeRegressor ....")
    rt_model = DecisionTreeRegressor()
    cross_vscore(rt_model,X,y,10)

    train_X, test_X, train_y, y_true = train_test_split(X, y,test_size=1 / 4.)

    rt_model = RegressorTree(max_depth=10,max_features="auto",min_samples_split=3)

    rt_model.fit(train_X,train_y)
    y_pred = rt_model.predict(test_X)
    print('RegressorTree MSE : %.4f' % (mean_squared_error(y_true, y_pred)))
    print('RegressorTree R^2 : %.4f' % (r2_score(y_true, y_pred)))
    print("\n")

    # rmse , r2 =  rt_model.score(test_X,y_true)
    #
    # print('RegressorTree RMSE : %.4f' % (rmse))
    # print('RegressorTree R^2 : %.4f' % (r2))
    # print("\n")
    #

    print("Regression OK...")

