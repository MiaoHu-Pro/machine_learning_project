

import random
import numpy as np
import pandas as pd
import sys
from regressor_tree import RegressorTree
from random_forest_regressor import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

if __name__ == "__main__":

    dataset = pd.read_csv('./data_set/8.Advertising.csv',)
    data = np.array(dataset)
    X = data[:, :-1] #
    y = data[:, -1]
    train_X, test_X, train_y, y_true = train_test_split(X, y,test_size=1 / 4.)

    rt_model = RandomForestRegressor(n_estimators = 10,max_features= "auto")

    rt_model.fit(train_X,train_y)
    y_pred = rt_model.predict(test_X)
    print('RandomForestRegressor MSE : %.4f' % (mean_squared_error(y_true, y_pred)))
    print('RandomForestRegressor R^2 : %.4f' % (r2_score(y_true, y_pred)))
    print("\n")

    # rmse , r2 =  rt_model.score(test_X,y_true)
    #
    # print('RegressorTree RMSE : %.4f' % (rmse))
    # print('RegressorTree R^2 : %.4f' % (r2))
    # print("\n")
    #

    print("Regression OK...")

