
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from knn_model import KNN
import numpy as np
import pandas as pd
import pickle

#存储模型
def save_model(rf_model):
    with open('./model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)  # 保存模型


def load_model():
    # 重新加载模型进行预测
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)  # 加载模型
    return model


if __name__ == "__main__":

    data_train = pd.read_csv('./data_set/iris.csv', header=0)
    train_data = np.array(data_train)

    X = train_data[:, :-1]
    y = train_data[:, -1]

    X_train, X_test, y_train, y_true = train_test_split(X, y,test_size=1 / 3., random_state=6)

    train_set = np.column_stack((X_train, y_train))
    knn_model = KNN(train_set)


    # 存储模型
    save_model(knn_model)
    pre_lab = knn_model.predict(X_test,k=5)
    print('y_true : ', y_true.tolist())
    print('pre_lab :', pre_lab)

    print('The accuracy was ', 100 * accuracy_score(y_true, pre_lab), '% on the test ')

