
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from decision_tree import  DecisionTreeClassifier
from utilities import sampling_bagging
import numpy as np
import pandas as pd


"""
对决策树进行测试
"""

def test():

    data_train = pd.read_csv('./data_set/iris_1_3.csv', header=0)
    train_data = np.array(data_train ,'float')

    #
    X = train_data[:,:-1]
    y = train_data[:, -1]

    X_train, X_test, y_train, y_true = train_test_split(X, y,test_size=1 / 3., random_state= 6)

    d_tree = DecisionTreeClassifier(criterion = 'gini')

    #使用有放回抽样，抽样数据进行训练树，包外数据进行验证

    X_subset, y_subset, out_of_bag_data = sampling_bagging(X_train,y_train)

    d_tree.fit(X_subset, y_subset)

    #使用袋外数据进行树的调整;

    print('y_true : ', y_true.tolist())
    pre_lab  = d_tree.predict(X_test)
    print('pre_lab: ',pre_lab.tolist())
    # print('test_data\n',np.column_stack((X_test,y_true)))
    print('The accuracy was ',100 * accuracy_score(y_true,pre_lab),'% on the test ')



def save_data_to_csv(test_data):
    # 数组数据保存到csv
    save_path = './data_set/test_data.csv'
    df = pd.DataFrame(test_data)  # data_set 转为dataFrame ，然后写入csv
    df.to_csv(save_path, sep=',', index=False, header=False)

if __name__ == "__main__":
    test()