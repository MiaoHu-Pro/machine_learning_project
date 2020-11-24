import  numpy as np
import pandas as pd
import random
def gini(y):

    # 类别有多少统计不同
    n_calss = list(set(y))  # [1,2]
    k_n_calss = len(n_calss) # 2


    n_calss_num = [0 for i in range(k_n_calss)]
    length = len(y)
    for i in range(k_n_calss):
        for j in range(length):
            if y[j] == n_calss[i]:
                n_calss_num[i] +=1

    print(n_calss_num)
    temp = 0
    for i in range(k_n_calss):
        temp += np.power(n_calss_num[i]/length,2)

    gini = 1 - temp
    print(gini)
    return gini

def test2(y):

    # 类别有多少统计不同
    n_calss = list(set(y))  # [1,2]
    k_n_calss = len(n_calss) # 2


    n_calss_num = [0 for i in range(k_n_calss)]
    length = len(y)

    feature_indices = random.sample(range(5),3) #[0,1,2,3,4] 抽3个

    print(feature_indices)


def test3():
    list_array = np.array(
   [[1, 0], [0, 3], [0, 2], [3, 2], [3, 0], [3, 1], [0, 3], [3, 0], [0, 2], [0, 3], [3, 1],
    [2, 1], [2, 3], [3, 1], [3, 2], [3, 2], [1, 3], [3, 1], [2, 3], [1, 2], [0, 3], [2, 0],
    [3, 2], [1, 0], [1, 2], [3, 1], [2, 3], [0, 3], [2, 1], [3, 2], [0, 3], [2, 3], [2, 0],
    [3, 0], [3, 2], [2, 1], [2, 1], [2, 3], [2, 3], [0, 1], [0, 2], [3, 1], [2, 1], [3, 1],
    [0, 1], [1, 2], [2, 3], [0, 3], [2, 1], [3, 1], [1, 3], [2, 1], [3, 0], [2, 1], [3, 1],
    [0, 3], [2, 3], [3, 2], [2, 3], [0, 3], [1, 0], [0, 1], [0, 3], [0, 3], [2, 3], [1, 0],
    [1, 3], [3, 2], [0, 3], [0, 3], [1, 0], [0, 3], [3, 0], [1, 2], [1, 2], [1, 0], [2, 3],
    [0, 3], [2, 3], [0, 3], [2, 3], [1, 2], [1, 3], [3, 1], [3, 1], [1, 2], [0, 3], [2, 0],
    [1, 2], [3, 1], [0, 1], [1, 0]])

    list_1 = list_array.tolist()

    unique_list = list()

    for  i in range(len(list_1)):
        if list_1[i] not in unique_list:
            unique_list.append(list_1[i])

    list_num = [0 for i in range(len(unique_list))
                ]
    for j in range(len(unique_list)):
        for i in range(len(list_1)):
            if unique_list[j] == list_1[i]:
                list_num[j] +=1

    print(unique_list)
    print(list_num)


    # print(len(list_1))
    #
    # list_2 = (list_1)
    # print(list_2)
    # list_2 = set(list_1)
    # print(list_2)

    # list_num = [0 for i in range(len(list_2))]
    #
    # for j in range(len(list_2)):
    #     for i in range(len(list_1)):
    #         if list_2[j] == list_1[i]:
    #             list_num[j] +=1
    #
    # print(len(list_1))
    #
    # print(list_2)
    # print(list_num)
#

def test3():
    n_features = 5
    select_list = [0,3]
    n_features_list = [i for i in range(n_features)]
    print(select_list)
    cha_list = list(set(n_features_list) - set(select_list))
    print(set(n_features_list) - set(select_list))
    print(cha_list)

    index = random.randrange(len(cha_list))
    print(cha_list[index])

    list_a = [1, 2, 3, 4]
    list_b = [1, 2, 3, 4,5]
    a = set(list_a)
    b = set(list_b)

    print(a == b)


def test4():
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    X = np.array([[1, 3], [3, 4], [1, 0], [3, 4],[1, 2],
                  [3, 2], [1, 2], [0, 4],[1, 2], [3, 4]])
    y = np.array([0, 0, 0, 0,0, 1, 1, 1, 1, 1])
    skf = StratifiedKFold(n_splits= 5)
    for train_index, test_index in skf.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(X_train,'\n',X_test)

        print(y_train,'\n', y_test)

        print("-----------------")



def test5():
    from sklearn.model_selection import KFold
    import numpy as np
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4],[1, 2], [3, 4], [1, 2], [3, 4],[1, 2], [3, 4]])
    y = np.array([0, 1, 0, 1,0, 1, 0, 1,0,1])
    kf = KFold(n_splits= 10)

    i = 0
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(X_train,'\n',y_train)

        print(X_test,'\n', y_test)
        i += 1
        print("-----------------"+str(i))



if __name__ == '__main__':

    # test3()
    # fa_freature_index = random.randrange(5)
    # print(fa_freature_index)


    # y = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
    # gini(y)
    # test2(y)

    # data_set = pd.read_csv('./data_set/seeds_1_3.csv')
    # data = np.array(data_set)

    # from decomposition_tool import pca_fun
    # pca_data_X_train = data[:,:-1]
    # pca_data_y_train = data[:,-1]
    # data = pca_fun(pca_data_X_train,3)
    # data = np.column_stack((data,pca_data_y_train))

    # data_show(data)

    # test3()

    test4()

    # test5()