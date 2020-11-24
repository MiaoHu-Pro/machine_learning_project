import numpy as np
import pandas as pd
from random_forest import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold




import pickle

class NoveltyDetection(object):

    def __init__(self):
        self.k_class_ = list()
        self.list_class_ = None
        self.list_probability_mean_ = list()

        self.rf_model_ = None
        self.test_X_y = None
        self.rf_model = None

    def random_forest_classification(self,train_data):

        train_X = train_data[:, :-1]
        train_y = train_data[:, -1]

        print("train_X ......type.....")
        print(type(train_X))

        self.k_class_ = list(set(train_y)) #

        rf_model = RandomForestClassifier(n_estimators= 10 , criterion='gini',max_features='sqrt',max_depth=20)

        rf_model.fit(train_X, train_y)
        # 保存模型
        self.rf_model = rf_model
        save_model_rf_model(rf_model)


    def nolvety_detection(self, test_data, novelty_auto = 0.5,relaxation_factor = 0.01,leaf_sample_num = 2):

        #组合测试数据
        # test_data = np.vstack((test_data,self.test_X_y))

        test_X = test_data[:, :-1]
        y_true = test_data[:, -1]

        # 获取随机森林模型
        result_pro = self.rf_model.predict_novelty(test_data, novelty_auto, relaxation_factor, leaf_sample_num)
        print('---------nolvety_detection-----------')
        print(result_pro)
        # 统计计算
        #=================================================
        length = len(result_pro)
        #异常检测时，预测概率
        print("异常检测时，预测概率 :\n")

        data_path = './data_set/true_think_novelty.csv'

        data_file = open(data_path, 'w+', newline='')
        import csv
        write_file_to_novelty = csv.writer(data_file)

        data_path_rigth = './data_set/novelty_think_right.csv'

        data_file_rigth = open(data_path_rigth, 'w+', newline='')

        write_file_to_right = csv.writer(data_file_rigth)

        nolvety_num = 0  # 异常点的总个数
        detection_nolvety_num = 0  # 是异常点，切被检测出来

        true_classification = 0 #正常点正确分类
        true_think_nolvety = 0 #正常点误认为是异常点

        err_classification = 0

        for row in range(length):

            if y_true[row] not in self.k_class_ :
                #是异常点，是否正确的识别为异常点
                nolvety_num +=1
                if result_pro[row] == -1:
                    detection_nolvety_num += 1 #是异常点且正确的识别为异常点
                else:
                    #是异常点，，没有检测出来
                    print('异常的，没有检测出来，认为是正常的: ')
                    print(test_data[row],'--->',result_pro[row])
                    write_file_to_right.writerow(test_data[row])
            else:
                #不是异常点，检测分类是否正确
                if result_pro[row] == y_true[row]:
                    true_classification  += 1 #正确分类计数

                elif result_pro[row] == -1:
                    true_think_nolvety += 1 #正确的误认为是异常的
                    print ('正确的异常化 ：' ,test_X[row],y_true[row],'------>',result_pro[row])
                    print(test_data[row])
                    write_file_to_novelty.writerow(test_data[row]) #正常的异常化，进行输出
                else:
                    err_classification +=1
                    print('错误分类 ',y_true[row],'------>',result_pro[row])

        print('样本集中一共有'+str(length) + '个样本点')
        print('样本集中一共有'+str(nolvety_num) + '个异常点')
        print('检测出 ' + str(detection_nolvety_num) + '个异常点')
        recall = float((detection_nolvety_num) / nolvety_num )

        print('一共有'+str(length - nolvety_num )+'个正常点')
        print('是正常点，且正确分类 ：',true_classification)
        print('是正常点，误认为是异常点 ：', true_think_nolvety)
        print('错误分类 ：', err_classification)
        accuracy = (detection_nolvety_num + true_classification) /length
        #是异常点且正确识别 / 识别的异常点总数
        precision = (detection_nolvety_num) /(detection_nolvety_num + true_think_nolvety)

        return recall,precision,accuracy

def save_model_rf_model(rf_model):
    with open('./rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)  # 保存模型

def load_model_rf_model():
    # 重新加载模型进行预测
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)  # 加载模型
    return model

def save_model_nd(novelty_detection):

    with open('./NoveltyDetection.pkl', 'wb') as f:
        pickle.dump(novelty_detection, f)  # 保存模型

def load_model_nd():
    # 重新加载模型进行预测
    with open('NoveltyDetection.pkl', 'rb') as f:
        model = pickle.load(f)  # 加载模型
    return model

if __name__ == '__main__':

    # 导入正常类
    train_set = pd.read_csv('./data_set/seeds_1_3.csv')
    data_set = np.array(train_set)

    #导入异常类
    test_set = pd.read_csv('./data_set/seeds_2.csv')
    test_data = np.array(test_set)

    X = data_set[:, :-1]
    y = data_set[:, -1]

    skf = StratifiedKFold(n_splits = 10)
    recall_list = list()
    precision_list = list()
    accuracy_list = list()

    for train_index, test_index in skf.split(X,y):

        train_X, test_X = X[train_index], X[test_index]
        train_y, y_true = y[train_index], y[test_index]

        # 构造训练集合
        train_data = np.column_stack((train_X, train_y))

        # 构造测试集合
        another_test_data = np.column_stack((test_X, y_true))
        union_test_data = np.vstack((test_data, another_test_data))

        nd = NoveltyDetection()
        nd.random_forest_classification(train_data)
        novelty_auto = 0.2
        relaxation_factor = 0.01
        leaf_sample_num = 2
        recall, precision, accuracy = nd.nolvety_detection(union_test_data,novelty_auto,relaxation_factor,leaf_sample_num)

        #保存模型
        # save_model_nd(nd)
        recall_list.append(recall)
        precision_list.append(precision)
        accuracy_list.append(accuracy)

    print(recall_list)
    print(precision_list)
    print(accuracy_list)
    print('召回率 ：%f ' % np.mean(recall_list))
    print('准确率 ：%f ' % np.mean(precision_list))
    print('正确率 ：%f' % np.mean(accuracy_list))

