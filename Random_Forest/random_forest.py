
from __future__ import division
import numpy as np
import sys
from scipy.stats import mode
from utilities import sampling_with_reset,sampling_bagging
from decision_tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO

class RandomForestClassifier(object):

    """
        A random forest classifier.
    """

    def __init__(self, n_estimators=32,criterion = 'gini', max_features = None, max_depth=None,
        min_samples_split = 2,min_impurity_split= 1e-7, bootstrap = 1):
        """
        Args:
            n_estimators: 树的个数.
            criterion = gini or entropy
            max_features: 分裂节点随机选择的属性
            max_depth: 树的深度
            min_samples_split: 每个划分最少的样本数 min_samples_split = 2，小于2时，停止划分
            min_impurity_split：节点划分最小不纯度：1e-7：即当节点的gini系数小于1e-7时，
            或者 当前节点的信息增益为 0停止划分
            bootstrap: 抽取的样本比例
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.bootstrap = bootstrap #随机选择多少数据训练决策树
        self.forest = [] #
        self.rbt_forest = []
        self.n_calss_num = None
        self.n_calss = list()


    def fit(self, X, y):

        # print('===', sys._getframe().f_code.co_filename, sys._getframe().f_code.co_name, sys._getframe().f_lineno,"===")
        self.forest = []
        self.n_calss_num = len(set(y)) #有几个类
        self.n_calss = list(set(y)) #类标签集合

        for i in range(self.n_estimators):

            #随机的取数据 self.bootstrap 比率 ，表示抽取样本集的比例
            X_subset,y_subset = sampling_with_reset(X,y,self.bootstrap)
            ###########################################
            tree = DecisionTreeClassifier(self.max_features, self.criterion, self.max_depth, self.min_samples_split, self.min_impurity_split)
            #打印树的信息
            print('tree_'+str(i))
            tree.fit(X_subset, y_subset)

            self.forest.append(tree) #树的集合



    def __predict_(self,X):

        """ Predict the class of each sample in X.  private function"""
        n_samples = X.shape[0]
        n_trees = len(self.forest)
        predictions = np.empty([n_trees, n_samples]) #返回一个初始化为随机值的二维数组
        for i in range(n_trees):
            predictions[i] = self.forest[i].predict(X)

        return predictions


    def __predict_novelty(self,X,relaxation_factor,leaf_sample_num):

        """ Predict the class of each sample in X.  private function"""
        n_samples = X.shape[0]
        n_trees = len(self.forest)
        n_trees_sign = np.empty([n_trees, n_samples]) #每一个样本被标记
        predictions = np.empty([n_trees, n_samples]) #返回一个初始化为随机值的二维数组
        for i in range(n_trees):
            # predictions[i] = self.forest[i].predict(X)
            predictions[i] = self.forest[i].novelty_predict(X,relaxation_factor,leaf_sample_num)
            # n_trees_sign[i] = self.forest[i].sign

        # 打印每一个树，每一棵样本被标记为几次异常！
        # print('===', sys._getframe().f_code.co_filename, sys._getframe().f_code.co_name, sys._getframe().f_lineno,"===")
        # print(n_trees_sign)
        return predictions

    def predict_novelty(self, X,novelty_auto,relaxation_factor,leaf_sample_num):
        """ 预测样本X的类别 """
        test_X = X[:, :-1]
        y_true = X[:, -1]
        predictions = self.__predict_novelty(test_X,relaxation_factor,leaf_sample_num)
        print('===', sys._getframe().f_code.co_filename, sys._getframe().f_code.co_name, sys._getframe().f_lineno,
              "===")
        print(predictions)

        file_path = './data_set/predictions_proba.txt'
        fd = open(file_path,'w+')
        fd.write(str(predictions))
        fd.close()


        # print('===', sys._getframe().f_code.co_filename, sys._getframe().f_code.co_name, sys._getframe().f_lineno,"===")

        row ,column = predictions.shape
        lab_list = list()
        for i in range(column): #每一个样本
            row_list = list((predictions[:,i]))
            # print('predict_novelty of random_forest')
            # print(row_list)
            num = 0
            for j in range(len(row_list)):
                if row_list[j] == -1:
                    num +=1

            # print(y_true[i])#打印每一个样本的真实标签
            # print(num)

            if num >= self.n_estimators * novelty_auto:

                lab_list.append(-1)
            else:
                lab_list.append(mode(predictions[:,i])[0][0])

        # return mode(predictions)[0][0] #取众数
        return lab_list

    def predict(self, X):
        """ 预测样本X的类别 """

        predictions = self.__predict_(X)

        # print('===', sys._getframe().f_code.co_filename, sys._getframe().f_code.co_name, sys._getframe().f_lineno,"===")

        return mode(predictions)[0][0] #取众数

    def predict_2(self,X):

        predictions = self.__predict_(X)

        # print('===', sys._getframe().f_code.co_filename, sys._getframe().f_code.co_name, sys._getframe().f_lineno,"===")
        #返回预测的标签，和每棵树的预测标签
        return mode(predictions)[0][0] ,predictions # 取众数



    #两个返回值，第一个是类标签，第二个是概率向量
    def predict_proba(self,X):
        """ 预测样本X的类别概率"""

        predictions_proba = self.__predict_(X)

        #print\
        #
        file_path = './data_set/predictions_proba.txt'
        fd = open(file_path,'w+')
        fd.write(str(predictions_proba))
        fd.close()

        # print('标签向量：')
        # print(predictions_proba)

        #行表示树的个数，列表示样本个数
        tree_num , sample_num = predictions_proba.shape

        sample_list_vote = list()
        for i in range(sample_num):#统计每一个样本
            one_sample_vote = list()
            for j in range(self.n_calss_num):
                #计算每个类别有多少票
                one_sample_vote.append(predictions_proba[:,i].tolist().count(self.n_calss[j]))
            #向量中每个元素除以树的个数，对票数归一化
            one_sample_vote = [one_sample_vote[i] /tree_num for i in range(len(one_sample_vote))]
            sample_list_vote.append(one_sample_vote)

        pre_pro = np.array(sample_list_vote)

        lab_list = list() #预测的类别标签
        for i in range(sample_num):
            index = pre_pro[i].tolist().index(np.max(pre_pro[i]))
            lab_list.append(self.n_calss[index])

        lab_list = np.array(lab_list)
        print('predit print lab_list')
        print(lab_list)
        return pre_pro


