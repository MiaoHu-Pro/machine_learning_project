
from __future__ import division
from collections import Counter
import random
import numpy as np
import sys


# 创建数据集的随机子样本,有放回的抽样
def sampling_with_reset(X, y, ratio):

    sample_X = list()
    sample_y = list()

    n_sample = round(len(y) * ratio)  # round() 方法返回浮点数x的四舍五入值。
    n_sample = int(n_sample)#取整
    while len(sample_X) < n_sample:
        index = random.randrange(len(y))  # 有放回的随机采样，有一些样本被重复采样，从而在训练集中多次出现，有的则从未在训练集中出现，此则自助采样法。从而保证每棵决策树训练集的差异性
        sample_X.append(X[index])
        sample_y.append(y[index])
    return np.array(sample_X),np.array(sample_y)#返回时数组类型

# 创建数据集的随机子样本,有放回的抽样
def sampling_bagging(X, y):

    sample_X = list()
    sample_y = list()

    out_of_bag_x = list()
    out_of_bag_y = list()
    index_list = list()

    n_sample = len(y)
    while len(sample_X) < n_sample:
        index = random.randrange(len(y))  # 有放回的随机采样，有一些样本被重复采样，从而在训练集中多次出现，有的则从未在训练集中出现，此则自助采样法。从而保证每棵决策树训练集的差异性
        index_list.append(index)
        sample_X.append(X[index])
        sample_y.append(y[index])


    for i in range(n_sample):

        if i not in index_list:
            out_of_bag_x.append(X[i])
            out_of_bag_y.append(y[i])

    out_of_bag_X = np.array(out_of_bag_x)
    out_of_bag_y = np.array(out_of_bag_y)

    out_of_bag_data = np.column_stack((out_of_bag_X,out_of_bag_y))

    return np.array(sample_X),np.array(sample_y),out_of_bag_data#返回时数组类型



def entropy(Y):
    """ In information theory, entropy is a measure of the uncertanty of a random sample from a group. """
    
    distribution = Counter(Y)#Counter类的目的是用来跟踪值出现的次数。它是一个无序的容器类型，以字典的键值对形式存储，
    s = 0.0
    total = len(Y)
    for y, num_y in distribution.items():
        probability_y = (num_y/total)
        s += probability_y * np.log(probability_y)
    return -s


def information_gain(y, y_true, y_false):
    """ The reduction in entropy from splitting data into two groups. """
    return entropy(y) - (entropy(y_true)*len(y_true) + entropy(y_false)*len(y_false))/len(y)


# 计算信息增益率
def information_gain_ratio(y, y_true, y_false):
    pass

def gini(y):

    distribution = Counter(y)  # Counter类的目的是用来跟踪值出现的次数。它是一个无序的容器类型，以字典的键值对形式存储，
    s = 0.0
    total = len(y)
    for y_index, num_y in distribution.items():
        s += np.power(num_y / total, 2)

    return 1 - s

#与gini一样，调试使用，更简单的写法
def gini_enhance(y):

    '''
        同方法 gini
        简单写法
    '''
    # distribution = Counter(Y)  # Counter类的目的是用来跟踪值出现的次数。它是一个无序的容器类型，以字典的键值对形式存储，
    # s = 0.0
    # total = len(Y)
    # for y, num_y in distribution.items():
    #     s += np.power(num_y / total,2)
    #
    # return 1 - s

    #复杂写法

    # 类别有多少统计不同
    n_calss = list(set(y))  # [1,2]
    k_n_calss = len(n_calss) # 2


    n_calss_num = [0 for i in range(k_n_calss)]
    length = len(y)
    for i in range(k_n_calss):
        for j in range(length):
            if y[j] == n_calss[i]:
                n_calss_num[i] +=1

    # print(n_calss_num)
    temp = 0
    for i in range(k_n_calss):
        temp += np.power(n_calss_num[i]/length,2)

    gini = 1 - temp
    return gini

def Gini_D_A(y,y_true,y_false):

    #定义左右两部分gini系数
    gini_ture = 0
    gini_false = 0

    #array转为list
    y = y.tolist()
    y_true = y_true.tolist()
    y_false = y_false.tolist()
    #计算长度和左右两部分数据比例
    len1 = len(y_true)
    len2 = len(y_false)
    len_y = len(y)
    ratio_1 = len1/len_y
    ratio_2 = len2 /len_y

    #y_true 部分的gini
    k_y_true = list(set(y_true))
    num_k_true = list()
    for i in range(len(k_y_true)):
        num_k_true.append(y_true.count(k_y_true[i]))
    for i in range(len(num_k_true)):
        gini_ture +=  np.power(num_k_true[i]/len1,2)
    gini_ture = 1 - gini_ture

    #y_false 部分的gini
    k_y_false = list(set(y_false))
    num_k_false = list()
    for i in range(len(k_y_false)):
        num_k_false.append(y_false.count(k_y_false[i]))
    for i in range(len(num_k_false)):
        gini_false += np.power(num_k_false[i] / len2, 2)
    gini_false = 1 - gini_false


    gini = ratio_1*gini_ture + ratio_2 * gini_false

    return  gini



class Leaf(object):
    """叶子节点，记录上节点分裂特征f，以及该叶节点中f的取值范围"""

    def __init__(self,labels, feature_index, max_value,min_value,
                 current_feature_index,current_max_value,current_min_value,
                 select_feature,leaf_data_set,sample_num, prior_node):

        self.labels = labels #该节点代表的标签
        self.fa_feature_index = feature_index #父节点的分裂特征 f
        self.fa_max_value = max_value         #叶节点f的最大值
        self.fa_min_value = min_value         #叶节点f的最小值

        #记录当前节点的特征
        self.current_feature_index = current_feature_index
        self.current_feature_max_value = current_max_value
        self.current_feature_min_value = current_min_value

        self.select_feature = select_feature

        self.sample_num = sample_num
        self.leaf_data_set = leaf_data_set

        self.prior_node = prior_node  # 前一个节点

class Head(object):
    def __init__(self,prior_node=None,branch_true = None,branch_false = None):
        self.prior_node = prior_node
        self.branch_true = branch_true #左孩子
        self.branch_false = branch_false #右孩子



class Node(object):

    """ 决策树中的节点. """

    def __init__(self, feature_index, fa_feature_index ,threshold,max_value,min_value,
                    fa_max_value,fa_min_value, gini_coefficient,node_data_set =None,prior_node =None, branch_true =None, branch_false =None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.gini_coefficient = gini_coefficient

        self.prior_node = prior_node #前一个节点
        self.branch_true = branch_true #左孩子
        self.branch_false = branch_false #右孩子

        self.max_value = max_value
        self.min_value = min_value

        #记录父节点的分裂特征；
        self.fa_feature_index = fa_feature_index
        self.fa_feature_max_value = fa_max_value
        self.fa_feature_min_value = fa_min_value

        self.node_data_set = node_data_set



#想打印树结构
def dran_tree(trunk):
    node = trunk

    if isinstance(node, Node):

        print('最优分割特征,最优分割特征值,对应的gnin系数 :',node.feature_index,' , ',node.threshold,' , ',node.gini_coefficient)
        print('最优分割特征',node.feature_index,'取值范围','[',node.min_value,' ,',node.max_value ,']')

        print('父亲节点最优分割特征', node.fa_feature_index, '取值范围', '[', node.fa_feature_min_value, ' ,', node.fa_feature_max_value, ']')

        print('父亲节点 ',node.prior_node)
        print('当前节点', node)
        # print('当前节点的样本\n',node.node_data_set)
        # print('当前节点的样本\n', node.node_data_set[node.node_data_set[:, node.feature_index].argsort()] ) # 安装指定的列排序


    if isinstance(node.branch_true, Node):
        print('left:')
        dran_tree(node.branch_true)
    else:
        leaf = node.branch_true
        print('left 叶子节点 :',leaf.labels)
        print('父分裂特征', leaf.fa_feature_index, '取值范围', '[', leaf.fa_min_value, ' ,', leaf.fa_max_value,']')
        print('随机选取的特征', leaf.current_feature_index, '取值范围', '[', leaf.current_feature_min_value, ' ,',
              leaf.current_feature_max_value, ']')
        print('该叶子节点的样本数 ：', leaf.sample_num)
        print('该分支已选的特征 ：', leaf.select_feature)
        print('父亲节点 ', leaf.prior_node)
        print('当前节点', leaf)
        print('该叶子节点的样本\n ',leaf.leaf_data_set)


        print('------end-------')

    if isinstance(node.branch_false, Node):
        print('right:')
        dran_tree(node.branch_false)
    else:
        leaf = node.branch_false
        print('right 叶子节点 :',leaf.labels)
        print('父分裂特征', leaf.fa_feature_index, '取值范围', '[', leaf.fa_min_value, ' ,', leaf.fa_max_value, ']')
        print('随机选取的特征', leaf.current_feature_index, '取值范围', '[', leaf.current_feature_min_value, ' ,',
              leaf.current_feature_max_value, ']')
        print('该叶子节点的样本数 ：', leaf.sample_num)
        print('该分支已选的特征 ：',leaf.select_feature)
        print('父亲节点 ', leaf.prior_node)
        print('当前节点', leaf)
        print('该叶子节点的样本\n ', leaf.leaf_data_set)
        print('------end-------')

#先序遍历，记录节点

def pre_order(trunk,node_list):

    node = trunk

    if isinstance(node, Node):
        print('内节点')
        node_list.append(node)

    if isinstance(node.branch_true, Node):
        # print('left:')

        pre_order(node.branch_true,node_list)
    else:
        leaf = node.branch_true
        node_list.append(leaf)
        print('----left-叶子-end-------')

    if isinstance(node.branch_false, Node):
        # print('right:')

        pre_order(node.branch_false,node_list)
    else:
        leaf = node.branch_false
        node_list.append(leaf)
        print('----right-叶子-end-------')


# 中序遍历，记录节点
def in_order(trunk,node_list):

    node = trunk



    if isinstance(node.branch_true, Node):
        # print('left:')

        in_order(node.branch_true,node_list)
    else:
        leaf = node.branch_true
        node_list.append(leaf)
        # print('----left-叶子-end-------')

    if isinstance(node, Node):
        # print('内节点')
        node_list.append(node)


    if isinstance(node.branch_false, Node):
        # print('right:')

        in_order(node.branch_false,node_list)
    else:
        leaf = node.branch_false
        node_list.append(leaf)
        # print('----right-叶子-end-------')







# 技巧性find_split 函数

def find_split(X, y, criterion ,feature_indices):

    """ 选择最优的划分属性和属性值. """

    best_gain = 0 #初始信息增益的比较值是0
    big_data = sys.maxsize #初始一个很大的数，作为gini的比较对象
    best_feature_index = 0 #最好的分裂属性
    best_threshold = 0     #最好的分裂值

    X_data = np.column_stack((X,y))

    for feature_index in feature_indices: # 遍历所有的候选特征

        # values = sorted(set(X[:, feature_index]))# n个属性值，并排序

        data = X_data[X_data[:,feature_index].argsort()] #安装指定的列排序

        # print(data)

        row, column = data.shape
        values = list()
        for i in range(row):
            if i == row - 1:
                break

            if data[i,-1] != data[i + 1,-1]:
                # print('index :', i)
                value = (data[i,feature_index] + data[i + 1,feature_index]) / 2
                # print('value ', )
                values.append(value)


        for j in range(len(values)): # 有n-1个分割点
            # print(values[j], values[j+1])

            # threshold = (values[j] + values[j+1])/2 #分割点定义为来个不同属性值的均值
            threshold = values[j]
            X_true, y_true, X_false, y_false = split(X, y, feature_index, threshold) #将样本分成两部分

            #计算信息增益和gini系数
            #计算信息增益
            if criterion == 'entropy':
                gain = information_gain(y, y_true, y_false) # 使用信息增益
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold

            # 计算gini
            if criterion == 'gini':
                gini = Gini_D_A(y, y_true, y_false)  # 使用信息增益
                if gini < big_data:
                    big_data = gini
                    best_feature_index = feature_index
                    best_threshold = threshold

    max_value = np.max(X[:, best_feature_index])
    min_value = np.min(X[:, best_feature_index])

    return best_feature_index, best_threshold,max_value,min_value,big_data #返回最佳分割属性和最佳分割属性值


##原始的find_split 函数
def find_split_备份(X, y, criterion ,feature_indices):

    """ 选择最优的划分属性和属性值. """

    best_gain = 0 #初始信息增益的比较值是0
    big_data = sys.maxsize #初始一个很大的数，作为gini的比较对象
    best_feature_index = 0 #最好的分裂属性
    best_threshold = 0     #最好的分裂值

    for feature_index in feature_indices: # 遍历所有的候选特征
        values = sorted(set(X[:, feature_index]))# n个属性值，并排序

        for j in range(len(values) - 1): # 有n-1个分割点
            threshold = (values[j] + values[j+1])/2 #分割点定义为来个不同属性值的均值
            X_true, y_true, X_false, y_false = split(X, y, feature_index, threshold) #将样本分成两部分

            #计算信息增益和gini系数
            # 计算信息增益
            if criterion == 'entropy':
                gain = information_gain(y, y_true, y_false) # 使用信息增益
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold

            # 计算gini
            if criterion == 'gini':
                gini = Gini_D_A(y, y_true, y_false)  # 使用信息增益
                if gini < big_data:
                    big_data = gini
                    best_feature_index = feature_index
                    best_threshold = threshold

    max_value = np.max(X[:, best_feature_index])
    min_value = np.min(X[:, best_feature_index])

    return best_feature_index, best_threshold,max_value,min_value,big_data #返回最佳分割属性和最佳分割属性值




def split(X ,y ,feature_index , threshold ):

    """ 样本集划分为两部分，分别是大于threshold 和 小于 threshold. """

    X_true = []
    y_true = []
    X_false = []
    y_false = []

    #一个数据集分成两个部分
    for j in range(len(y)):
        if X[j][feature_index] <= threshold:
            X_true.append(X[j])
            y_true.append(y[j])
        else:
            X_false.append(X[j])
            y_false.append(y[j])

    X_true = np.array(X_true)
    y_true = np.array(y_true)
    X_false = np.array(X_false)
    y_false = np.array(y_false)

    return X_true, y_true, X_false, y_false


