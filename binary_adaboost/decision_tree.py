
from __future__ import division
import random
import numpy as np
from scipy.stats import mode
from utilities import entropy,gini
from utilities import  Leaf,Node,Head
from utilities import dran_tree ,find_split ,split

class DecisionTreeClassifier(object):
    """
    决策树
    """
    def __init__(self, max_features = None ,criterion = 'gini', max_depth = 10,
                 min_samples_split = 2, min_impurity_split = 1e-7):
        """
        Args:
            max_features: 分裂节点随机选择的属性
            criterion = gini or enorpy
            max_depth: 决策树的最大深度
            min_samples_split: 每个划分最少的样本数 min_samples_split = 2，小于2时，停止划分
            min_impurity_split：节点划分最小不纯度：1e-7：即当节点的gini系数小于1e-7时，
            或者 当前节点的信息增益为 0停止划分
        """

        self.max_features = max_features #
        self.criterion = criterion #那种分裂指标
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.trunk = None #保存树结构
        self.select_feature = list()
        self.sample_num = list()
        self.gini_ = list()
        self.sign = list() # 记录  每一个样本被标记为几次异常！比较过的特征不在比较！
        self.pre_node_list = list()
        self.leaf_list = list()

        head = Head() #空节点
        self.head = head #给根节点定义一个头节点

    def fit(self, X, y):

        """
        训练函数
        """
        n_features = X.shape[1]

        #指定max_feature 的大小
        if self.max_features == 'sqrt':
            self.max_features = np.sqrt(n_features)
        elif self.max_features == 'log2':
            self.max_features = np.log2(n_features) + 1 #韩家伟书中用的是log2(x) + 1
        elif self.max_features == None:
            self.max_features = (n_features)

        # 取整
        self.max_features= int(self.max_features)

        # 随机取特征
        feature_indices = random.sample(range(n_features),self.max_features)

        # 根节点的父节点通过随机指定
        fa_freature_index = random.randrange(n_features)
        select_feature = list() #记录节点选择的特征

        self.trunk = self.build_tree(X, y, feature_indices,fa_freature_index,select_feature, self.head,0)

        # 画图
        print('按照树的先序遍历：')
        dran_tree(self.trunk)


    def build_tree(self, X, y, feature_indices,fa_feature_index,select_feature_fa, father_node,depth):
        """
        建立决策树
        X :
        y：
        feature_indices：随机选择的特征集合
        fa_feature_index：父节点选择的哪个特征作为分裂特征，、初始时为-1，
        depth ：树的深度
        select_feature_fa ：记录当前节点的父节点的最优分割属性
        """
        select_feature_fa.append(fa_feature_index)
        n_features = X.shape[1]
        n_features_list = [i for i in range(n_features)]
        #记录选择的特征
        self.select_feature.append(feature_indices)
        self.sample_num.append(len(y))

        node_data_set = np.column_stack((X, y))

        # 树终止条件
        if self.criterion == 'entropy':
            if depth is self.max_depth or len(y) < self.min_samples_split or entropy(y) is 0:
                return mode(y)[0][0]# 返回y数组的众数

        # 树终止条件
        if self.criterion == 'gini':
            temp_gini = gini(y)
            self.gini_.append(temp_gini)
            sample_num = len(y)
            if depth is self.max_depth or sample_num < self.min_samples_split or temp_gini < self.min_impurity_split:
            # if depth is self.max_depth or temp_gini < self.min_impurity_split:

                #所有的特征都已经被选择了，就随机选择一个特征，使得叶子节点构成双特征
                if set(n_features_list) == set(select_feature_fa):
                    index = random.randrange(len(n_features_list))
                    current_feature_index = n_features_list[index]
                    current_max_value = np.max(X[:, current_feature_index])
                    current_min_value = np.min(X[:, current_feature_index])

                else:
                    to_be_select = list(set(n_features_list) - set(select_feature_fa))
                    index = random.randrange(len(to_be_select))

                    current_feature_index = to_be_select[index]
                    current_max_value = np.max(X[:, current_feature_index])
                    current_min_value = np.min(X[:, current_feature_index])

                leaf = Leaf(mode(y)[0][0],fa_feature_index , np.max(X[:,fa_feature_index]),
                            np.min(X[:,fa_feature_index]),current_feature_index,current_max_value,
                            current_min_value,select_feature_fa,node_data_set,sample_num,prior_node= father_node)
                self.leaf_list.append(leaf)
                return leaf

        # feature_index最佳分割属性， threshold 最佳分割属性值,gini_ 系数
        feature_index, threshold, max_value ,min_value ,gini_ = find_split(X, y, self.criterion, feature_indices)

        fa_max_value = np.max(X[:, fa_feature_index])  # 该节点记录父节点分裂特征的最大值
        fa_min_value = np.min(X[:, fa_feature_index])  # 该节点记录父节点分裂特征的最小值

        X_true, y_true, X_false, y_false = split(X, y, feature_index, threshold)# 分成左子树和右子树

        # 没有元素
        if y_true.shape[0] is 0 or y_false.shape[0] is 0:

            if set(n_features_list) == set(select_feature_fa):
                index = random.randrange(len(n_features_list))
                current_feature_index = n_features_list[index]
                current_max_value = np.max(X[:, current_feature_index])
                current_min_value = np.min(X[:, current_feature_index])

            else:
                to_be_select = list(set(n_features_list) - set(select_feature_fa))
                index = random.randrange(len(to_be_select))

                current_feature_index = to_be_select[index]
                current_max_value = np.max(X[:, current_feature_index])
                current_min_value = np.min(X[:, current_feature_index])

            leaf = Leaf(mode(y)[0][0], fa_feature_index, np.max(X[:, fa_feature_index]), np.min(X[:, fa_feature_index]),
                        current_feature_index,current_max_value,current_min_value,select_feature_fa,node_data_set,prior_node= father_node,sample_num= 0)

            self.leaf_list.append(leaf)
            return leaf

        node = Node(feature_index=feature_index,
                    fa_feature_index = fa_feature_index,
                    threshold = threshold, max_value = max_value, min_value = min_value,
                    fa_max_value = fa_max_value, fa_min_value = fa_min_value,
                    gini_coefficient = gini_,
                    node_data_set = node_data_set)


        # # 随机的选特征
        n_features = X.shape[1]
        n_sub_features = int(self.max_features)
        #
        feature_indices = random.sample(range(n_features), n_sub_features)
        select_feature = list()
        select_feature += select_feature_fa  # 记录节点选择的特征
        ## 递归的创建左子树
        node.branch_true = self.build_tree(X_true, y_true, feature_indices,feature_index,
                                           select_feature,node,depth + 1)

        ## 随机的选特征
        feature_indices = random.sample(range(n_features), n_sub_features)
        # 递归的创建右子树
        select_feature = list()
        select_feature += select_feature_fa  # 记录节点选择的特征
        node.branch_false = self.build_tree(X_false, y_false, feature_indices,feature_index,
                                            select_feature,node,depth + 1)

        node.prior_node = father_node #指向前驱节点

        return node


    def recall_binary_tree_adjustment(self,node,adjustement_x):
        '''
        指定叶子节点，回溯该叶子节点
        :param node: 叶子节点，
        :param adjustement_x: 回溯的样本，调整内节点的特征范围
        :return:
        '''

        #
        if adjustement_x[node.fa_feature_index] <  node.fa_min_value:
            node.fa_min_value = adjustement_x[node.fa_feature_index]
        elif adjustement_x[node.fa_feature_index] >  node.fa_max_value:
            node.fa_max_value = adjustement_x[node.fa_feature_index]

        if adjustement_x[node.current_feature_index] < node.current_feature_min_value:
            node.current_feature_min_value = adjustement_x[node.current_feature_index]
        elif adjustement_x[node.current_feature_index] > node.current_feature_max_value:
            node.current_feature_max_value = adjustement_x[node.current_feature_index]

        while (node.prior_node != None) and (isinstance(node.prior_node, Head) == 0):  # 头结点不要遍历

            if adjustement_x[node.prior_node.fa_feature_index] < node.prior_node.fa_feature_min_value:
                node.prior_node.fa_feature_min_value = adjustement_x[node.prior_node.fa_feature_index]
            elif adjustement_x[node.prior_node.fa_feature_index] > node.prior_node.fa_feature_max_value:
                node.prior_node.fa_feature_max_value = adjustement_x[node.prior_node.fa_feature_index]

            if adjustement_x[node.prior_node.feature_index] < node.prior_node.min_value:
                node.prior_node.min_value = adjustement_x[node.prior_node.feature_index]
            elif adjustement_x[node.prior_node.feature_index] > node.prior_node.max_value:
                node.prior_node.max_value = adjustement_x[node.prior_node.feature_index]

            node = node.prior_node

        # print('-----一个叶子回溯结束--------\n')


    def recall_binary_tree(self):
        '''
        逐个回溯叶子节点
        :return:
        '''
        #
        leaf_num = len(self.leaf_list)

        for i in range(leaf_num):
            node = self.leaf_list[i]

            print('叶子节点标签:', node.labels)
            print('父亲节点最优分裂特征:', node.fa_feature_index, '取值范围:', '[', node.fa_min_value, ' ,',
                  node.fa_max_value, ']')
            print('随机选取的特征:', node.current_feature_index, '取值范围:', '[', node.current_feature_min_value, ' ,',
                  node.current_feature_max_value, ']')

            while (node.prior_node != None) and (isinstance(node.prior_node, Head) == 0):  # 头结点不要遍历
                print('\n内节点:')
                print('父亲节点最优分裂特征:', node.prior_node.fa_feature_index, '取值范围:', '[',
                      node.prior_node.fa_feature_min_value, ' ,',
                      node.prior_node.fa_feature_max_value, ']')
                print('最优分割属性:', node.prior_node.feature_index, '取值范围:', '[', node.prior_node.min_value, ' ,',
                      node.prior_node.max_value, ']', '最优分割点:', node.prior_node.threshold)
                node = node.prior_node
            print('-----一个叶子回溯结束--------\n')


    def predict(self, X):

        """ 预测样本X的类别 """

        num_samples = X.shape[0]
        y = np.empty(num_samples)
        for j in range(num_samples):
            node = self.trunk

            while isinstance(node, Node): #判断实例是否是这个类或者object是变量
                if X[j][node.feature_index] <= node.threshold:
                    node = node.branch_true
                else:
                    node = node.branch_false

            if isinstance(node,Leaf):
                y[j] = node.labels #不是node对象，就是一个类别值

        return y #返回样本点在这棵树上的预测标签

    def novelty_predict(self,X,relaxation_factor,leaf_sample_num = 2):

        num_samples = X.shape[0]
        y = np.empty(num_samples)
        for j in range(num_samples):
            node = self.trunk
            while isinstance(node, Node):  # 判断实例是否是这个类或者object是变量
                if X[j][node.feature_index] <= (node.max_value + (relaxation_factor * node.max_value)) and \
                        (X[j][node.feature_index] >= (node.min_value - (relaxation_factor * node.min_value))) and \
                                X[j][node.fa_feature_index] >= (
                            node.fa_feature_min_value - (relaxation_factor * node.fa_feature_min_value)) and \
                        (X[j][node.fa_feature_index] <= (
                            node.fa_feature_max_value + (node.fa_feature_max_value * relaxation_factor))):

                    if X[j][node.feature_index] <= node.threshold:
                        node = node.branch_true
                    else:
                        node = node.branch_false
                else:
                    node = -1
            if isinstance(node, Leaf):  # 叶子节点判断
                if node.sample_num < leaf_sample_num: # 该叶子节点只有一个样本，不做比较，直接返回类别
                    y[j] = node.labels
                    continue #当前循环结束

                # 做两次判断，一个是当前随机特征，一个是父节点的特征
                if (X[j][node.fa_feature_index] <= (node.fa_max_value + (relaxation_factor * node.fa_max_value))) and \
                        (X[j][node.fa_feature_index] >= node.fa_min_value - (relaxation_factor * node.fa_min_value)) and \
                        (X[j][node.current_feature_index] >= (
                            node.current_feature_min_value - (relaxation_factor * node.current_feature_min_value))) and \
                        (X[j][node.current_feature_index] <= (
                            node.current_feature_max_value + (relaxation_factor * node.current_feature_max_value))):

                    y[j] = node.labels
                else:
                    node = -1
            if node == -1:
                y[j] = -1
            ''' for '''
        return y  # 返回样本点在这棵树上的预测标签


