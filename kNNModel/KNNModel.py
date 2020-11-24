
import numpy as np
import pandas as pd
import sys
from utilities_knnmodel import package_data
from utilities_knnmodel import manhattan_distance,euclidean_distance
from utilities_knnmodel import LNRep ,OSRep

class KNNModel(object):

    def __init__(self,sample_n = 2 ,tolerance_r = 2,representative_set = list(),
                 similarity_matrix = None,obj_set = list(),column = None,n_class = None):

        '''
        :param sample_n: 全局邻域最少的样本数，少于sample_n 时，删除这个代表 默认2
        :param tolerance_r:  错误容忍度，即不同于全局邻域类的样本数 默认1
        :param representative_set: 全局代表集合
        :param similarity_matrix : #相似度矩阵，描述样本间的相似度
        :param obj_set : #对象集合
        :param column : #样本维度
        :param n_class : #样本类数

        '''

        self.sample_n = sample_n
        self.tolerance = tolerance_r
        self.representative_set = representative_set
        self.representative_set_len = None

        self.similarity_matrix = similarity_matrix #相似度矩阵，描述样本间的相似度
        self.obj_set = obj_set
        self.column = column #多少列数
        self.n_class = n_class #多少类
        self.obj_num = None

    def __comput_similarity(self,data):

        self.obj_set = package_data(data) #数据封装

        #计算相似度矩阵
        obj_num = len(self.obj_set) #对象个数
        self.obj_num = obj_num #元素个数

        similarity_matrix = np.zeros((obj_num,obj_num))

        obj_similarity_matrix = pd.DataFrame(similarity_matrix)

        if self.column >5: #维度大于等于5 ， 使用曼哈顿距离
            # print('维度大于等于5，使用manhattan_distance')
            for i in range(obj_num):
                for j in range(obj_num):
                    obj_similarity_matrix[i][j] = manhattan_distance(self.obj_set[i].data,self.obj_set[j].data)

        else:#小于 5 ，使用欧氏距离
            # print('维度小于等于5，使用euclidean_distance')
            for i in range(obj_num):
                for j in range(obj_num):
                    obj_similarity_matrix[i][j] = euclidean_distance(self.obj_set[i].data, self.obj_set[j].data)

        self.similarity_matrix = obj_similarity_matrix
        # print(self.similarity_matrix)

    def __find_neighbourhood(self):
        #每一个元组找到其邻域

        similarity_matrix = self.similarity_matrix
        #每一个样本得到一个集合，这个集合就是局部邻域
        obj_neighbourhood_set = list() #存放所有对象的邻域
        representative_set = list()
        for i in range(self.obj_num):
            obj_set = self.obj_set
            obj_neighbourhood = list() #当前对象的邻域

            if obj_set[i].flag == 0: #falg = 0 ,表示未分组
                #为这个未分组的样本寻找一个邻域
                similarity_list = list(similarity_matrix.loc[i]) #第i个样本与其他样本的相似度
                diff_kind_min_dis = sys.maxsize #异类最近距离

                ''' 该循环寻找最小的异类距离'''
                for j in range(self.obj_num):
                    #最近的异类作为半径，对象的邻域只有这一个异类
                    if obj_set[j].cls != obj_set[i].cls:  # 未标注且类别相同
                        #异类最近距离
                        diff_kind_dis = similarity_list[j]
                        if diff_kind_dis < diff_kind_min_dis:
                            diff_kind_min_dis = diff_kind_dis

                sim = 0  # sim表示邻域的最短距离
                for j in range(self.obj_num):

                    #  上述得到异类的最近距离 diff_kind_dis_min ，根据这个距离，构建
                    if obj_set[i].cls == obj_set[j].cls and  similarity_list[j] < diff_kind_min_dis : #未标注且类别相同

                        if similarity_list[j] > sim :
                            sim = similarity_list[j]
                        obj_neighbourhood.append(obj_set[j])

                local_neighbourhood = LNRep(cls= obj_set[i].cls,num= len(obj_neighbourhood),
                                            sim= sim,rep= obj_set[i],local_neighb_list= obj_neighbourhood)

                obj_neighbourhood_set.append(local_neighbourhood) #i的局部邻域

        #寻找最大邻域
        obj_neighbourhood_set_len = len(obj_neighbourhood_set)

        for i in self.n_class_lable: #对于每个类别
            num_max = 0
            index = None
            for  j in range(obj_neighbourhood_set_len):
                # 变量每个类别的每个节点的局部邻域，寻找样本点最多的邻域，将其作为全局邻域
                if i == obj_neighbourhood_set[j].cls :
                    if obj_neighbourhood_set[j].num > num_max:
                        num_max = obj_neighbourhood_set[j].num
                        index = j
            if index == None: #当递归3、4时，存在某一类不存在时，跳过
                continue
            #找到最大的局部邻域，封装全局邻域对象
            osrep = OSRep(cls= i ,num= obj_neighbourhood_set[index].num,
                          sim= obj_neighbourhood_set[index].sim,
                          rep= obj_neighbourhood_set[index].rep,
                          overall_neighb_list= obj_neighbourhood_set[index].local_neighb_list)
            representative_set.append(osrep)

        #遍历全局邻域，把已经覆盖的样本点置1 ，表示已经分组
        representative_set_len = len(representative_set)
        for i in range(representative_set_len):
            len_representative = representative_set[i].num
            for j in range(len_representative):
                index = representative_set[i].overall_neighb_list[j].number
                self.obj_set[index].flag = 1 #标记，已经被覆盖

        self.representative_set +=  representative_set #记录每次迭代的全局邻域

    #修建全局邻域，个数小于self.sample_n，删除
    def pruning_operator(self):

        temp_list = []
        #获取元素num属性大于self.sample_n 的元素下标
        less_sample_n_list = [i for i in range(self.representative_set_len) if self.representative_set[i].num >= self.sample_n]
        for i in less_sample_n_list: #
            temp_list.append(self.representative_set[i])  # 删除指定下标的元素

        self.representative_set = temp_list
        self.representative_set_len = len(self.representative_set) #重新计算邻域集合的长度

    def fit(self, x_train, y_train):

        self.column = x_train.shape[1] #列数
        self.n_class = len(set(y_train))
        self.n_class_lable = list(set(y_train))

        train_data = np.column_stack((x_train,y_train)) #数据与标签合并
        #1 计算显示度矩阵，所有元素标注为分组(flag = 0)
        self.__comput_similarity(train_data) #数据封装

        #遍历每一个对象的flag，查找是否有为覆盖的，若有，继续寻找邻域
        flag_list = [ i for i in range(len(self.obj_set)) if self.obj_set[i].flag == 0]
        while len(flag_list) !=0:
            self.__find_neighbourhood()
            flag_list = [i for i in range(len(self.obj_set)) if self.obj_set[i].flag == 0]

        #全局邻域的个数
        self.representative_set_len = len(self.representative_set)

        # print(self.representative_set_len)
        #遍历代表集合，若某一个邻域样本数小于 1 ，则删除
        self.pruning_operator()#剪枝操作
        # print(self.representative_set_len)

    def predict(self,test_data):

        test_len = test_data.shape[0]

        pre_lab = list()

        if self.column > 5 :
            for i in range(test_len):
                pre_lab_flag = list()
                dis_list = list()

                for j in range(self.representative_set_len):
                    dis = manhattan_distance(self.representative_set[j].rep.data,test_data[i])
                    if dis <= self.representative_set[j].sim:
                        pre_lab_flag.append(1)
                        dis_list.append(dis)
                    else:
                        pre_lab_flag.append(0)
                        dis_list.append(dis)
                # print(pre_lab_flag)
                #分析决策标签pre_lab_flag
                if sum(pre_lab_flag) == 1: #样本点只落入一个邻域内
                    index = pre_lab_flag.index(1)
                    cls = self.representative_set[index].cls
                    pre_lab.append(cls)
                elif sum(pre_lab_flag) > 1:#样本点落入多个邻域内，样本属于样本数多的邻域
                    index_list = [i for i, x in enumerate(pre_lab_flag) if x == 1]
                    max_num = 0
                    index = None
                    for i in index_list:
                        if self.representative_set[i].num > max_num:
                            max_num = self.representative_set[i].num
                            index = i
                    cls = self.representative_set[index].cls
                    pre_lab.append(cls)
                elif sum(pre_lab_flag) == 0:#

                    dis_2 = list()
                    dis_list_len = len(dis_list)

                    for i in range(dis_list_len):
                        dis_2.append(dis_list[i] - self.representative_set[i].sim)

                    index = dis_2.index(min(dis_2))
                    cls = self.representative_set[index].cls
                    pre_lab.append(cls)

        else:
            for i in range(test_len):
                pre_lab_flag = list()
                dis_list = list()
                for j in range(self.representative_set_len):
                    dis = euclidean_distance(self.representative_set[j].rep.data,test_data[i])
                    if dis <= self.representative_set[j].sim:
                        pre_lab_flag.append(1)
                        dis_list.append(dis)
                    else:
                        pre_lab_flag.append(0)
                        dis_list.append(dis)

                # print(pre_lab_flag)
                if sum(pre_lab_flag) == 1:
                    index = pre_lab_flag.index(1)
                    cls = self.representative_set[index].cls
                    pre_lab.append(cls)
                elif sum(pre_lab_flag) > 1:

                    index_list = [i for i ,x in enumerate(pre_lab_flag) if x == 1]
                    max_num = 0
                    index = None
                    for i in index_list:
                        if self.representative_set[i].num > max_num:
                            max_num = self.representative_set[i].num
                            index = i
                    cls = self.representative_set[index].cls
                    pre_lab.append(cls)
                elif sum(pre_lab_flag) == 0:

                    dis_2 = list()
                    dis_list_len = len(dis_list)

                    for i in range(dis_list_len):
                        dis_2.append(dis_list[i] - self.representative_set[i].sim)

                    index = dis_2.index(min(dis_2))
                    cls = self.representative_set[index].cls
                    pre_lab.append(cls)

        return pre_lab

    def novelty_detection(self,test_data):
        test_len = test_data.shape[0]
        pre_lab = list()
        if self.column > 5:
            for i in range(test_len):
                pre_lab_flag = list()
                dis_list = list()

                for j in range(self.representative_set_len):
                    dis = manhattan_distance(self.representative_set[j].rep.data, test_data[i])
                    if dis <= self.representative_set[j].sim:
                        pre_lab_flag.append(1)
                        dis_list.append(dis)
                    else:
                        pre_lab_flag.append(0)
                        dis_list.append(dis)
                # 分析决策标签pre_lab_flag
                # print(pre_lab_flag)
                if sum(pre_lab_flag) == 0:  #没有落入任何邻域内

                    pre_lab.append(-1)
                else:
                    pre_lab.append(1)
        else:
            for i in range(test_len):
                pre_lab_flag = list()
                dis_list = list()
                for j in range(self.representative_set_len):
                    dis = euclidean_distance(self.representative_set[j].rep.data, test_data[i])
                    if dis <= self.representative_set[j].sim:
                        pre_lab_flag.append(1)
                        dis_list.append(dis)
                    else:
                        pre_lab_flag.append(0)
                        dis_list.append(dis)
                # print(pre_lab_flag)
                # 分析决策标签pre_lab_flag
                if sum(pre_lab_flag) == 0:  # 没有落入任何邻域内

                    pre_lab.append(-1)
                else:
                    pre_lab.append(1)

        return pre_lab







