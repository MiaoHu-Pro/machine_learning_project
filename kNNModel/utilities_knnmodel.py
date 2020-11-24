
import numpy as np
import pandas as pd
import time

#Local neighborhood Representative 局部代表
class LNRep(object):
    def __init__(self, cls=None,num=None,sim = None, rep=None,local_neighb_list = list()):
        self.cls = cls
        self.sim = sim
        self.num = num
        self.rep = rep
        self.local_neighb_list = local_neighb_list
# Overall Situation Representative 全局代表
class OSRep(object):

    def __init__(self,cls = None,sim = None,num = None,rep = None,overall_neighb_list = list()):
        '''
        <Cls(di),Sim(di),Num(di),Rep(di)>
        :param cls:   中心点di的标签
        :param sim:   被全局阈覆盖的数据元组到中心点di的最低相似度
        :param num:   被全局阈覆盖的数据元组个数
        :param rep:   中心点di本身
        '''
        self.cls = cls
        self.sim = sim
        self.num = num
        self.rep = rep
        self.overall_neighb_list = overall_neighb_list
#数据对象
class OData(object):
    def __init__(self,number,data,cls,flag):

        '''
        :param number: 样本标号
        :param data: 样本数据域
        :param cls: 样本标签
        :param flag: 样本是否分组 ，1 是，0 否
        '''

        self.number  = number #样本标号
        self.data = data #样本数据域
        self.cls = cls   #样本标签
        self.flag = flag #样本是否分组 ，1 是，0 否



#余弦距离
def cos_distance(vector1,vector2):

    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a,b in zip(vector1,vector2):
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB==0.0:
        return None
    else:
        return dot_product /((normA*normB)**0.5)

#计算欧式距离 方法1 省时，高效
def euclidean_distance(vector1, vector2):
    d = 0
    for a, b in zip(vector1, vector2):
        d += (a - b) ** 2
    # return d ** 0.5
    return np.sqrt(d)
#计算欧式距离 方法2
def euclidean_distance_2(vector1, vector2):

    dis = np.linalg.norm(np.array(vector1) - np.array(vector2)) #

    return dis

#计算曼哈顿距离
def manhattan_distance(vector1,vector2):

    return sum(abs(np.array(vector1) - np.array( vector2)))

#样本封装成对象
def package_data(data):

    data_set = data
    length = data_set.shape[0]  # 多少样本
    object_set = list()

    for i in range(length):

        number = i #这是节点编号
        data = data_set[i,:-1] #样本数据域
        cls =  data_set[i,-1]  #样本标签
        flag = 0
        object_set.append(OData (number= number,data = data ,cls=cls , flag = flag))

    return object_set


if __name__ == '__main__':

    v1 = [1,2,3,5,6,7,8,9,7,4,2,5]
    v2 = [3,4,5,6,7,8,9,0,6,5,4,3]

    t3 = time.time()
    dis = euclidean_distance_2(v1,v2)
    t4 = time.time()
    # print(t4 - t3 ,'\ndis = ',dis)

    t1 = time.time()
    dis_2 = euclidean_distance(v1,v2)
    t2 = time.time()
    # print(t2 - t1 ,'\ndis = ',dis_2)
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    dis_3 = manhattan_distance(v1,v2)
    # print(dis_3)

    # a = [i for i in range(10) if i == 10]
    # print(len(a))

    a = [72, 56, 76,76, 84, 80, 88]
    print(a.index(76))















