import numpy as np

#数据对象
class OData(object):
    def __init__(self,number,data,cls):

        '''
        :param number: 样本标号
        :param data: 样本数据域
        :param cls: 样本标签
        '''
        self.number  = number #样本标号
        self.data = data #样本数据域
        self.cls = cls   #样本标签

#计算欧式距离 方法1 省时，高效
def euclidean_distance(vector1, vector2):
    d = 0
    for a, b in zip(vector1, vector2):
        d += (a - b) ** 2
    return np.sqrt(d)
#计算曼哈顿距离
def manhattan_distance(vector1,vector2):

    return sum(abs(np.array(vector1) - np.array( vector2)))

#样本封装成对象
def package_data(data, data_num):

    data_set = data
    length = data_num
    object_set = list()
    for i in range(length):
        number = i #这是节点编号
        data = data_set[i,:-1] #样本数据域
        cls = data_set[i,-1]  #样本标签
        object_set.append(OData (number= number,data = data ,cls=cls))

    return object_set