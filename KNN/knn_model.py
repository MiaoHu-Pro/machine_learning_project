
from utilities_knn import package_data,euclidean_distance
from scipy.stats import mode


class KNN(object):

    def __init__(self,train_set):

        self.train_data_num = train_set.shape[0]  # 多少样本
        #样本封装成对象
        self.train_data = package_data(train_set,self.train_data_num)

    def predict(self,x_test,k = 5):

        data = x_test
        data_num = data.shape[0]  # 多少样本
        pre_lab = []
        #计算每一个样本与所有train_data的距离

        for i in range(data_num):
            dis_dict = dict() #创建字典存放与训练数据的距离，key 表示训练样本编号，value表示距离
            for j in range(self.train_data_num):
                d = euclidean_distance(data[i],self.train_data[j].data)
                dis_dict[j] = d #存为字典元素
            #字典按照值（距离）排序
            # dis_dict = {'kye1': 100 ,'kye2': 120 ,'kye3': 140}
            dis_dict_after_sort = sorted(dis_dict.items(), key=lambda v: v[1])
            lab = []
            #记录前k个的类别
            for i in range(k):
                index = dis_dict_after_sort[i][0] #字典的第i个元素的第0个值，也就是训练样本的编号
                lab.append(self.train_data[index].cls)#根据训练样本的编号获取样本的类别
            #取众数
            pre_lab.append(mode(lab)[0][0])#取众数

        return pre_lab













