# -*- coding: utf-8 -*-
# @Time    : 2018/6/25 21:26
# @Author  : HuMiao
# @Email   : humiao001@163.com
# @File    : test.py
# @Software: PyCharm Community Edition

from collections import Counter
import random
import numpy as np
import pandas as pd

#模块调试

resultList=random.sample(range(0,10),10) # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素。上面的方法写了那么多，其实Python一句话就完成了。
print(resultList)# 打印结果

"""
x = [[1,2,3,5,32,452,2,1,12,35],
[1,2,3,5,32,452,2,1,12,35],
[1,2,3,5,32,452,2,1,12,35],
[1,2,3,5,32,452,2,1,12,35],
[1,2,3,5,32,452,2,1,12,35],
[1,2,3,5,32,452,2,1,12,35]]

x_df = pd.DataFrame(x)

random_feature = random.sample(range(0,10),3)
print(random_feature)
x_df_f = x_df[random_feature]
x= np.array(x_df_f)
print(x_df_f)
print(x)
"""
from sklearn.datasets import load_boston
from sklearn.datasets import load_diabetes

# boston = load_boston()
# print(boston.data)
# print(boston.target)
# boston = np.column_stack((boston.data,boston.target))
#
# df = pd.DataFrame(boston)
# print(df)
# df.to_csv('./data_set/boston.csv', index=False, header=False, )

diabetes = load_diabetes()
print(diabetes.data)
print(diabetes.target)
diabetes = np.column_stack((diabetes.data,diabetes.target))

df = pd.DataFrame(diabetes)
print(df)
df.to_csv('./data_set/diabetes.csv', index=False, header=False, )













