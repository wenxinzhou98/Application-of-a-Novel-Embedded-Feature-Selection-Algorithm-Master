'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']#让中文的地方显示出来
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("E:\\Learning\\研究生\\文档\\小论文\\同济大学学报\\faults.csv", delimiter=",", usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26], encoding='gbk')
df_coor=df.corr()

df_coor = df_coor*(np.triu(np.ones(df_coor.shape), k=1))

sns.set(font="simhei")

plt.subplots(figsize=(27,27),facecolor='w')# 设置画布大小，分辨率，和底色

#sns.heatmap(df_coor, cmap='white')#annot为热力图上显示数据；fmt='.2g'为数据保留两位有效数字,square呈现正方形，vmax最大值为1
sns.heatmap(df_coor, cmap="Blues", square=True, vmax=1)
plt.show()
'''



#coding:utf-8
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score
import pyswarms as ps
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from pyswarms.utils.plotters import plot_cost_history

data_temp=pd.read_csv('faults.csv', encoding='gbk')
data = data_temp.iloc[:,:27]

# 将数据标准化
min_max_scaler = preprocessing.MinMaxScaler(feature_range=[-1,1])
np_scaled = min_max_scaler.fit_transform(data)
data_norm = pd.DataFrame(np_scaled)

data_norm.columns = list(data.columns)


corr_matrix = data_norm.corr().abs()

under = corr_matrix*(np.triu(np.ones(corr_matrix.shape), k=0))


plt.subplots(figsize=(27,27),facecolor='w')

sns.set_style('whitegrid',{'font.sans-serif':['simhei','Arial']}) 
sns.heatmap(under, cmap="Blues")

plt.show()

