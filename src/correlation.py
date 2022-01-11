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

