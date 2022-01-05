import numpy as np
import pandas as pd
from svm_method import data_process
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

feature = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
 [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
 [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
 [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
 [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
 [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
 [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
 [0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1]])

label = 7
list_for_label = []
for i in range(label):
    for j in range(i+1,label):
        list_for_label.append((i,j)) # 强行把1-7转换成0-6

y = list(data_process()[1])
H_C = 0

for i in range(label):
    H_C += -((y.count(i)/len(y))*np.log2(y.count(i)/len(y)))

entropy = []
for i in range(feature.shape[1]):
    # 遍历所有的特征
    # 计算p(t),p(C|t),p(c|t0)

    # 首先计算p(t)，即T出现的概率
    times_occur = 0
    for j in range(feature.shape[0]): # 遍历所有的行
        if feature[j][i] == 1:
            times_occur += 1
    
    p_t = times_occur / feature.shape[0]

    H_c_T = 0
    H_C_t0 = 0
    p_c_t = 0
    p_c_t0 = 0

    for k in range(label):
        # 计算p(C|t)
        count_c_t = 0
        for a in range(feature.shape[0]): # 遍历所有的行
            if feature[a][i] == 1 and k in list_for_label[a]:
                count_c_t += 1
        p_c_t = count_c_t / feature.shape[0]
        p_c_t0 = 1 - p_c_t
        H_c_T += -p_c_t*np.log2(p_c_t)
        H_C_t0 = -p_c_t0*np.log2(p_c_t0)

    entropy.append(p_t*H_c_T+(1-p_t)*H_C_t0)

info_gain = []
for i in range(len(entropy)):
    info_gain.append(entropy[i]-H_C)

import time

start = time.time()

def accuracy(info_gain, threshold, X, y):
    index = []
    for i in range(len(info_gain)):
        if info_gain[i] > threshold:
            index.append(i)
    list_for_concat = []
    for i in range(len(index)):
        temp = X.iloc[:,index[i]]
        list_for_concat.append(temp)
    result = pd.concat(list_for_concat, axis=1)
    return index, result


sns.set_theme(color_codes=True)

threshold = [0.2,0.5,0.8,1]
len_feat = []
score = []
for i in range(len(threshold)):
    index, X = accuracy(info_gain, threshold[i], data_process()[0], data_process()[1])
    len_feat.append(len(index))
    X_train, X_test, y_train, y_test = train_test_split(X, data_process()[1], test_size=0.3, random_state=None)
    model = SVC(C=5, kernel='rbf', gamma=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    temp = model.score(X_test, y_test)
    score.append(temp)


plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel("选取特征数目")
plt.ylabel("模型预测准确率")

plt.title("信息增益阈值和模型预测准确率之间的关系")
plt.scatter(len_feat[0], score[0], c="red", marker="v")
plt.scatter(len_feat[1], score[1], c='blue', marker='o')
plt.scatter(len_feat[2], score[2], c='black', marker='x')
plt.scatter(len_feat[3], score[3], c='green', marker='D')
plt.legend(["阈值=0.2", "阈值=0.5", "阈值=0.8", "阈值=1"])
plt.show()

'''

print(score[1])

end = time.time()

print(end-start)

'''