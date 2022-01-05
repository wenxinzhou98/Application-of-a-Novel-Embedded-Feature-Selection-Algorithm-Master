import numpy as np
from scipy.stats import t
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy import stats
import scipy.stats as st
import statsmodels as sm
import matplotlib
import pandas as pd
import io
import requests
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import precision_score
import time

start = time.time()

def data_process():
    data_temp=pd.read_csv('faults.csv')
    data = data_temp.iloc[:,:27]

    # 将数据标准化
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=[-1,1])
    np_scaled = min_max_scaler.fit_transform(data)
    data_norm = pd.DataFrame(np_scaled)

    data_norm.columns = list(data.columns)

    low = 0.05
    high = 0.95
    quantiles = data_norm.quantile([low, high])
    quantile_norm = data_norm.apply(lambda col: col[(col >= quantiles.loc[low,col.name]) & 
                                        (col <= quantiles.loc[high,col.name])], axis=0)

    corr_matrix = data_norm.corr().abs()

    under = corr_matrix*(np.triu(np.ones(corr_matrix.shape), k=1))

    to_drop = [column for column in under.columns if any(under[column] > 0.95)]

    data_norm = data_norm.drop(data_norm[to_drop], axis=1)


    output_y = data_temp.iloc[:,27:]
    input_x = data_norm

    le=LabelEncoder()
    targets=output_y.idxmax(1)
    Y=le.fit_transform(targets)

    return input_x, Y

input_x, Y = data_process()

X_train, X_test, y_train, y_test = train_test_split(input_x, Y, test_size=0.3, random_state=None)

model = SVC()

model.fit(X_train, y_train)

model_pred = model.predict(X_test)

print(classification_report(y_test, model_pred))

print(f'accuracy {round(model.score(X_test, y_test)*100, 1)}%')

end = time.time()

print(end-start)