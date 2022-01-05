import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import seaborn as sns
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier

file_name = "faults.csv"
df = pd.read_csv(file_name, delimiter=',')


drop_features = ['X_Minimum', 'Y_Minimum', 'Pixels_Areas', 'X_Perimeter', 'Sum_of_Luminosity', 
    'Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'TypeOfSteel_A300', 'Outside_X_Index', 
    'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas',
    'Log_X_Index', 'Log_Y_Index'
]

df.drop(drop_features, inplace=True, axis=1)

X = df[['X_Maximum', 'Y_Maximum', 'Y_Perimeter', 'Length_of_Conveyer',
       'TypeOfSteel_A400', 'Steel_Plate_Thickness',
       'Edges_Index', 'Empty_Index', 'Square_Index',
       'Orientation_Index', 'Luminosity_Index','SigmoidOfAreas']]

y_list = []

conditions=[(df['Pastry'] == 1) & (df['Z_Scratch'] == 0)& (df['K_Scatch'] == 0)& (df['Stains'] == 0)& (df['Dirtiness'] == 0)& (df['Bumps'] == 0)& (df['Other_Faults'] == 0), (df['Pastry'] == 0) & (df['Z_Scratch'] == 1)& (df['K_Scatch'] == 0)& (df['Stains'] == 0)& (df['Dirtiness'] == 0)& (df['Bumps'] == 0)& (df['Other_Faults'] == 0),(df['Pastry'] == 0) & (df['Z_Scratch'] == 0)& (df['K_Scatch'] == 1)& (df['Stains'] == 0)& (df['Dirtiness'] == 0)& (df['Bumps'] == 0)& (df['Other_Faults'] == 0),(df['Pastry'] == 0) & (df['Z_Scratch'] == 0)& (df['K_Scatch'] == 0)& (df['Stains'] == 1)& (df['Dirtiness'] == 0)& (df['Bumps'] == 0)& (df['Other_Faults'] == 0),(df['Pastry'] == 0) & (df['Z_Scratch'] == 0)& (df['K_Scatch'] == 0)& (df['Stains'] == 0)& (df['Dirtiness'] == 1)& (df['Bumps'] == 0)& (df['Other_Faults'] == 0),(df['Pastry'] == 0) & (df['Z_Scratch'] == 0)& (df['K_Scatch'] == 0)& (df['Stains'] == 0)& (df['Dirtiness'] == 0)& (df['Bumps'] == 1)& (df['Other_Faults'] == 0),(df['Pastry'] == 0) & (df['Z_Scratch'] == 0)& (df['K_Scatch'] == 0)& (df['Stains'] == 0)& (df['Dirtiness'] == 0)& (df['Bumps'] == 0)& (df['Other_Faults'] == 1)]
choices = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
df['class'] = np.select(conditions, choices)
# 将原先的列去除
df.drop(choices, inplace=True, axis=1)

y_list = []

for i in range(X.shape[0]):
    if df['class'][i] == 'Pastry':
        y_list.append(1)
    elif df['class'][i] == 'Z_Scratch':
        y_list.append(2)
    elif df['class'][i] == 'K_Scatch':
        y_list.append(3)
    elif df['class'][i] == 'Stains':
        y_list.append(4)
    elif df['class'][i] == 'Dirtiness':
        y_list.append(5)
    elif df['class'][i] == 'Bumps':
        y_list.append(6)
    else:
        y_list.append(7)

remove_class = ['class']

df.drop(remove_class, inplace=True, axis=1)
df['class'] = y_list

#y = df[['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']]

y = df['class']

X = np.array(X)

def my_z_score(input_X):
    new_X = []
    for i in range(len(input_X)):
        mean_X = np.mean(input_X)
        std_X = np.std(input_X, ddof=1)
        temp = []
        for j in range(len(input_X[i])):
            calc = (input_X[i][j]-mean_X)/std_X
            temp.append(calc)
        new_X.append(temp)
    return new_X


X = np.array(my_z_score(X))
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state =1)

# normalization
#X_train = X_train.apply(zscore)
#X_test = X_test.apply(zscore)

'''
rf = RandomForestClassifier()
#parameters = { 'n_estimators': [10,50,100,200],'criterion': ['gini', 'entropy'],'min_samples_split':[2,5,10,15] ,'max_depth': [None, 2], 'min_samples_leaf': [1,3,10,15], 'max_features': [None, 'auto','sqrt', 'log2' ]  }
#clf = GridSearchCV(rf, parameters, cv=10)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=7)
print(metrics.auc(fpr, tpr))

#print(metrics.f1_score(y_test, y_pred, average="macro"))
#def min_max_approach(X, y, X_test, y_test, C2=0.1):
'''
random_state = np.random.RandomState(0)
model = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True, random_state=random_state))
clf = model.fit(X_train, y_train)
y_pred = clf.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=7)
print(metrics.auc(fpr, tpr))