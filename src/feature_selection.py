import sys
sys.path.append('../')
#code below used to deal with special characters on the file path during read_csv()
sys._enablelegacywindowsfsencoding()

import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pyswarms as ps
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# Define objective function
def f_per_particle(m, alpha):
    """Computes for the objective function per particle

    Inputs
    ------
    m : numpy.ndarray
        Binary mask that can be obtained from BinaryPSO, will
        be used to mask features.
    alpha: float (default is 0.5)
        Constant weight for trading-off classifier performance
        and number of features

    Returns
    -------
    numpy.ndarray
        Computed objective function
    """
    total_features = X.shape[1]
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0: 
        #if the particle subset is only zeros, get the original set of attributes
        X_subset = X
    else:
        X_subset = X[:,m==1]
        
    #X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.20, random_state=None)
    # Perform classification and store performance in P
    #classifier.fit(X_train, y_train)
    #P = (classifier.predict(X_test) == y_test).mean()
    
    scores = cross_val_score(classifier, X_subset, y, cv=3)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    P = scores.mean()
    particleScore.append(P)
    particleSize.append(X_subset.shape[1])
    # Compute for the objective function
    j = (alpha * (1.0 - P)
       + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

    
    #j = (alpha * (1.0 - P)) + (1 - alpha) * (1 - (total_features - X_subset.shape[1]) / total_features)
    #print("Particle j: ", j)
    return j

def f(x, alpha=0.7):
    """Higher-level method to do classification in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    #print("f j: ", j)
    return np.array(j)


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
input_X = data_norm

le=LabelEncoder()
targets=output_y.idxmax(1)
input_y=le.fit_transform(targets)


# 训练1/2*m*(m-1)个SVM
matrix_for_best_pos = []
percentage = []
labels = 7 # 0 1 2 3 4 5 6
for i in range(labels):
    for j in range(i+1, labels):
        X1 = input_X[input_y==i]
        X2 = input_X[input_y==j]
        X = X1.append(X2)
        y = []
        X = X.values
        for _ in range(X1.shape[0]):
            y.append(1)
        for _ in range(X2.shape[0]):
            y.append(0)

        df = pd.DataFrame(X)
        df["class"] = pd.Series(y)

        classifier = SVC() # 到时候再grid search

        particleScore = list()
        particleSize = list()

        # Initialize swarm, arbitrary
        options = {'c1': 2, 'c2': 2, 'w':0.3, 'k': 20, 'p':2}
        # Call instance of PSO
        dimensions = X.shape[1] # dimensions should be the number of features
        #optimizer.reset()
        #optimizer = ps.single.GlobalBestPSO(n_particles=1, dimensions=dimensions,
        #                                    options=options)
        optimizer = ps.discrete.BinaryPSO(n_particles=22, dimensions=dimensions, options=options)

        # Perform optimization
        best_cost, best_pos = optimizer.optimize(f, iters=100, verbose=2)

        #print(optimizer.mean_pbest_history)
        #print(best_pos)
        matrix_for_best_pos.append(best_pos)
        percentage.append(X.shape[0]/input_X.shape[0])

matrix_for_best_pos = np.array(matrix_for_best_pos)

print(matrix_for_best_pos)