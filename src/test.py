import sys
sys.path.append('../')
#code below used to deal with special characters on the file path during read_csv()
sys._enablelegacywindowsfsencoding()

import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt #MatPlotLib usado para desenhar o gráfico criado com o NetworkX

# Import PySwarms
import pyswarms as ps

'''
### Generating a toy dataset using scikit-learn
We'll be using `sklearn.datasets.make_classification` to generate a 100-sample, 15-dimensional dataset with three classes. 
We will then plot the distribution of the features in order to give us a qualitative assessment of the feature-space.

For our toy dataset, we will be rigging some parameters a bit. Out of the 10 features, 
we'll have only 5 that are informative, 5 that are redundant, and 2 that are repeated. 
Hopefully, we get to have Binary PSO select those that are informative, and prune those that are redundant or repeated.
'''

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=500, n_features=20, n_classes=2, 
                           n_informative=5, n_redundant=0, n_repeated=0, 
                           random_state=None, shuffle=True)

#X, X_test, y, y_test = train_test_split(X, y, test_size=0.20, random_state=None)
#X, y = make_classification(n_samples=100, n_features=15, n_classes=3,
#                           n_informative=4, n_redundant=1, n_repeated=2,
#                           random_state=1)

df = pd.DataFrame(X)
df['labels'] = pd.Series(y)

"""
As we can see, there are some features that causes the two classes to overlap with one another. 
These might be features that are better off unselected. 
On the other hand, we can see some feature combinations where the two classes are shown to be clearly separated. 
These features can hopefully be retained and selected by the binary PSO algorithm.
We will then use a simple logistic regression technique using `sklearn.linear_model.LogisticRegression` to perform classification. 
A simple test of accuracy will be used to assess the performance of the classifier.

## Writing the custom-objective function
As seen above, we can write our objective function by simply taking the performance of the classifier (in this case, the accuracy), 
and the size of the feature subset divided by the total (that is, divided by 10), to return an error in the data. 
We'll now write our custom-objective function
"""


from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

# Create an instance of the classifier
classifier = linear_model.LogisticRegression()
#classifier = RandomForestClassifier(n_estimators = 64,
#                                    #max_features = 30,
#                                    bootstrap = True,
#                                    random_state = None)
    
#clf = forest
#clf.fit(X_trainOhFeatures, y_train)
#predictions = clf.predict(X_testOhFeatures)
#accuracy = accuracy_score(y_test, predictions)

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

def f(x, alpha=0.9):
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

"""
## Using Binary PSO
With everything set-up, we can now use Binary PSO to perform feature selection. 
For now, we'll be doing a global-best solution by setting the number of neighbors equal to the number of particles. 
The hyperparameters are also set arbitrarily. 
Moreso, we'll also be setting the distance metric as 2 (truth is, it's not really relevant because each particle will see one another).
"""

from datetime import datetime as dt
import time
from pyswarms.utils.plotters import plot_cost_history

start = dt.now()
print("Started at: ", str(start))
particleScore = list()
particleSize = list()
#mySubsets = list()

# Initialize swarm, arbitrary
options = {'c1': 2, 'c2': 2, 'w':0.3, 'k': 20, 'p':2}

# Call instance of PSO
dimensions = X.shape[1] # dimensions should be the number of features
#optimizer.reset()
#optimizer = ps.single.GlobalBestPSO(n_particles=1, dimensions=dimensions,
#                                    options=options)
optimizer = ps.discrete.BinaryPSO(n_particles=20, dimensions=dimensions, options=options)

# Perform optimization
best_cost, best_pos = optimizer.optimize(f, iters=100, verbose=2)

#plot_cost_history(optimizer.cost_history)
#plt.show()

#print(cost,pos)
end = dt.now()
print("Finished at: ", str(end))
total = end-start
print("Total time spent: ", total)

print(optimizer.mean_pbest_history)
#print(optimizer.personal_best_pos[5])
print(best_pos)

rank = list()

fullSet = cross_val_score(classifier, X, y, cv=5)
print("Full set Accuracy: %0.2f (+/- %0.2f)" % (fullSet.mean(), fullSet.std() * 2))
print("----------------------------------------------------------------------------")
bests = best_pos
for b in bests:
    # Get the selected features from the final positions
    X_selected_features = X[:,b==1]  # subset

    # Perform classification and store performance in P
    #classifier.fit(X_selected_features, y)
    scores = cross_val_score(classifier, X_selected_features, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), b)
    rank.append([scores.mean(), b])
    # Compute performance
    #subset_performance = (c1.predict(X_selected_features) == y).mean()
    #subset_performance = (classifier.predict(X_selected_features) == y).mean()
    

    #print('Subset performance: %.3f' % (subset_performance))

"""
fullSetAccuracy = round(fullSet.mean(),4)
bestSubsetAccuracy = round(max(rank)[0],4)
bestSubsetLength = sum(max(rank)[1])

print("Full set accuracy:", fullSetAccuracy)
print("Best subset accuracy:", bestSubsetAccuracy)
print("Best subset length:", bestSubsetLength)
print("-----------------------------------------------")
print("-----------------------------------------------")
print("             length  |  accuracy")
print("full set     ", X.shape[1], "      ", fullSetAccuracy)
print("best subset  ", bestSubsetLength, "      ", bestSubsetAccuracy)
"""