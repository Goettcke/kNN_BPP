from numpy import genfromtxt
from scipy.sparse.construct import random
from random import sample
from sklearn.utils import shuffle
from kNN_BPP import kNN_BPP

# Load an imbalanced dataset 
numpy_array = genfromtxt("dataset/dermatology.csv", delimiter=",")

# Split the dataset into 90% training and 10% testing data
X,y = shuffle(numpy_array[:, 0: len(numpy_array[0]) - 1], numpy_array[:, len(numpy_array[0]) - 1])
clf = kNN_BPP(n_neighbors=10)
train_indices = sample(range(len(y)), int(len(y)*0.9))
test_indices =  list(set(range(len(y))) - set(train_indices))

# Instantiate the classifier and fit it with the training data
clf = kNN_BPP(n_neighbors=10)
clf.fit(X=X[train_indices], y=y[train_indices])

# Print the predictions made by kNN_BPP
print(clf.predict(X[test_indices]))





