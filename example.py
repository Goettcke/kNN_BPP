from numpy import genfromtxt
from scipy.sparse.construct import random
from random import sample
from sklearn.utils import shuffle
from knn_bpp import kNN_BPP
from cw_knn import CW_kNN
from direct_cs_knn import DIRECT_CS_kNN

# Load an imbalanced dataset 
numpy_array = genfromtxt("dataset/dermatology.csv", delimiter=",")

# Split the dataset into 90% training and 10% testing data
X,y = shuffle(numpy_array[:, 0: len(numpy_array[0]) - 1], numpy_array[:, len(numpy_array[0]) - 1])
train_indices = sample(range(len(y)), int(len(y)*0.9))
test_indices =  list(set(range(len(y))) - set(train_indices))

# Instantiate the classifiers and fit them with the training data
knn_bpp_clf = kNN_BPP(n_neighbors=10)
knn_bpp_clf.fit(X=X[train_indices], y=y[train_indices])

cw_knn_clf = CW_kNN(n_neighbors=10)
cw_knn_clf.fit(X=X[train_indices], y=y[train_indices])

dir_cs_knn_clf = DIRECT_CS_kNN(n_neighbors=10)
dir_cs_knn_clf.fit(X=X[train_indices], y=y[train_indices])


# Print the predictions made by kNN_BPP, CW_kNN and DIRECT_CS_kNN in the following manner, or use them for further evaluation.
print(knn_bpp_clf.predict(X[test_indices]))
print(cw_knn_clf.predict(X[test_indices]))
print(dir_cs_knn_clf.predict(X[test_indices]))






