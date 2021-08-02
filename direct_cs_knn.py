"""
An implementation of the algorith Direct-CS-kNN from the
2013 paper (Cost-Sensitive Classification with k-Nearest Neighbors)
The same method is proposed in the 2020 paper (Cost-sensitive {KNN} classification)
"""
#import math
import numpy as np
from sklearn.neighbors import BallTree
from collections import Counter

class DIRECT_CS_kNN: 
    def __init__(self, n_neighbors=7, cost_matrix_type="regular", verbose=False):
        self.X = None
        self.y = None
        self.n_neighbors = n_neighbors
        self.ball_tree = None
        self.classes_ = None
        self.ni = None
        self.n = None
        self.cost_matrix = None
        self.cost_matrix_type = cost_matrix_type # Can be regular or imbalance 
        self.verbose = verbose

    def fit(self,X,y):
        self.X = X 
        self.y = y 
        
        self.n = len(X)

        self.dimensionality = X.shape[1]
        self.ni = Counter(self.y)

        self.classes_ = sorted(list(set(y)))
        self.m = len(self.classes_)
        
        self.class_indices = {label: index for index, label in enumerate(self.classes_)}

        self.ball_tree = BallTree(self.X,leaf_size = self.n_neighbors + 1) 
        self.build_cost_matrix()

    def build_cost_matrix(self):
        self.cost_matrix = np.zeros((self.m,self.m))
        if self.cost_matrix_type == "regular":  
            for i in range(self.m): 
                self.cost_matrix[i,i] = 1 
                
        elif self.cost_matrix_type == "class_imbalance": 
            for i in range(self.m): 
                for j in range(self.m):
                    self.cost_matrix[i,j] = self.ni[self.classes_[i]]/self.ni[self.classes_[j]]
                    if self.verbose:  
                        print(f"({i},{j}): {self.ni[self.classes_[j]]},{self.ni[self.classes_[i]]}", end=" ")
                        print(f"weight: {self.cost_matrix[i,j]}")
        elif self.cost_matrix_type == "basic_prior": 
            for i in range(self.m): 
                for j in range(self.m): 
                    if i == j:
                        self.cost_matrix[i,i] = self.ni[self.classes_[i]]/self.ni[self.key_with_min_val(self.ni)]
                    else: 
                        self.cost_matrix[i,j] = 0
        else: 
            pass
        if self.verbose: 
            print(self.cost_matrix)

    def key_with_min_val(self,d):
     v=list(d.values())
     k=list(d.keys())
     return k[v.index(min(v))]

    def predict(self,x, verbose=False):
        assert self.ball_tree != None
        distances, neighbors = self.ball_tree.query(x,self.n_neighbors)
        predictions = []
        for point_neighbors in neighbors:  
            neighbor_labels = list(self.y[point_neighbors])
            k = len(neighbor_labels)
            occurrences = {key: neighbor_labels.count(key) for key in self.classes_}
            
            cost_times_probabilities = {}

            for i in occurrences.keys(): 
                if occurrences[i] != 0: 
                    l = []
                    for j in occurrences.keys():
                        probabilty_of_i = occurrences[i]/k
                        probability_of_not_i = 1-(probabilty_of_i/sum([1-(x/k) for x in occurrences.values()]))
                        l.append(probability_of_not_i*self.cost_matrix[self.class_indices[i],self.class_indices[j]])
                    cost_times_probabilities[i] = sum(l)


            prediction = self.key_with_min_val(cost_times_probabilities) 
            predictions.append(prediction)
            if self.verbose == True: 
                print(neighbor_labels)
                print(occurrences)
                print(f"ni: {self.ni}")
                print(cost_times_probabilities)
            
        return predictions
    
    
    def predict_proba(self): 
        pass 

    def score(self): 
        pass

    def set_params(self): 
        pass

    def get_params(self): 
        pass 

    def kneighbors(self): 
        pass

    def kneighbors_graph(self): 
        pass

