# paper "Class Based Weighted K-Nearest Neighbor over Imbalance Dataset"
# PAKDD 2013 Pacific-Asia Conference on Knowledge Discovery and Data Mining

# Implementation by Jonatan Møller Nuutinen Gøttcke July 2020
from sklearn.neighbors import NearestNeighbors
from collections import Counter 


class CW_kNN: 
    def __init__(self, n_neighbors=7): 
        self.n_neighbors = n_neighbors
        self.nearest_neighbor_list = None 
        self.X = None 
        self.y = None 
        self.nn = None # Here we use the scikit-learn implementation of the nearest neighbor query object. 
        self.classes_ = None 

    def fit(self,X,y): 
        self.X = X 
        self.y = y 
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.nn.fit(self.X, self.y)
        self.classes_ = list(set(y))
        self.m = len(self.classes_)
        self.nearest_neighbor_list = self.nn.kneighbors(self.X)


    def get_params(self): 
        pass

    def kneighbors(self): 
        pass

    def kneighbors_graph(self): 
        pass

    def n_func(self,x,k,c): 
        if type(x) == list: 
            neighbors_sub_list = x[0:k] # notice that k is often smaller than self.n_neighbors
            return [xi for xi in neighbors_sub_list if self.y[xi] == c]

        else: 
            neighbors = self.nearest_neighbor_list[1][x] 
            neighbors_sub_list = neighbors[0:k] 
            return [xi for xi in neighbors_sub_list if self.y[xi] == c]
        
    def _mv_decision_rule(self, xi): 
        neighbor_indices = self.nearest_neighbor_list[1][xi]
        neighbor_labels = [self.y[xt] for xt in neighbor_indices]
        return self._key_with_max_val(dict(Counter(neighbor_labels)))

    def _key_with_max_val(self,d):
        v=list(d.values())
        k=list(d.keys())
        return k[v.index(max(v))]


    def getcoef(self, xi: int) -> float:
        return len(self.n_func(xi, self.n_neighbors, c=self._mv_decision_rule(xi)))/len(self.n_func(xi, k=self.n_neighbors, c=self.y[xi]))

    def alpha(self,c, x_neighbors): 
        return sum([(self.m * self.getcoef(xi)/self.n_neighbors) for xi in x_neighbors[0:int(self.n_neighbors/self.m)] if self.y[xi] == c])

    def weighting_factor(self,c, x_neighbors): 
        return self.alpha(c, x_neighbors)/(1+self.alpha(c, x_neighbors))  
    
    def predict(self, X): 
        predictions = []
        for x in X: 
            _, neighbors_list = self.nn.kneighbors([x])
            neighbors = neighbors_list[0] 
            neighbor_labels = [self.y[i] for i in neighbors] 
            neighbor_label_counter = dict(Counter(neighbor_labels))
            decisions = {label: neighbor_label_counter[label]*self.weighting_factor(label, neighbors) for label in neighbor_label_counter.keys()}
            predictions.append(self._key_with_max_val(decisions))
        return predictions

   