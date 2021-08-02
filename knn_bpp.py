# Implementation of the kNN BPP  classifier
from sklearn.neighbors import BallTree
from collections import Counter, defaultdict
from sklearn.neighbors._base import NeighborsBase, KNeighborsMixin
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score


class kNN_BPP(NeighborsBase, KNeighborsMixin, ClassifierMixin): 
    def __init__(self, n_neighbors=7,include_ties = False):
        self.X = None
        self.y = None
        self.n_neighbors = n_neighbors
        self.ball_tree = None
        self.classes_ = None
        self.ni = None
        self.include_ties = include_ties

    def fit(self,X,y):
        self.X = X 
        self.y = y 
        self.classes_ = list(set(y))
        self.ball_tree = BallTree(self.X,leaf_size = self.n_neighbors + 10)
        self.dimensionality = X.shape[1]
        self.ni = Counter(self.y)


    def key_with_max_val(self,d):
        v=list(d.values())
        k=list(d.keys())
        return k[v.index(max(v))]


    def predict(self,x):
        assert self.ball_tree != None
        distances, neighbors = self.ball_tree.query(x,self.n_neighbors)
        predictions = []
        for point_neighbors in neighbors:  
            neighbor_labels = list(self.y[point_neighbors])
            occurrences = {key : neighbor_labels.count(key) for key in self.classes_} 
            denominator = sum([occurrences[key]/self.ni[key] for key in occurrences.keys()])
            probabilities = {key : (occurrences[key]/self.ni[key])/denominator for key in self.classes_}
            prediction = self.key_with_max_val(probabilities) 
            predictions.append(prediction)
        return predictions
    
    
    def predict_proba(self, X): 
        assert self.ball_tree != None
        distances, neighbors = self.ball_tree.query(X, self.n_neighbors)
        predicted_probabilities = []
        for point_neighbors in neighbors:  
            neighbor_labels = list(self.y[point_neighbors])
            occurrences = {key : neighbor_labels.count(key) for key in self.classes_} 
            denominator = sum([occurrences[key]/self.ni[key] for key in occurrences.keys()])
            probabilities = [(occurrences[key]/self.ni[key])/denominator for key in self.classes_]
            predicted_probabilities.append(probabilities)
        return predicted_probabilities


    def score(self, X, y):
        """
        Returns the mean accuracy of the predictions on X
        """
        y_pred = self.predict(X) 
        return accuracy_score(y_pred=y_pred, y_true=y)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    # set_params is the default method taken directly from scikit-learn
    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self
