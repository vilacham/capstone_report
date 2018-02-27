import numpy as np
from sklearn.metrics import accuracy_score


class MajorityVoteClassifier:
    """ A majority vote classifier
    
    Parameters
    ----------
    classifiers: array-like, shape = [n_classifiers]
    Different classifiers for the ensemble
    """
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def fit(self, X, y):
        """ Fit classifiers
        
        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
        Matrix of training samples
        
        y: array-like, shape = [n_samples]
        Vector of target class labels
        
        Returns
        -------
        self: object
        """
        fitted_classifiers = []
        for classifier in self.classifiers:
            fitted_classifier = (classifier[0],
                                 classifier[1].fit(X, y),
                                 classifier[2])
            fitted_classifiers.append(fitted_classifier)
        self.classifiers = fitted_classifiers
        return self

    def predict(self, X):
        """ Predict class labels for X
        
        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
        Matrix of training samples
        
        Returns
        -------
        majority_votes: array_like, shape = [n_samples]
        Predicted class labels
        """
        predictions = np.asarray([classifier[1].predict(X) for classifier in
                                  self.classifiers]).T

        majority_vote = []
        for prediction in predictions:
            if sum(prediction) >= len(prediction)/2:
                majority_vote.append(1)
            else:
                majority_vote.append(0)
        
        return majority_vote

    def score(self, X, y):
        score = accuracy_score(y, self.predict(X))
        return score
