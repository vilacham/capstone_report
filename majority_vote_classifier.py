from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score

import numpy as np


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
	""" A majority vote ensemble classifier

	Parameters
	----------
	classifiers: array-like, shape = [n_classifiers]
	Different classifiers for the ensemble

	vote: str, {'classlabel', 'probability'}
	Default: 'classlabel'
	If 'classlabel' the prediction is based on the argmax of class label, else if 'probability' the argmax of the sum of
	probabilities is used to predict the class label (recommended for calibrated classifiers)
	"""
	def __init__(self, classifiers, vote='classlabel'):
		self.classifiers = classifiers
		self.vote = vote

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
		self.classifiers_ = []
		for classifier in self.classifiers:
			fitted_classifier = clone(classifier).fit(X, y)
			self.classifiers_.append(fitted_classifier)
		return self

	def predict(self, X):
		""" Predict class labels for X

		Parameters
		----------
		X: {array-like, sparse matrix}, shape = [n_samples, n_features]
		Matrix of training samples

		Returns
		-------
		majority_vote: array-like, shape = [n_samples]
		Predicted class labels
		"""
		if self.vote == 'classlabel':
			# Collect results from clf.predict calls
			predictions = np.asarray([classifier.predict(X) for classifier in self.classifiers_]).T
			majority_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions)
		else:  # if self.vote == 'probability'
			majority_vote = np.argmax(self.predict_proba(X), axis=1)
		return majority_vote

	def predict_proba(self, X):
		""" Predict class probabilities for X

		Parameters
		----------
		X: {array-like, sparse matrix}, shape = [n_samples, n_features]
		Matrix of training vectors

		Returns
		-------
		average_probabilities: array-like, shape[n_samples, n_classes]
		Average probability for each class per sample
		"""
		probabilities = np.asarray([classifier.predict_proba(X) for classifier in self.classifiers_])
		average_probabilities = np.average(probabilities, axis=0)
		return average_probabilities

	def score(self, X, y):
		score = accuracy_score(y, self.predict(X))
		return score
