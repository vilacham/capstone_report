# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Import dataset as DataFrame and create a copy of it
original_data = pd.read_excel('capstone_database.xlsx')
data = original_data

# Rename data columns
data.columns = ['DATE', 'A TEAM', 'H TEAM', 'osite', 'A PTS', 'H PTS', 'A STK', 'H STK', 'A PTS LG', 'A FGM LG', 
				'A FGA LG', 'A 3PM LG', 'A 3PA LG', 'A FTM LG', 'A FTA LG', 'A OREB LG', 'A DREB LG', 'A REB LG', 
				'A AST LG', 'A TOV LG', 'A STL LG', 'A BLK LG', 'H PTS LG', 'H FGM LG', 'H FGA LG', 'H 3PM LG', 
				'H 3PA LG', 'H FTM LG', 'H FTA LG', 'H OREB LG', 'H DREB LG', 'H REB LG', 'H AST LG', 'H TOV LG', 
				'H STL LG', 'H BLK LG', 'A PTS L2G', 'A FGM L2G', 'A FGA L2G', 'A 3PM L2G', 'A 3PA L2G', 'A FTM L2G', 
				'A FTA L2G', 'A OREB L2G', 'A DREB L2G', 'A REB L2G', 'A AST L2G', 'A TOV L2G', 'A STL L2G', 
				'A BLK L2G', 'H PTS L2G', 'H FGM L2G', 'H FGA L2G', 'H 3PM L2G', 'H 3PA L2G', 'H FTM L2G', 'H FTA L2G', 
				'H OREB L2G', 'H DREB L2G', 'H REB L2G', 'H AST L2G', 'H TOV L2G', 'H STL L2G', 'H BLK L2G', 
				'A PTS L5G', 'A FGM L5G', 'A FGA L5G', 'A 3PM L5G', 'A 3PA L5G', 'A FTM L5G', 'A FTA L5G', 'A OREB L5G',
				'A DREB L5G', 'A REB L5G', 'A AST L5G', 'A TOV L5G', 'A STL L5G', 'A BLK L5G', 'H PTS L5G', 'H FGM L5G',
				'H FGA L5G', 'H 3PM L5G', 'H 3PA L5G', 'H FTM L5G', 'H FTA L5G', 'H OREB L5G', 'H DREB L5G', 
				'H REB L5G', 'H AST L5G', 'H TOV L5G', 'H STL L5G', 'H BLK L5G', 'OVT', 'DAY', 'MTH', 'PLAYOFFS']

# Replace '-' and 'away' by NaN values and then replace it
data.replace(to_replace='-', value=np.nan, inplace=True, regex=True)
data.replace(to_replace='away', value=np.nan, inplace=True, regex=True)
data.dropna(inplace=True)
data.reset_index(inplace=True)

# Deal with 'DAY' and 'MTH' columns
data = pd.get_dummies(data, columns=['DAY', 'MTH'])

# Create label column (1 for home team win, 0 for visitor win)
data['WINNER'] = (data['H PTS'] > data['A PTS']).astype(int)

# Drop unecesary columns
columns_to_drop = ['index', 'DATE', 'A TEAM', 'H TEAM', 'osite', 'A PTS', 'H PTS']
data.drop(columns_to_drop, axis=1, inplace=True)

# Create dictionary containing index and features (only to make dividing the dataset easier)
features_dictionary = {}
for i in range(len(data.columns)):
	features_dictionary[i] = data.columns[i]
#print(features_dictionary)

# Drop outliers
counter = Counter()
for feature in data.iloc[:, 2:87].columns:
	data[feature] = data[feature].apply(pd.to_numeric)
	quartile_1 = np.percentile(data[feature], 25)
	quartile_3 = np.percentile(data[feature], 75)
	step = (quartile_3 - quartile_1) * 1.5
	outliers = data[~((data[feature] >= quartile_1 - step) & 
								   (data[feature] <= quartile_3 + step))]
	#print('{} outliers for the feature {}:\n{}'.format(len(outliers), feature, outliers.index.values))
	counter.update(outliers.index.values)
#print(counter.items())
frequent_outliers = [outlier[0] for outlier in counter.items() if outlier[1] > 1]
#print('{} outliers for more than one feature:\n{}'.format(len(frequent_outliers), frequent_outliers))
print("Original data had {} samples.".format(data.shape[0]))
data = data.drop(data.index[frequent_outliers]).reset_index(drop = True)
print("New data has {} samples.".format(data.shape[0]))

# Split the dataset into training and testing sets
X, y = data.iloc[:, :-1], data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize continuous variables
standard_scaler = StandardScaler()
columns_to_standardize = ['A PTS LG', 'A FGM LG', 'A FGA LG', 'A 3PM LG', 'A 3PA LG', 'A FTM LG', 'A FTA LG', 
						  'A OREB LG', 'A DREB LG', 'A REB LG', 'A AST LG', 'A TOV LG', 'A STL LG', 'A BLK LG', 
						  'H PTS LG', 'H FGM LG', 'H FGA LG', 'H 3PM LG', 'H 3PA LG', 'H FTM LG', 'H FTA LG', 
						  'H OREB LG', 'H DREB LG', 'H REB LG', 'H AST LG', 'H TOV LG', 'H STL LG', 'H BLK LG', 
						  'A PTS L2G', 'A FGM L2G', 'A FGA L2G', 'A 3PM L2G', 'A 3PA L2G', 'A FTM L2G', 'A FTA L2G', 
						  'A OREB L2G', 'A DREB L2G', 'A REB L2G', 'A AST L2G', 'A TOV L2G', 'A STL L2G', 'A BLK L2G', 
						  'H PTS L2G', 'H FGM L2G', 'H FGA L2G', 'H 3PM L2G', 'H 3PA L2G', 'H FTM L2G', 'H FTA L2G', 
						  'H OREB L2G', 'H DREB L2G', 'H REB L2G', 'H AST L2G', 'H TOV L2G', 'H STL L2G', 'H BLK L2G', 
						  'A PTS L5G', 'A FGM L5G', 'A FGA L5G', 'A 3PM L5G', 'A 3PA L5G', 'A FTM L5G', 'A FTA L5G', 
						  'A OREB L5G', 'A DREB L5G', 'A REB L5G', 'A AST L5G', 'A TOV L5G', 'A STL L5G', 'A BLK L5G', 
						  'H PTS L5G', 'H FGM L5G', 'H FGA L5G', 'H 3PM L5G', 'H 3PA L5G', 'H FTM L5G', 'H FTA L5G', 
						  'H OREB L5G', 'H DREB L5G', 'H REB L5G', 'H AST L5G', 'H TOV L5G', 'H STL L5G', 'H BLK L5G', 
						  'OVT']
pd.options.mode.chained_assignment = None # default='warn'
X_test[columns_to_standardize] = X_test[columns_to_standardize].apply(pd.to_numeric)
X_train.loc[:, columns_to_standardize] = standard_scaler.fit_transform(X_train[columns_to_standardize])
X_test.loc[:, columns_to_standardize] = standard_scaler.transform(X_test[columns_to_standardize])

# Divide training set in three
X_train_last_game = X_train[list(X_train.columns[:30]) + list(X_train.columns[86:])]
X_test_last_game = X_test[list(X_test.columns[:30]) + list(X_test.columns[86:])]
X_train_last_two_games = X_train[list(X_train.columns[:2]) + list(X_train.columns[30:58]) + list(X_train.columns[86:])]
X_test_last_two_games = X_test[list(X_test.columns[:2]) + list(X_test.columns[30:58]) + list(X_test.columns[86:])]
X_train_last_five_games = X_train[list(X_train.columns[:2]) + list(X_train.columns[58:])]
X_test_last_five_games = X_test[list(X_test.columns[:2]) + list(X_test.columns[58:])]

# Reduce dimensionality of the last game data
pca = PCA(n_components=None)
X_train_last_game_pca = pca.fit_transform(X_train_last_game)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_exp = np.cumsum(explained_variance_ratio)
aux = 0
n_comp = 0
for i in range(len(cumulative_exp)):
	aux = cumulative_exp[i]
	if aux > 0.7:
		print('Number of components: {}'.format(i + 1))
		n_comp = i + 1
		break
pca = PCA(n_components=n_comp)
X_train_last_game_transformed = pca.fit_transform(X_train_last_game)
X_test_last_game_transformed = pca.transform(X_test_last_game)

lr = LogisticRegression()
lr.fit(X_train_last_game_transformed, y_train)
train_score = lr.score(X_train_last_game_transformed, y_train) * 100
test_score = lr.score(X_test_last_game_transformed, y_test) * 100
print('Logistic regression score in the training set: {:.2f}%'.format(train_score))
print('Logistic regression score in the testing set: {:.2f}%'.format(test_score))

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)
mlp.fit(X_train_last_game_transformed, y_train)
train_score = mlp.score(X_train_last_game_transformed, y_train) * 100
test_score = mlp.score(X_test_last_game_transformed, y_test) * 100
print('Multi-layer Perceptron score in the training set: {:.2f}%'.format(train_score))
print('Multi-layer Perceptron score in the testing set: {:.2f}%'.format(test_score))




from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=42)
clf2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)
clf3 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
clf4 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
clf5 = GaussianNB()

classifiers = [clf1, clf2, clf3, clf4, clf5]

from majority_vote_classifier import Majority_Vote_Classifier
mvc = Majority_Vote_Classifier(classifiers)
mvc.fit(X_train_last_game_transformed, y_train)
train_score = mvc.score(X_train_last_game_transformed, y_train) * 100
test_score = mvc.score(X_test_last_game_transformed, y_test) * 100
print('Majority vote classifier score in the training set: {:.2f}%'.format(train_score))
print('Majority vote classifier score in the testing set: {:.2f}%'.format(test_score))