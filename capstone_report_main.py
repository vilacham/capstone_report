# Imports
import pandas as pd
import functions as f
from sklearn.model_selection import train_test_split

# Import data and create a copy of it
try:
    original_data = pd.read_excel('capstone_database.xlsx')
    data = original_data
    print('Data was successfully imported and has {} samples with {} features each.'.format(*data.shape))
except:
    print('Data could not be loaded. Is it missing?')

# Preprocess data
data = f.preprocess(data)

# Drop outliers for more than one feature
outliers = f.get_frequent_outliers(data)
good_data = data.drop(data.index[outliers]).reset_index(drop=True)
print('Original data had {} samples.'.format(data.shape[0]))
print('{} samples were outliers for more than one feature).'.format(len(outliers)))
print('New data has {} samples.'.format(good_data.shape[0]))

# Split data into training set and testing set
X, y = good_data.iloc[:, :-1], good_data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize data sets
X_train, X_test = f.standardize(X_train, X_test)

# Divide data
train_lg, test_lg, train_l2g, test_l2g, train_l5g, test_l5g = f.divide_data(X_train, X_test)

# Get principal components for the last game data
n_comp = f.get_n_principal_components(train_lg)
print('Number of components: {}.'.format(n_comp))

# Reduce last game data dimensionality to n_comp features
train_lg_reduced, test_lg_reduced = f.reduce(train_lg, test_lg, n_comp)

# Create classifiers
classifiers = f.create_classifiers()

from sklearn.model_selection import StratifiedKFold, GridSearchCV
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
clf_list = list()
for classifier, parameters in classifiers:
    gs = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=skf, n_jobs=1)
    gs = gs.fit(train_lg_reduced, y_train)
    print('----------\nBest parameters: {}\nBest score: {:.2f} %'.format(gs.best_params_, gs.best_score_ * 100))
    clf_list.append(gs.best_estimator_)

from majority_vote_classifier import MajorityVoteClassifier
mvc = MajorityVoteClassifier(clf_list)
mvc.fit(train_lg_reduced, y_train)
train_score = mvc.score(train_lg_reduced, y_train) * 100
test_score = mvc.score(test_lg_reduced, y_test) * 100
print('Majority vote classifier score in the training set: {:.2f}%'.format(train_score))
print('Majority vote classifier score in the testing set: {:.2f}%'.format(test_score))