def preprocess(data):
    # Imports
    import numpy as np
    import pandas as pd

    # Rename columns
    data.columns = ['DATE', 'A TEAM', 'H TEAM', 'osite', 'A PTS', 'H PTS', 'A STK', 'H STK', 'A PTS LG', 'A FGM LG',
                    'A FGA LG', 'A 3PM LG', 'A 3PA LG', 'A FTM LG', 'A FTA LG', 'A OREB LG', 'A DREB LG', 'A REB LG',
                    'A AST LG', 'A TOV LG', 'A STL LG', 'A BLK LG', 'H PTS LG', 'H FGM LG', 'H FGA LG', 'H 3PM LG',
                    'H 3PA LG', 'H FTM LG', 'H FTA LG', 'H OREB LG', 'H DREB LG', 'H REB LG', 'H AST LG', 'H TOV LG',
                    'H STL LG', 'H BLK LG', 'A PTS L2G', 'A FGM L2G', 'A FGA L2G', 'A 3PM L2G', 'A 3PA L2G',
                    'A FTM L2G', 'A FTA L2G', 'A OREB L2G', 'A DREB L2G', 'A REB L2G', 'A AST L2G', 'A TOV L2G',
                    'A STL L2G', 'A BLK L2G', 'H PTS L2G', 'H FGM L2G', 'H FGA L2G', 'H 3PM L2G', 'H 3PA L2G',
                    'H FTM L2G', 'H FTA L2G', 'H OREB L2G', 'H DREB L2G', 'H REB L2G', 'H AST L2G', 'H TOV L2G',
                    'H STL L2G', 'H BLK L2G', 'A PTS L5G', 'A FGM L5G', 'A FGA L5G', 'A 3PM L5G', 'A 3PA L5G',
                    'A FTM L5G', 'A FTA L5G', 'A OREB L5G', 'A DREB L5G', 'A REB L5G', 'A AST L5G', 'A TOV L5G',
                    'A STL L5G', 'A BLK L5G', 'H PTS L5G', 'H FGM L5G', 'H FGA L5G', 'H 3PM L5G', 'H 3PA L5G',
                    'H FTM L5G', 'H FTA L5G', 'H OREB L5G', 'H DREB L5G', 'H REB L5G', 'H AST L5G', 'H TOV L5G',
                    'H STL L5G', 'H BLK L5G', 'OVT', 'DAY', 'MTH', 'PLAYOFFS']

    # Replace '-' and 'away' by NaN values and then drop replace it
    data.replace(to_replace='-', value=np.nan, inplace=True, regex=True)
    data.replace(to_replace='away', value=np.nan, inplace=True, regex=True)
    data.dropna(inplace=True)
    data.reset_index(inplace=True)

    # Deal with 'DAY' and 'MTH' features (both are categorical)
    data = pd.get_dummies(data, columns=['DAY', 'MTH'])

    # Create label column (1 for home team; 0 for visitor team)
    data['WINNER'] = (data['H PTS'] > data['A PTS']).astype(int)

    # Drop unnecessary columns
    columns_to_drop = ['index', 'DATE', 'A TEAM', 'H TEAM', 'osite', 'A PTS', 'H PTS']
    data.drop(columns_to_drop, axis=1, inplace=True)

    # Convert all features to numerical
    data = data.apply(pd.to_numeric)

    return data


def get_frequent_outliers(data):
    # Imports
    import numpy as np
    from collections import Counter

    counter = Counter()
    for feature in data.iloc[:, 2:87].columns:
        quartile_1 = np.percentile(data[feature], 25)
        quartile_3 = np.percentile(data[feature], 75)
        step = (quartile_3 - quartile_1) * 1.5
        outliers = data[~((data[feature] >= quartile_1 - step) & (data[feature] <= quartile_3 + step))]
        counter.update(outliers.index.values)
    frequent_outliers = [outlier[0] for outlier in counter.items() if outlier[1] > 1]
    return frequent_outliers


def standardize(train, test):
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    columns_to_standardize = ['A PTS LG', 'A FGM LG', 'A FGA LG', 'A 3PM LG', 'A 3PA LG', 'A FTM LG', 'A FTA LG',
                              'A OREB LG', 'A DREB LG', 'A REB LG', 'A AST LG', 'A TOV LG', 'A STL LG', 'A BLK LG',
                              'H PTS LG', 'H FGM LG', 'H FGA LG', 'H 3PM LG', 'H 3PA LG', 'H FTM LG', 'H FTA LG',
                              'H OREB LG', 'H DREB LG', 'H REB LG', 'H AST LG', 'H TOV LG', 'H STL LG', 'H BLK LG',
                              'A PTS L2G', 'A FGM L2G', 'A FGA L2G', 'A 3PM L2G', 'A 3PA L2G', 'A FTM L2G',
                              'A FTA L2G', 'A OREB L2G', 'A DREB L2G', 'A REB L2G', 'A AST L2G', 'A TOV L2G',
                              'A STL L2G', 'A BLK L2G', 'H PTS L2G', 'H FGM L2G', 'H FGA L2G', 'H 3PM L2G',
                              'H 3PA L2G', 'H FTM L2G', 'H FTA L2G', 'H OREB L2G', 'H DREB L2G', 'H REB L2G',
                              'H AST L2G', 'H TOV L2G', 'H STL L2G', 'H BLK L2G', 'A PTS L5G', 'A FGM L5G',
                              'A FGA L5G', 'A 3PM L5G', 'A 3PA L5G', 'A FTM L5G', 'A FTA L5G', 'A OREB L5G',
                              'A DREB L5G', 'A REB L5G', 'A AST L5G', 'A TOV L5G', 'A STL L5G', 'A BLK L5G',
                              'H PTS L5G', 'H FGM L5G', 'H FGA L5G', 'H 3PM L5G', 'H 3PA L5G', 'H FTM L5G',
                              'H FTA L5G', 'H OREB L5G', 'H DREB L5G', 'H REB L5G', 'H AST L5G', 'H TOV L5G',
                              'H STL L5G', 'H BLK L5G', 'OVT']
    standard_scaler = StandardScaler()
    pd.options.mode.chained_assignment = None  # default = 'warn'
    train.loc[:, columns_to_standardize] = standard_scaler.fit_transform(train[columns_to_standardize])
    test.loc[:, columns_to_standardize] = standard_scaler.transform(test[columns_to_standardize])
    return train, test


def divide_data(train, test):
    # Divide training set and get last game data
    train_lg = train[list(train.columns[:30]) + list(train.columns[86:])]
    test_lg = test[list(test.columns[:30]) + list(test.columns[86:])]

    # Divide training set and get last two games data
    train_l2g = train[list(train.columns[:2]) + list(train.columns[30:58]) + list(train.columns[86:])]
    test_l2g = test[list(test.columns[:2]) + list(test.columns[30:58]) + list(test.columns[86:])]

    # Divide training set and get last five games data
    train_l5g = train[list(train.columns[:2]) + list(train.columns[58:])]
    test_l5g = test[list(test.columns[:2]) + list(test.columns[58:])]

    return train_lg, test_lg, train_l2g, test_l2g, train_l5g, test_l5g


def get_n_principal_components(data):
    # Imports
    from sklearn.decomposition import PCA
    import numpy as np

    pca = PCA(n_components=None)
    # data_pca = pca.fit_transform(data)
    pca.fit_transform(data)
    evr = pca.explained_variance_ratio_
    c_evr = np.cumsum(evr)
    n_comp = 0
    for i in range(len(c_evr)):
        aux = c_evr[i]
        if aux >= 0.6:
            n_comp = i + 1
            break
    n_features = data.shape[1]
    # plot_pca_graph(n_features, evr, c_evr)
    return n_comp


def plot_pca_graph(n_features, evr, c_evr):
    # Import plt
    import matplotlib.pyplot as plt

    # Clear the plot
    plt.clf()

    # Plot data
    plt.bar(range(n_features), evr, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(n_features), c_evr, where='mid', label='Cumulative explained variance')

    # Plot guide line
    plt.plot([i for i in range(n_features)], [0.6] * n_features, '--', color='g')

    # Plot legend
    plt.legend(loc='best')

    # Display
    plt.show()


def reduce(train, test, n_comp):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_comp)
    train_reduced = pca.fit_transform(train)
    test_reduced = pca.transform(test)
    return train_reduced, test_reduced


def create_classifiers():
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    classifiers = list()
    classifiers.append((LogisticRegression(), {'C': [.001, .01, .1, 1, 10, 100, 1000],
                                               'solver': ('liblinear', 'sag', 'newton-cg', 'lbfgs'),
                                               'class_weight': (None, 'balanced')}))
    classifiers.append((DecisionTreeClassifier(), {'criterion': ('gini', 'entropy'),
                                                  'min_samples_split': [.1, .05, .025, .01, 2],
                                                  'min_samples_leaf': [.05, .025, .0125, .005, 1]}))
    classifiers.append((KNeighborsClassifier(), {'n_neighbors': [1, 3, 5, 7, 9],
                                                 'weights': ('uniform', 'distance'),
                                                 'metric': ('euclidean', 'manhattan', 'chebyshev', 'minkowski')}))
    #classifiers.append((MLPClassifier(), {'activation' : ('identity', 'logistic', 'tanh', 'relu'),
    #                                      'solver': ('lbfgs', 'sgd', 'adam'),
    #                                      'learning_rate': ('constant', 'invscaling', 'adaptive')}))
    #classifiers.append((SVC(), {'C': [.001, .01, .1, 1, 10, 100, 1000],
    #                            'kernel': ('poly', 'rbf', 'sigmoid')}))
    classifiers.append((GaussianNB(), {}))
    #classifiers.append((RandomForestClassifier(), {'n_estimators': [10, 20, 25, 50, 100],
    #                                               'criterion': ('giny', 'entropy'),
    #                                               'max_features': ('sqrt', 'log2'),
    #                                               'min_samples_split': [.1, .05, .025, .01, 2],
    #                                               'min_samples_leaf': [.05, .025, .0125, .005, 1]}))
    return classifiers


def optimize_classifiers(classifiers, features, labels):
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    optimized_classifiers = list()
    for clf, param in classifiers:
        gs = GridSearchCV(estimator=clf, param_grid=param, scoring='accuracy', cv=skf, n_jobs=1)
        gs = gs.fit(features, labels)
        print('-' * 50)
        print('Best parameters: {}'.format(gs.best_params_))
        print('Best score: {:.3f}'.format(gs.best_score_))
        optimized_classifiers.append(gs.best_estimator_)
    return optimized_classifiers
