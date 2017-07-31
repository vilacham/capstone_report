# Imports
import pandas as pd
from functions import preprocess, get_frequent_outliers, standardize, divide_data, get_n_principal_components
from sklearn.model_selection import train_test_split

# Import data and create a copy of it
try:
    original_data = pd.read_excel('capstone_database.xlsx')
    data = original_data
    print('Data was successfully imported and has {} samples with {} features each.'.format(*data.shape))
except:
    print('Data could not be loaded. Is it missing?')

# Preprocess data
data = preprocess(data)

# Drop outliers for more than one feature
outliers = get_frequent_outliers(data)
good_data = data.drop(data.index[outliers]).reset_index(drop=True)
print('Original data had {} samples.'.format(data.shape[0]))
print('{} samples were outliers for more than one feature).'.format(len(outliers)))
print('New data has {} samples.'.format(good_data.shape[0]))

# Split data into training set and testing set
x, y = good_data.iloc[:, :-1], good_data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize data sets
X_train, X_test = standardize(X_train, X_test)

# Divide data
train_lg, test_lg, train_l2g, test_l2g, train_l5g, test_l5g = divide_data(X_train, X_test)

# Get principal components for the last game data
n_comp = get_n_principal_components(train_lg)
print('Number of components: {}.'.format(n_comp))