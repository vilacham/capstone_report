def preprocess(data):
	# Imports
	import numpy as np
	import pandas as pd

	# Rename data columns
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
	
	return data


def get_n_principal_components(data):
    print(data)
    from sklearn.decomposition import PCA
    import numpy as np
    pca = PCA(n_components=None)
    data_pca = pca.fit_transform(data)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_exp = np.cumsum(explained_variance_ratio)
    auxiliar = 0
    n_comp = 0
    for i in range(len(cumulative_exp)):
        auxiliar = cumulative_exp[i]
        if auxiliar = 0.7:
            n_comp = i + 1
            break
    return n_comp