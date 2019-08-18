import pandas as pd
import numpy as np
import datetime
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor


def calucrate_mean_std_eachday(waittimedf):
    meanstddf = pd.DataFrame()

    # Multi Index
    multidf = waittimedf.copy()
    multidf.set_index([multidf.index.date, multidf.index], inplace=True)
    multidf.index.names = ['date', 'datetime']

    for name, group in multidf.groupby(level='date'):
        #drop multiindex
        df = group.reset_index(level='date').copy()
        df.drop(columns='date', inplace=True)

        # add mean and std
        meanstddf.loc[name, 'mean'] = df.mean().values
        meanstddf.loc[name, 'std'] = df.std().values
    
    meanstddf.index.name = 'date'
    
    return meanstddf


def feature_selection(datasetdf):
    # delete NaN
    df = datasetdf.dropna(how='any')

    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), threshold="median")

    selector.fit(X, y)
    mask = selector.get_support()

    selected_features_list = [f for i,f in enumerate(X.columns) if mask[i]]
    return selected_features_list


def adapt_features(df, features_list):
    for feature in features_list:
        if feature in df.columns:
            continue
        
        df[feature] = 0
        
    return df[features_list]
