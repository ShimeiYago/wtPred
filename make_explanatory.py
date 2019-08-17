import pandas as pd
import numpy as np
import datetime

def eachday(lower_date_str, upper_date_str, datasetdir_path):

    ### each date data ###
    df = pd.read_csv(f'{datasetdir_path}/date_data/datedata.csv', index_col='date', parse_dates=['date'])
    df = df[['holiday', 'weekday', 'dayoff', 'dayoff_prev', 'dayoff_next']].copy()

    # onehot
    df = pd.get_dummies(df, columns=['weekday'])

    datasetdf = df.copy()


    ### weather data ###
    df = pd.read_csv(f'{datasetdir_path}/weather/perday/data.csv', index_col='date', parse_dates=['date'])
    df['rainy'] = df['precipitation'].apply(lambda x: 0 if x == 0 else 1)
    df = df[['precipitation', 'rainy']].copy()

    # join
    datasetdf = datasetdf.join(df, how='inner')


    ### add other columns ###
    df = datasetdf.copy()

    df['month'] = df.index.month
    df['md'] = df.index.strftime('%m-%d')

    # onehot
    df = pd.get_dummies(df, columns=['month', 'md'])

    datasetdf = df.copy()

    
    return datasetdf[lower_date_str:upper_date_str]
