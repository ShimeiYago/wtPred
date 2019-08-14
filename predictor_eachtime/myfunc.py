import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt


def calucrate_mean_std(waittimedf):
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


def make_dataset(datasetdf, datasetdir_path):

    ### each date data ###
    df = pd.read_csv(f'{datasetdir_path}/date_data/datedata.csv', index_col='date', parse_dates=['date'])
    df = df[['holiday', 'weekday', 'dayoff', 'dayoff_prev', 'dayoff_next']].copy()

    # onehot
    df = pd.get_dummies(df, columns=['weekday'])

    # join
    datasetdf = datasetdf.join(df, how='inner')


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

    
    return datasetdf


def adapt_features(df, features_list):
    for feature in features_list:
        if feature in df.columns:
            continue
        
        df[feature] = 0
        
    return df[features_list]


def save_result_plot(df, outdir):

    os.makedirs(outdir, exist_ok=True)

    # x軸表示用の関数
    def timerange(start, end, interval_minute=60):
        start = datetime.datetime(1,1,1,start[0], start[1])
        end = datetime.datetime(1,1,1,end[0], end[1])
        
        li = []
        x = (end-start).seconds / 60
        for n in range(0, int(x), interval_minute):
                li.append((start + datetime.timedelta(minutes=n)).time())
        
        return li


    # plot用の関数
    def saveplot(month, df):
        title = '2018_' + str(month)
        
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(1, 1, 1)
        df.plot(ax=ax)
        ax.set_xticks(list(df.index))
        ax.grid(which = "major", axis = "x", color = "gray", alpha = 0.8,
            linestyle = "-", linewidth = 1)
        
        plt.title(title)

        plt.xlabel('date')
        plt.ylabel('waittime')

        plt.savefig(f'{outdir}/{title}.png', bbox_inches='tight')
        plt.close()


    # plot waittime

    # Multi Index
    df.set_index([df.index.month, df.index.day], inplace=True)
    df.index.names = ['month', 'day']

    for name, group in df.groupby(level='month'):
        #drop multiindex
        group = group.reset_index(level='month').copy()
        group.drop(columns='month', inplace=True)
        
        saveplot(name, group)