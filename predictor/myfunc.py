import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor



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


def round(df, rounddict, attracID):

    roundlist = rounddict[attracID]

    ### roundlistの中から、wtに最も近い値を取り出す関数
    def round_waittime(wt):
        idx = np.abs(np.asarray(roundlist) - wt).argmin()
        return roundlist[idx]
    

    ### round ###
    # for col in df.columns:
    #     wtlist = list(df[col])
    #     df[col] = list(map(round_waittime, wtlist))

    firstcol = df.columns[0]
    wtlist = list(df[firstcol])
    df[firstcol] = list(map(round_waittime, wtlist))

    

    return df



def make_dataset_eachday(datasetdf, datasetdir_path):

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


def make_dataset_eachtime(datasetdf, datasetdir_path, park):

    ### opening hour data ###
    ocdf = pd.read_csv(f'{datasetdir_path}/openclosetime/openclose.csv', index_col='date', parse_dates=['date'])


    ### each date data ###
    dtdf = pd.read_csv(f'{datasetdir_path}/date_data/datedata.csv', index_col='date', parse_dates=['date'])
    dtdf = dtdf[['dayoff']].copy()


    ### weather data ###
    wedf = pd.read_csv(f'{datasetdir_path}/weather/weather.csv', index_col='datetime', parse_dates=['datetime'])
    wedf['rainy'] = wedf['precipitation'].apply(lambda x: 0 if x == 0 else 1)
    wedf = wedf[['precipitation', 'rainy']].copy()

    ### join ###

    ## datasetdf + wedf
    def regulate_dt(dt):
        m = dt.minute
        regulated_dt = dt.replace(minute=0, second=0, microsecond=0)

        if 20<=m and m<=59:
            regulated_dt = regulated_dt + datetime.timedelta(hours=1)

        return regulated_dt

    # Multi Index
    datasetdf['regulateddt'] = datasetdf.index
    datasetdf['regulateddt'] = datasetdf['regulateddt'].apply(lambda dt: regulate_dt(dt))
    datasetdf.set_index(['regulateddt', datasetdf.index], inplace=True)

    wedf.index.name = 'regulateddt'
    df = datasetdf.join(wedf, how='inner')

    # drop multiindex
    df.reset_index(level='regulateddt', inplace=True)
    df.drop(columns='regulateddt', inplace=True)


    ## df + ocdf + dtdf

    # Multi Index
    df.set_index([df.index.date, df.index], inplace=True)
    df.index.names = ['date', 'datetime']

    df = df.join(ocdf, how='inner')
    df = df.join(dtdf, how='inner')

    #drop multiindex
    df.reset_index(level='date', inplace=True)
    df.drop(columns='date', inplace=True)


    ### add other columns ###
    df['time'] = df.index.time

    df['month'] = df.index.month
    df['md'] = df.index.strftime('%m-%d')


    ### from-open and to-close ###
    def fromopen(row, park):
        x = row['open'+park]
        openH, openM = [int(x) for x in x.split(":")]
        opendt = datetime.datetime(2018, 1, 1, openH, openM)
        nowH, nowM = row['time'].hour, row['time'].minute
        nowdt = datetime.datetime(2018, 1, 1, nowH, nowM)
        elap = nowdt - opendt
        h = int(elap.seconds / 3600)
        m = int((elap.seconds % 3600) / 60)
        return str(h) +':'+ str(m)

    def toclose(row, park):
        x = row['close'+park]
        closeH, closeM = [int(x) for x in x.split(":")]
        closedt = datetime.datetime(2018, 1, 1, closeH, closeM)
        nowH, nowM = row['time'].hour, row['time'].minute
        nowdt = datetime.datetime(2018, 1, 1, nowH, nowM)
        elap = closedt - nowdt
        h = int(elap.seconds / 3600)
        m = int((elap.seconds % 3600) / 60)
        return str(h) +':'+ str(m)

    df['fromopen'] = df.apply(fromopen, park=park, axis=1)
    df['toclose'] = df.apply(toclose, park=park, axis=1)

    del df['openL']
    del df['closeL']
    del df['openS']
    del df['closeS']


    ### onehot ###
    df = pd.get_dummies(df, columns=['time', 'fromopen', 'toclose', 'month', 'md'])


    datasetdf = df.copy()

    
    return datasetdf



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