import pandas as pd
import numpy as np
import datetime

def eachday(lowerdate_str, upperdate_str, datasetdir_path):

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

    
    return datasetdf[lowerdate_str:upperdate_str]


def eachtime(lowerdate_str, upperdate_str, park, datasetdir_path):

    ### make empty df ###
    explanatory_df = make_emptydf_eachtime(lowerdate_str, upperdate_str)


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
    explanatory_df['regulateddt'] = explanatory_df.index
    explanatory_df['regulateddt'] = explanatory_df['regulateddt'].apply(lambda dt: regulate_dt(dt))
    explanatory_df.set_index(['regulateddt', explanatory_df.index], inplace=True)

    wedf.index.name = 'regulateddt'
    df = explanatory_df.join(wedf, how='inner')

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


    ### fit index to openclose
    df = fit_to_openclose(df, f'open{park}', f'close{park}')


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


    explanatory_df = df.copy()

    
    return explanatory_df


def make_emptydf_eachtime(lowerdate_str, upperdate_str):

    datetime_index_list = []
    for date in pd.date_range(lowerdate_str, upperdate_str):
        date_str = date.strftime('%Y-%m-%d')

        oneday_datetime_index = pd.date_range(start=f'{date_str} 8:15', end=f'{date_str} 21:45', freq='30T')
        datetime_index_list.extend(oneday_datetime_index)

    datetime_index = pd.DatetimeIndex(datetime_index_list)

    df = pd.DataFrame(index=datetime_index)
    return df



# 各時刻を比較演算子で開園・閉園時間と比較して、おかしかったら削除するという戦略
def fit_to_openclose(df, colname_open, colname_close):
    for index, row in df[[colname_open, colname_close]].copy().iterrows():
        nowtime = datetime.time(index.hour,index.minute)
        opentime = datetime.datetime.strptime(row[colname_open], '%H:%M').time()
        closetime = datetime.datetime.strptime(row[colname_close], '%H:%M').time()

        # 開園時間で比較
        if nowtime < opentime:
            df.drop(index, inplace=True)
        
        # 閉園時間で比較
        elif nowtime > closetime:
            df.drop(index, inplace=True)
    
    return df
