import pandas as pd
import datetime

from fbprophet import Prophet


def de_trend(waittimedf):

    # rename columns
    data = waittimedf.reset_index()
    data.columns = ['ds', 'y']

    # FBprophet
    model = Prophet(weekly_seasonality=False, daily_seasonality=False)
    model.fit(data)
    future = model.make_future_dataframe(periods=0)
    analyzed_df = model.predict(future)

    # trend + yealy
    df = analyzed_df.copy()
    df = df[['ds', 'trend', 'yearly']].copy()
    df = df.rename(columns={'ds': 'date'})
    pd.to_datetime(df['date'])
    df = df.set_index('date')
    df['trend_and_seasonal'] = df['trend'] + df['yearly']
    df = df[['trend_and_seasonal']]
    df = df.rename(columns={'trend_and_seasonal': 'trend'})

    # waittime - trend -> resid
    waittimedf['trend'] = df.iloc[:, 0].values
    waittimedf['resid'] = waittimedf.iloc[:, 0] - waittimedf['trend']
    
    return waittimedf



def predict_trend(past_waittime_df, upperdate_str):
    # periods of predict
    periods_predict = ( datetime.datetime.strptime(upperdate_str, '%Y-%m-%d') - past_waittime_df.index[-1] ).days

    # rename columns
    data = past_waittime_df.reset_index()
    data.columns = ['ds', 'y']

    # FBprophet
    model = Prophet(weekly_seasonality=False, daily_seasonality=False)
    model.fit(data)
    future = model.make_future_dataframe(periods=periods_predict)
    analyzed_df = model.predict(future)

    # trend + yealy -> trend
    df = analyzed_df.copy()
    df = df[['ds', 'trend', 'yearly']].copy()
    df = df.rename(columns={'ds': 'date'})
    pd.to_datetime(df['date'])
    df = df.set_index('date')
    df['trend_and_seasonal'] = df['trend'] + df['yearly']
    df = df[['trend_and_seasonal']]
    df = df.rename(columns={'trend_and_seasonal': 'trend'})

    # print(df)
    # print(lowerdate_str)
    return df[['trend']]