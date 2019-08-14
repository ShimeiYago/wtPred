import pandas as pd
import numpy as np
import datetime
import copy
from fbprophet import Prophet


def ranfore(model, explanatories_df):

    pred = model.predict(explanatories_df)

    preddf = explanatories_df[[]].copy()
    preddf['pred'] = pred

    return preddf


def online_trend(waittimedf, predict_periods=365):

    data = waittimedf.reset_index()
    data.columns = ['ds', 'y']

    # FBprophet
    model = Prophet(weekly_seasonality=False, daily_seasonality=False)
    model.fit(data)
    future = model.make_future_dataframe(periods=predict_periods)
    forecast = model.predict(future)
    forecast = forecast.iloc[-predict_periods]

    # trend + yealy
    df = forecast.copy()
    df = df[['ds', 'trend', 'yearly']].copy()
    df = df.rename(columns={'ds': 'date'})
    pd.to_datetime(df['date'])
    df = df.set_index('date')
    df['trend_and_seasonal'] = df['trend'] + df['yearly']
    df = df[['trend_and_seasonal']]
    df = df.rename(columns={'trend_and_seasonal': 'trend'})

    return df


def online_error(model, errors_array):
    if np.nan in errors_array:
        return 0
    
    pred = model.predict(errors_array)

    return pred[0]


def ranfore_interval(model, X, percentile=95):
    err_down = []
    err_up = []
    for x in range(len(X)):
        preds = []
        for pred in model.estimators_:
            preds.append(pred.predict(X[[x]])[0])
        err_down.append(np.percentile(preds, (100 - percentile) / 2. ))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
    return err_down, err_up