import pandas as pd
import numpy as np
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


from sklearn.ensemble import RandomForestRegressor
def randomforest_model(traindf):

    # define model
    model = RandomForestRegressor(
        random_state=123,
        bootstrap=True, criterion='mse', max_depth=30,
        max_features='auto', max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_impurity_split=None,
        min_samples_leaf=1, min_samples_split=2,
        min_weight_fraction_leaf=0.0, n_estimators=90, n_jobs=1,
        oob_score=False, verbose=0, warm_start=False
    )

    # learning
    traindf = traindf.dropna(how='any') # drop nan
    X_train = traindf.iloc[:, 1:]
    y_train = traindf.iloc[:, 0]

    model.fit(X_train, y_train)

    return model
