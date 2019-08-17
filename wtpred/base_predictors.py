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


def predict_trend(past_waittime_df, lowerdate_str, upperdate_str):
    # periods of predict
    periods_predit = ( datetime.datetime.strptime(upperdate_str, '%Y-%m-%d') - past_waittime_df.index[-1] ).days

    # rename columns
    data = past_waittime_df.reset_index()
    data.columns = ['ds', 'y']

    # FBprophet
    model = Prophet(weekly_seasonality=False, daily_seasonality=False)
    model.fit(data)
    future = model.make_future_dataframe(periods=periods_predit)
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

    
    return df[['trend']][lowerdate_str:upperdate_str]


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


import statistics
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf
import keras.backend as K
def error_predict_model(datasetdf, N_lookbucks=6):

    ### make error_seq ###
    df = datasetdf.copy()

    # Multi Index
    df.set_index([df.index.year, df.index], inplace=True)
    df.index.names = ['year', 'date']

    error_seq = []
    for name, _ in df.groupby(level='year'):

        # split dataset to test and train
        objective_year = str(name)
        testdf = datasetdf[objective_year].copy()
        traindf = datasetdf.drop(testdf.index)

        # make model
        model = randomforest_model(traindf)

        # predict
        testdf = testdf.dropna(how='any')
        X_test = testdf.iloc[:, 1:]
        y_test = testdf.iloc[:, 0]
        pred = model.predict(X_test)

        # calculate error
        error = y_test.values - pred
        error_seq += list(error)

    
    ### learn error_seq ###

    ## normalize
    mean = statistics.mean(error_seq)
    std = statistics.stdev(error_seq)
    error_seq = [(x - mean) / std for x in error_seq]


    ## make_dataset
    explanatory, response = [], []
    for i in range(len(error_seq)-N_lookbucks):

        # if there are NaN, skip
        if np.nan in error_seq[i:i+N_lookbucks+1]:
            continue

        explanatory.append(error_seq[i:i+N_lookbucks])
        response.append(error_seq[i+N_lookbucks])

    # reshape
    X = np.array(explanatory).reshape(len(explanatory), N_lookbucks, 1)
    y = np.array(response).reshape(len(response), 1)


    ## define model
    N_hidden = 1000
    N_inout_neurons = 1

    model = Sequential()
    model.add(SimpleRNN(N_hidden, batch_input_shape=(None, N_lookbucks, N_inout_neurons), return_sequences=False))
    model.add(Dense(N_hidden))
    model.add(Dense(N_hidden))
    model.add(Dense(N_inout_neurons))
    model.add(Activation("linear"))
    optimizer = Adam(lr=0.001)
    model.compile(loss="mean_squared_error", optimizer=optimizer)

    ## learn
    epochs = 100
    batch_size = 64

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    fit = model.fit(X, y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            callbacks=[early_stopping]
    )

    return model