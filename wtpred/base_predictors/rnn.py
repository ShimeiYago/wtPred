import pandas as pd
import numpy as np
import datetime

from wtpred.base_predictors import randomforest

import statistics
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf
import keras.backend as K


def make_model(datasetdf, N_lookbucks=6):

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
        model = randomforest.make_model(traindf)

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