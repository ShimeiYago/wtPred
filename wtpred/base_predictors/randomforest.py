import pandas as pd
import numpy as np
import datetime

from sklearn.ensemble import RandomForestRegressor


def make_model(traindf):

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


def predict(model, explanatory_df):

    pred = model.predict(explanatory_df)

    preddf = explanatory_df[[]].copy()
    preddf['pred'] = pred

    return preddf


def predict_interval(model, explanatorydf, percentile=95):
    X = explanatorydf.values

    err_down = []
    err_up = []
    for x in range(len(X)):
        preds = []
        for pred in model.estimators_:
            preds.append(pred.predict(X[[x]])[0])
        err_down.append(np.percentile(preds, (100 - percentile) / 2. ))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
    return err_down, err_up