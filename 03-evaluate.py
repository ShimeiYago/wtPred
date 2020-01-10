#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime
import argparse
import os
from wtpred import preprocess

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt


# predicted results dir
PRED_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'outputs/02-predict')

# output dir for plot
PLOTDIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'outputs/03-plot')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', help='attracID', default='L00')
    parser.add_argument('-d', '--datasetdir', help='dataset dir path', default='example-datasets')
    args = parser.parse_args()

    # dataset dir
    datasetdir = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.datasetdir)



    ### 01. prepare actual waittime data ###
    # which park
    if args.id[0] is 'L':
        park = 'land'
    else:
        park = 'sea'

    # get actual waittime eachtime
    filepath = os.path.join(datasetdir, 'waittime', f'{park}.csv')
    actdf_eachtime = pd.read_csv(filepath, index_col='datetime', parse_dates=['datetime'])[[args.id]]
    actdf_eachtime.columns = ['actual'] # rename column

    # get actual mean waittime eachday
    actdf_eachday = preprocess.calucrate_mean_std_eachday(actdf_eachtime)[['mean']]
    actdf_eachday.columns = ['actual'] # rename column



    ### 02. prepare predicted waittime data ###
    # get predicted waittime eachtime
    filepath = os.path.join(PRED_DIR, args.id, 'predicted-waittime-eachtime.csv')
    preddf_eachtime = pd.read_csv(filepath, index_col='datetime', parse_dates=['datetime'])[['value']]
    preddf_eachtime.columns = ['predict'] # rename column

    filepath = os.path.join(PRED_DIR, args.id, 'predicted-mean-waittime-eachday.csv')
    preddf_eachday = pd.read_csv(filepath, index_col='date', parse_dates=['date'])[['mean']]
    preddf_eachday.columns = ['predict'] # rename column



    ### 03. evaluate ###
    print('--- prediction of mean waittime eachday ---')
    evaluate(actdf_eachday, preddf_eachday)

    print('--- prediction of waittime eachtime ---')
    evaluate(actdf_eachtime, preddf_eachtime)



def evaluate(actdf, preddf):
    # join
    df = joindf(actdf, preddf)

    actual = df['actual']
    pred = df['predict']

    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    print( 'MSE : %.1f' %mse)
    print( 'RMSE : %.2f' %rmse)
    print( 'MAE : %.2f' %mae)
    print('R2 : %.3f' %r2)



def joindf(actdf, preddf):
    # join
    df = actdf.join(preddf, how='inner')

    # drop NaN
    df = df.dropna(how='any').copy()

    return df
 


if __name__ == '__main__':
    main()