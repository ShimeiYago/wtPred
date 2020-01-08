#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import datetime
import argparse
import os
import pickle
import wtpred
from wtpred import preprocess
import make_explanatory

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', help='attracID', default='L00')
    parser.add_argument('-d', '--datasetdir', help='dataset dir path', default='test-datasets')
    parser.add_argument('-l', '--lower_date', help='lower date for prediction (Y-M-D)', default='2015-1-1')
    parser.add_argument('-u', '--upper_date', help='upper date for prediction (Y-M-D)', default='2015-12-31')
    args = parser.parse_args()

    # check date format
    datetime.datetime.strptime(args.lower_date, '%Y-%m-%d')
    datetime.datetime.strptime(args.upper_date, '%Y-%m-%d')

    # dataset dir
    datasetdir = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.datasetdir)

    pre_1day_date = datetime.datetime.strftime( datetime.datetime.strptime(args.lower_date, '%Y-%m-%d') - datetime.timedelta(days=1),'%Y-%m-%d' )
    pre_7day_date = datetime.datetime.strftime( datetime.datetime.strptime(args.lower_date, '%Y-%m-%d') - datetime.timedelta(days=10),'%Y-%m-%d' )



    ### 01. prepare past waittime data ###
    # which park
    if args.id[0] is 'L':
        park = 'land'
    else:
        park = 'sea'

    # get waittime data
    filepath = os.path.join(datasetdir, 'waittime', f'{park}.csv')
    past_waittimedf = pd.read_csv(filepath, index_col='datetime', parse_dates=['datetime'])[[args.id]][:pre_1day_date]
    past_waittimedf.columns = ['waittime']


    
    ### 02. prapare explanatory data ###
    explanatory_eachday_df = make_explanatory.eachday(pre_7day_date, args.upper_date, datasetdir)
    explanatory_eachtime_df = make_explanatory.eachtime(pre_7day_date, args.upper_date, args.id[0], datasetdir)
    


    ### 03. predict ###
    # load model
    model = wtpred.model()
    loaded_obj = pickle.load(open(f'models/{args.id}.pickle', 'rb'))
    model.load(loaded_obj)


    # predict
    preddf_eachday, preddf_eachtime = model.predict(past_waittimedf, explanatory_eachday_df, explanatory_eachtime_df, args.lower_date, args.upper_date)
        # preddf_eachday  : predicted mean waittime each day (columns -> ['mean'])
        # preddf_eachtime : predicted waittime each datetime and possible lower-and-upper values (columns -> ['value', 'value_lower', 'value_upper'])


    # outdir = f'simurated-result/{args.id}'
    # os.makedirs(outdir, exist_ok=True)

    # preddf_eachday.to_csv(f'{outdir}/mean.csv')
    # preddf_eachtime.to_csv(f'{outdir}/pred.csv')



if __name__ == '__main__':
    main()