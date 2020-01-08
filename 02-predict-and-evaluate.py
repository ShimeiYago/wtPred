import pandas as pd
import numpy as np
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

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', help='attracID', default='L00')
    parser.add_argument('-l', '--lower_date', help='lower date for prediction', default='2015-1-1')
    parser.add_argument('-u', '--upper_date', help='upper date for learning', default='2015-12-31')
    args = parser.parse_args()

    # check format
    datetime.datetime.strptime(args.todate, '%Y-%m-%d')
    datetime.datetime.strptime(args.upperdate, '%Y-%m-%d')
    

    ### dataset ###
    # which park
    if args.id[0] is 'L':
        park = 'land'
    else:
        park = 'sea'

    # get waittime data
    filepath = f'dataset/waittime/{park}.csv'
    past_waittimedf = pd.read_csv(filepath, index_col='datetime', parse_dates=['datetime'])[[args.id]][:args.todate]
    past_waittimedf.columns = ['waittime']


    
    ### explanatory data ###
    lowerdate_str = '2018-12-1'
    upperdate_str = args.upperdate

    explanatory_eachday_df = make_explanatory.eachday(lowerdate_str, upperdate_str, 'dataset')
    explanatory_eachtime_df = make_explanatory.eachtime(lowerdate_str, upperdate_str, args.id[0], 'dataset')
    

    ### load models ###
    loaded_obj = pickle.load(open(f'models/{args.id}.pickle', 'rb'))

    model = wtpred.model()
    model.load(loaded_obj)


    ### predict ###
    pred_meandf, preddf = model.predict(past_waittimedf, explanatory_eachday_df, explanatory_eachtime_df, args.todate, args.upperdate)
        # columns -> ['mean'], ['value', 'value_lower', 'value_upper']


    outdir = f'simurated-result/{args.id}'
    os.makedirs(outdir, exist_ok=True)

    pred_meandf.to_csv(f'{outdir}/mean.csv')
    preddf.to_csv(f'{outdir}/pred.csv')



if __name__ == '__main__':
    main()