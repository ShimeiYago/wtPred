import pandas as pd
import numpy as np
import datetime
import argparse
import os
import pickle
from predictor import learn_past, myfunc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('id', help='attracID') #args.id
    parser.add_argument('dataset', help='dataset dir path') #args.dataset
    parser.add_argument('-l', '--lower_date', help='lower date of dataset for learning') #args.lower_date
    parser.add_argument('-u', '--upper_date', help='upper date of dataset for learning') #args.upper_date
    args = parser.parse_args()

    ### dataset ###
    # which park
    if args.id[0] is 'L':
        park = 'land'
    else:
        park = 'sea'

    # get waittime data
    filepath = f'{args.dataset}/waittime/{park}.csv'
    waittimedf = pd.read_csv(filepath, index_col='datetime', parse_dates=['datetime'])[[args.id]]
    waittimedf.columns = ['waittime']

    # cut waittime data
    if args.lower_date:
        waittimedf = waittimedf[args.lower_date:].copy()
    
    if args.upper_date:
        waittimedf = waittimedf[:args.upper_date].copy()

    ## mean and std
    meanstddf = myfunc.calucrate_mean_std(waittimedf)

    ## normalize each day
    # Multi Index
    waittimedf.set_index([waittimedf.index.date, waittimedf.index], inplace=True)
    waittimedf.index.names = ['date', 'datetime']

    df = waittimedf.join(meanstddf, how='inner')

    # drop multiindex
    df.reset_index(level='date', inplace=True)
    df.drop(columns='date', inplace=True)


    df['value'] = (df['waittime'] - df['mean']) / df['std']
    df = df[['value']]


    ### learning ###

    # make dataset
    datasetdf = myfunc.make_dataset_eachtime(df, args.dataset, args.id[0])

    # feature selection
    features_list = myfunc.feature_selection(datasetdf)
    datasetdf = datasetdf[ ['value']+features_list ]

    # learn model
    resid_predict_model = learn_past.resid_predict_modeling(datasetdf)

    
    ### save models ###
    outdir = f'models/{args.id}/time'
    os.makedirs(outdir, exist_ok=True)

    # resid predict models (Random Forest)
    normal_predict_models = [normal_predict_model, features_list]
    pickle.dump(normal_predict_models, open(f'{outdir}/normalvalue_predict_models.pickle', 'wb'))



if __name__ == '__main__':
    main()