import pandas as pd
import numpy as np
import datetime
import argparse
import os
import pickle
from predictor import learn_past
from predictor import myfunc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('id', help='attracID') #args.id
    parser.add_argument('dataset', help='dataset dir path') #args.dataset
    parser.add_argument('-m', '--mean', action='store_true', help='mean predict model') #args.mean
    parser.add_argument('-s', '--std', action='store_true', help='std predict model') #args.std
    parser.add_argument('-l', '--lower_date', help='lower date of dataset for learning') #args.lower_date
    parser.add_argument('-u', '--upper_date', help='upper date of dataset for learning') #args.upper_date
    args = parser.parse_args()

    ### mean or std ###
    if args.std:
        mean_or_std = "std"
    else:
        mean_or_std = "mean"
    

    ### dataset ###
    # which park
    if args.id[0] is 'L':
        park = 'land'
    else:
        park = 'sea'

    # get waittime data
    filepath = f'{args.dataset}/waittime/{park}.csv'
    waittimedf = pd.read_csv(filepath, index_col='datetime', parse_dates=['datetime'])[[args.id]]

    # cut waittime data
    if args.lower_date:
        waittimedf = waittimedf[args.lower_date:].copy()
    
    if args.upper_date:
        waittimedf = waittimedf[:args.upper_date].copy()

    # mean and std
    meanstddf = myfunc.calucrate_mean_std(waittimedf)
    

    ### learning ###

    # detrend
    detrended_df = learn_past.trend_decompose(meanstddf[[mean_or_std]])
        # columns -> mean, trend, resid

    # make dataset
    datasetdf = myfunc.make_dataset_eachday(detrended_df[['resid']], args.dataset)

    # feature selection
    features_list = myfunc.feature_selection(datasetdf)
    datasetdf = datasetdf[ ['resid']+features_list ]

    # learn resid predict model
    resid_predict_model = learn_past.resid_predict_modeling(datasetdf)

    # learn error predict model
    error_predict_model, error_mean, error_std = learn_past.error_predict_modeling(datasetdf)

    
    ### save models ###
    outdir = f'models/{args.id}/{mean_or_std}'
    os.makedirs(outdir, exist_ok=True)

    # resid predict models (Random Forest)
    resid_predict_models = [resid_predict_model, features_list]
    pickle.dump(resid_predict_models, open(f'{outdir}/resid_predict_models.pickle', 'wb'))

    # error predict model (Keras)
    model_json_str = error_predict_model.to_json() # model
    weights = error_predict_model.get_weights() # parameter

    error_predict_models = [model_json_str, weights, error_mean, error_std]
    pickle.dump(error_predict_models, open(f'{outdir}/error_predict_models.pickle', 'wb'))



if __name__ == '__main__':
    main()