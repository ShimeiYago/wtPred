import pandas as pd
import numpy as np
import datetime
import argparse
import os
import pickle
from predictor import learn_past, predict_future
from predictor import myfunc

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='dataset dir path') #args.dataset
    parser.add_argument('id', help='attracID') #args.id
    parser.add_argument('-l', '--lower_date', help='lower date for predict') #args.lower_date
    parser.add_argument('-u', '--upper_date', help='upper date for predict') #args.upper_date
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

    # mean and std
    meanstddf = myfunc.calucrate_mean_std(waittimedf)

    ### detrend ###
    detrended_df = learn_past.trend_decompose(meanstddf[['mean']])
        # columns -> mean, trend, resid

    # cut waittime data
    if args.lower_date:
        meanstddf = meanstddf[args.lower_date:].copy()
        detrended_df = detrended_df[args.lower_date:].copy()
    
    if args.upper_date:
        meanstddf = meanstddf[:args.upper_date].copy()
        detrended_df = detrended_df[:args.upper_date].copy()


    ### load models ###
    modeldir = 'models'

    [resid_predict_model, features_list] = pickle.load(open(f'{modeldir}/resid_predict_models.pickle', 'rb'))


    ### predict resid ###
    explanatories_df = myfunc.make_dataset(meanstddf[[]], args.dataset)
    explanatories_df = explanatories_df[features_list]

    predicted_residdf = predict_future.long_term_resid(resid_predict_model, explanatories_df)



    ### evaluate ###
    df = meanstddf[['mean']].join([ predicted_residdf[['resid']], detrended_df[['trend']] ])
    df['pred'] = df['resid'] + df['trend']
    df = df[['mean', 'pred']]
    df = df.rename(columns={'mean': 'real'})
    # df.to_csv('evaluate.csv')

    df = df.dropna(how='any')
    print('MSE\nwaittime : %.3f' 
      % (mean_squared_error(df['real'].values, df['pred'].values))
     )
    print('RMSE\nwaittime : %.3f' 
        % (np.sqrt(mean_squared_error(df['real'].values, df['pred'].values)))
        )
    print('MAE\nwaittime : %.3f' 
        % (mean_absolute_error(df['real'].values, df['pred'].values))
        )
    print('R2\nwaittime : %.3f' 
        % (r2_score(df['real'].values, df['pred'].values))
        )



if __name__ == '__main__':
    main()