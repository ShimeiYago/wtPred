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

from keras.models import model_from_json

import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('id', help='attracID') #args.id
    parser.add_argument('dataset', help='dataset dir path') #args.dataset
    parser.add_argument('-l', '--lower_date', help='lower date for predict') #args.lower_date
    parser.add_argument('-u', '--upper_date', help='upper date for predict') #args.upper_date
    args = parser.parse_args()

    print(f'--------------------------{args.id}------------------------------')


    ### which park ###
    if args.id[0] is 'L':
        park = 'land'
    else:
        park = 'sea'


    ### actual data ###
    # waittime data
    filepath = f'{args.dataset}/waittime/{park}.csv'
    waittimedf = pd.read_csv(filepath, index_col='datetime', parse_dates=['datetime'])[[args.id]]

    # meanstddf
    meanstddf = myfunc.calucrate_mean_std(waittimedf)


    ### explanatories ###
    explanatories_eachday_df = myfunc.make_dataset_eachday(meanstddf[[]], args.dataset)
    explanatories_eachtime_df = myfunc.make_dataset_eachtime(waittimedf[[]], args.dataset, args.id[0])


    ### predict ###
    ## mean
    predicted_meandf = predict_meanstd(meanstddf, 'mean', args.id, explanatories_eachday_df, lowerdate=args.lower_date, upperdate=args.upper_date, display_result=True)
        # columns = ['mean']

    ## std
    predicted_stddf = predict_meanstd(meanstddf, 'std', args.id, explanatories_eachday_df, lowerdate=args.lower_date, upperdate=args.upper_date, display_result=False)
        # columns = ['std']

    ## normal
    predicted_normaldf = predict_eachtime_value(args.id, explanatories_eachtime_df, lowerdate=args.lower_date, upperdate=args.upper_date)
        # columns = ['normal', 'normal_lower', 'normal_upper']


    ### final prediction result ###
    # mean + std
    pred_meanstddf = predicted_meandf.join(predicted_stddf, how='inner')

    df = predicted_normaldf.copy()
    # Multi Index
    df.set_index([df.index.date, df.index], inplace=True)
    df.index.names = ['date', 'datetime']

    # join
    df = df.join(pred_meanstddf, how='inner')

    #drop multiindex
    df.reset_index(level='date', inplace=True)
    df.drop(columns='date', inplace=True)

    # de normalize
    df['pred'] = df['normal'] * df['std'] + df['mean']
    df['pred_lower'] = df['normal_lower'] * df['std'] + df['mean']
    df['pred_upper'] = df['normal_upper'] * df['std'] + df['mean']
    preddf = df[['pred', 'pred_lower', 'pred_upper']].copy()


    ### evaluate ###
    # real data
    waittimedf.columns = ['real']

    # resultdf
    resultdf = waittimedf.join(preddf, how='inner')

    # drop NaN
    df = resultdf.dropna(how='any').copy()

    actual = df['real']
    pred = df['pred']

    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    print('-------------- final results ----------------')

    print( 'MSE : %.1f' %mse)
    print( 'RMSE : %.2f' %rmse)
    print( 'MAE : %.2f' %mae)
    print('R2 : %.3f' %r2)

    outdir = f'evaluate/{args.id}/02-final'
    save_result_plot_final(resultdf, outdir)




def predict_meanstd(valuedf, mean_or_std, attracID, explanatories_df, lowerdate=None, upperdate=None, display_result=True):
    valuedf = valuedf[[mean_or_std]].copy()
    valuedf.columns = ['real']

    modeldir = f'models/{attracID}/{mean_or_std}'
    outdir = f'evaluate/{attracID}/01-{mean_or_std}'


    ### detrend ###
    detrended_df = learn_past.trend_decompose(valuedf)
        # columns -> value, trend, resid


    ### cut waittime data ###
    if lowerdate:
        valuedf = valuedf[lowerdate:].copy()
    
    if upperdate:
        valuedf = valuedf[:upperdate].copy()


    ### predict resid ###
    ## load models
    [resid_predict_model, features_list] = pickle.load(open(f'{modeldir}/resid_predict_models.pickle', 'rb'))

    ## explanatories
    explanatories_df = myfunc.adapt_features(explanatories_df, features_list)

    ## predict
    predicted_residdf = predict_future.ranfore(resid_predict_model, explanatories_df)
    predicted_residdf.columns = ['resid']

    ## join and add
    df = valuedf[['real']].join([ predicted_residdf[['resid']], detrended_df[['trend']] ])
    df['pred'] = df['resid'] + df['trend']
    resultdf = df[['real', 'pred']]

    ## evaluate and plot
    if display_result:
        print(f'------------ {mean_or_std} -------------')

        df = resultdf.dropna(how='any')
        print('--- ranfore ---')
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
        
        # plot
        myfunc.save_result_plot(resultdf.copy(), outdir=f'{outdir}/ranfore')


    ### predict error ###

    ## load models
    [model_json_str, weights, error_mean, error_std] = pickle.load(open(f'{modeldir}/error_predict_models.pickle', 'rb'))
    error_predict_model = model_from_json(model_json_str) # model
    error_predict_model.set_weights(weights) # parameter


    ## caluclate error
    df = resultdf.copy()
    df['error'] = df['real'] - df['pred']


    ## make_dataset
    N_lookbucks = 6
    explanatory, datelist = [], []
    for i in range(len(df)-N_lookbucks):
        errors = list(df.iloc[i:i+N_lookbucks, :]['error'])
        date = df.index[i+N_lookbucks]

        # if there are NaN, skip
        if np.nan in errors:
            continue

        explanatory.append(errors)
        datelist.append(date) 

    # reshape
    X = np.array(explanatory).reshape(len(explanatory), N_lookbucks, 1)

    ## predict
    predicted_error = error_predict_model.predict(X).flatten()

    ## add column pred_error
    df['pred_error'] = 0
    for date, error in zip(datelist, predicted_error):
        df.loc[date, 'pred_error'] = error
    
    ## modify predict by pred_error
    df['pred'] = df['pred'] + df['pred_error']
    resultdf = df[['real', 'pred']].copy()


    ## evaluate and plot
    if display_result:
        df = resultdf.dropna(how='any')
        print('--- RNN ---')
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
        
        # plot
        myfunc.save_result_plot(resultdf.copy(), outdir=f'{outdir}/RNN')


    ## return predict
    df = resultdf[['pred']]
    df.columns = [mean_or_std]
    return df


def predict_eachtime_value(attracID, explanatories_df, lowerdate=None, upperdate=None):
    modeldir = f'models/{attracID}/time'

    ### cut waittime data ###
    if lowerdate:
        explanatories_df = explanatories_df[lowerdate:].copy()
    
    if upperdate:
        explanatories_df = explanatories_df[:upperdate].copy()


    ### predict value ###
    ## load models
    [normal_predict_model, features_list] = pickle.load(open(f'{modeldir}/normalvalue_predict_models.pickle', 'rb'))

    ## explanatories
    explanatories_df = myfunc.adapt_features(explanatories_df, features_list)

    ## predict
    predicted_df = predict_future.ranfore(normal_predict_model, explanatories_df)
    predicted_df.columns = ['normal']

    ## predict interval ##
    err_lower, err_upper = predict_future.ranfore_interval(normal_predict_model, explanatories_df.values, percentile=98)
    predicted_df['normal_lower'] = err_lower
    predicted_df['normal_upper'] = err_upper


    ### return pred ###
    return predicted_df



def save_result_plot_final(resultdf, outdir):

    os.makedirs(outdir, exist_ok=True)

    # x軸表示用の関数
    def timerange(start, end, interval_minute=60):
        start = datetime.datetime(1,1,1,start[0], start[1])
        end = datetime.datetime(1,1,1,end[0], end[1])
        
        li = []
        x = (end-start).seconds / 60
        for n in range(0, int(x), interval_minute):
                li.append((start + datetime.timedelta(minutes=n)).time())
        
        return li


    # plot用の関数
    def saveplot(date, df):
        title = str(date.year) + '_' + str(date.month) + '_' + str(date.day)
        
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(1, 1, 1)
        df.plot(ax=ax, style=['b', 'r', 'r--', 'r--'])
        # ax.set_xticks(timerange((8,15), (22,0)))
        ax.grid(which = "major", axis = "x", color = "gray", alpha = 0.8,
            linestyle = "-", linewidth = 1)
        
        plt.title(title)
        plt.xlabel(' ')
        plt.ylabel(' ')
        plt.rcParams["font.size"] = 20
        
        plt.xticks(rotation=70)
        
        ax.get_legend().remove()

        plt.savefig(f'{outdir}/{title}.png', bbox_inches='tight')
        plt.close()

    # plot waittime

    # Multi Index
    df = resultdf.copy()
    df.set_index([df.index.date, df.index.time], inplace=True)
    df.index.names = ['date', 'time']

    for name, group in df.groupby(level='date'):
        #drop multiindex
        group.reset_index(level='date', inplace=True)
        group.drop(columns='date', inplace=True)

        group.index = group.index.map(str)

        
        saveplot(name, group)

if __name__ == '__main__':
    main()