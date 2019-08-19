import pandas as pd
import numpy as np
import datetime
import argparse
import os
import pickle
import wtpred
import make_explanatory

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

    # cut waittime data
    if args.lower_date:
        waittimedf = waittimedf[args.lower_date:].copy()
    
    if args.upper_date:
        waittimedf = waittimedf[:args.upper_date].copy()

    
    ### explanatory data ###
    lowerdate_str = waittimedf.index[0].strftime('%Y-%m-%d')
    upperdate_str = waittimedf.index[-1].strftime('%Y-%m-%d')

    explanatory_eachday_df = make_explanatory.eachday(lowerdate_str, upperdate_str, args.dataset)
    explanatory_eachtime_df = make_explanatory.eachtime(lowerdate_str, upperdate_str, args.id[0], args.dataset)
    

    ### learning ###
    model = wtpred.model()
    model.fit(explanatory_eachday_df, explanatory_eachtime_df, waittimedf)

    saveobj = model.save()
    pickle.dump(saveobj, open(f'models/{args.id}.pickle', 'wb'))


if __name__ == '__main__':
    main()