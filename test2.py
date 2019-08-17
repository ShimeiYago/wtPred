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


    
    ### explanatory data ###
    lowerdate_str = args.lower_date
    upperdate_str = args.upper_date

    explanatory_eachday_df = make_explanatory.eachday(lowerdate_str, upperdate_str, args.dataset)
    explanatory_eachtime_df = make_explanatory.eachtime(lowerdate_str, upperdate_str, args.id[0], args.dataset)
    

    ### load models ###
    loaded_obj = pickle.load(open('wtpred-model.pickle', 'rb'))

    model = wtpred.model()
    model.load(loaded_obj)


    model.predict(waittimedf, explanatory_eachday_df, explanatory_eachtime_df)


if __name__ == '__main__':
    main()