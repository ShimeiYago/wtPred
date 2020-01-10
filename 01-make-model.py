#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import datetime
import argparse
import os
import pickle
import wtpred
from wtpred import make_explanatory


# output dir
OUTDIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'outputs/01-saved-models')
os.makedirs(OUTDIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', help='attracID', default='L00')
    parser.add_argument('-d', '--datasetdir', help='dataset dir path', default='example-datasets')
    parser.add_argument('-l', '--lower_date', help='lower date of dataset for learning (Y-M-D)', default='2012-1-1')
    parser.add_argument('-u', '--upper_date', help='upper date of dataset for learning (Y-M-D)', default='2014-12-31')
    args = parser.parse_args()

    # check date format
    datetime.datetime.strptime(args.lower_date, '%Y-%m-%d')
    datetime.datetime.strptime(args.upper_date, '%Y-%m-%d')

    # dataset dir
    datasetdir = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.datasetdir)



    ### 01. prepare waittime data ###
    # which park
    if args.id[0] is 'L':
        park = 'land'
    else:
        park = 'sea'

    # get waittime df
    filepath = os.path.join(datasetdir, 'waittime', f'{park}.csv')
    waittimedf = pd.read_csv(filepath, index_col='datetime', parse_dates=['datetime'])[[args.id]][args.lower_date:args.upper_date]



    ### 02. prepare explanatory df ###
    lowerdate_str = waittimedf.index[0].strftime('%Y-%m-%d')
    upperdate_str = waittimedf.index[-1].strftime('%Y-%m-%d')

    explanatory_eachday_df = make_explanatory.eachday(lowerdate_str, upperdate_str, datasetdir)
    explanatory_eachtime_df = make_explanatory.eachtime(lowerdate_str, upperdate_str, args.id[0], datasetdir)
    


    ### 03. learn model ###
    # learn and save
    model = wtpred.model()
    model.fit(explanatory_eachday_df, explanatory_eachtime_df, waittimedf)
    saveobj = model.save()

    # # output
    outpath = os.path.join(OUTDIR, f'{args.id}.pickle')
    pickle.dump(saveobj, open(outpath, 'wb'))
    print(f'the model was saved to "{outpath}"')



if __name__ == '__main__':
    main()