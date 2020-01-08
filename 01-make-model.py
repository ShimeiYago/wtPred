#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    parser.add_argument('--id', help='attracID', default='L00')
    parser.add_argument('-d', '--datasetdir', help='dataset dir path', default='test-datasets')
    parser.add_argument('-l', '--lower_date', help='lower date of dataset for learning', default='2012-1-1')
    parser.add_argument('-u', '--upper_date', help='upper date of dataset for learning', default='2014-12-31')
    args = parser.parse_args()


    ### 01. prepare waittime data ###
    # which park
    if args.id[0] is 'L':
        park = 'land'
    else:
        park = 'sea'

    # get waittime df
    filepath = f'{args.datasetdir}/waittime/{park}.csv'
    waittimedf = pd.read_csv(filepath, index_col='datetime', parse_dates=['datetime'])[[args.id]]

    # cut waittime df
    if args.lower_date:
        waittimedf = waittimedf[args.lower_date:].copy()
    if args.upper_date:
        waittimedf = waittimedf[:args.upper_date].copy()

    
    ### 02. prepare explanatory df ###
    lowerdate_str = waittimedf.index[0].strftime('%Y-%m-%d')
    upperdate_str = waittimedf.index[-1].strftime('%Y-%m-%d')

    explanatory_eachday_df = make_explanatory.eachday(lowerdate_str, upperdate_str, args.datasetdir)
    explanatory_eachtime_df = make_explanatory.eachtime(lowerdate_str, upperdate_str, args.id[0], args.datasetdir)
    

    ### 03. learn model ###
    # learn and save
    # model = wtpred.model()
    # model.fit(explanatory_eachday_df, explanatory_eachtime_df, waittimedf)
    # saveobj = model.save()

    # # output
    outdir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models')
    os.makedirs(outdir, exist_ok=True)

    outpath = os.path.join(outdir, f'{args.id}.pickle')
    # pickle.dump(saveobj, open(outpath, 'wb'))
    print(f"the model was saved to {outpath}")


if __name__ == '__main__':
    main()