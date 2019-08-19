import pandas as pd
import numpy as np
import datetime


def make_roundlist(df, n_min=2):
    counteddf = df.apply(pd.value_counts)

    roundlist = [i for i,v in counteddf.iloc[:, 0].iteritems() if v >= n_min]

    return roundlist



def round_waittime(rawlist, roundlist):
    processed_list = []
    for x in rawlist:
        if x < 0:
            processed_list.append(min(roundlist))
            continue
        
        
        nearest = take_nearest(x, roundlist)

        if np.abs(nearest-x) < 2.5:
            processed_list.append(nearest)
        
        else:
            processed_list.append( round_nearest_multiple(x) )

    return processed_list



### take out the nearest to value from list ###
def take_nearest(x, roundlist):
    idx = np.abs(np.asarray(roundlist) - x).argmin()
    return roundlist[idx]


### round to multiple of 5 #####
def round_nearest_multiple(x, base=5):
    return int(base * round(x/base))