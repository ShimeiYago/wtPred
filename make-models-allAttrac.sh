#!/usr/bin/bash

# for i in $(/usr/bin/seq -w 0 37); do
#     id=L$i
#     echo ------------------------------$id--------------------------------
#     python make-models.py $id dataset -u 2018 >& log.txt

# done

for i in $(/usr/bin/seq -w 0 32); do
    id=S$i
    echo ------------------------------$id--------------------------------
    python make_models_by_wtpred.py $id dataset -u 2018 >& log.txt

done