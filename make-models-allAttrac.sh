#!/usr/bin/bash

for i in $(/usr/bin/seq -w 0 37); do
    id=L$i
    echo ------------------------------$id--------------------------------
    python make_models.py $id dataset -u 2018 -s >& log.txt

done

for i in $(/usr/bin/seq -w 0 32); do
    id=S$i
    echo ------------------------------$id--------------------------------
    python make_models.py $id dataset -u 2018 -s >& log.txt

done