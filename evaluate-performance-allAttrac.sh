#!/usr/bin/bash

for i in $(/usr/bin/seq -w 0 37); do
    id=L$i
    echo ------------------------------$id--------------------------------
    python evaluate_performance.py $id dataset -l 2019

done

for i in $(/usr/bin/seq -w 0 32); do
    id=S$i
    echo ------------------------------$id--------------------------------
    python evaluate_performance.py $id dataset -l 2019

done