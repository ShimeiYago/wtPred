WtPred
====

WtPred is a model to predict future wait-time of Disney Parks attractions in Japan.

## Description

There are many attractions in Disney Parks in Japan, but you have to wait in line for a long time in order to ride popular attractions. WtPred can predit future congestion degree of the park and wait-time each time by learning past. The future prediction results is published at "[wtchart.com](https://wtchart.com)".

## Performance
I will show prediction performance about a attraction, named "Big Thunder Mountain".

|  learning wait-time data  |  test wait-time data  |
----|---- 
|  from 2012 to 2017 |  2018 |


### prediction of congestion degree each day ###
Congestion degree is simple mean of wait-time every day.

<img src="https://raw.githubusercontent.com/ShimeiYago/wtPred/images/daily-L10-orig.png" alt="L10-eachday" width="800px">

### prediction of wait-time each time ###
I'll show only some results.

<img src="https://raw.githubusercontent.com/ShimeiYago/wtPred/images/L10.png" alt="L10-eachtime" width="800px">

## Requirement
### conda ###
```
conda create -n wtpred --file 
```

## Example
I prepared `example-datasets/`.
### 01. learning
Make a model by the data from 2012 to 2014.
```
./01-make-model.py
```
The model will be saved to `outputs/01-saved-models/`.

### 02. prediction
Predict 2015 wait-time by the model.
```
./02-predict.py
```
The predicted congestion degree and wait-time will be saved to `outputs/02-predict/`.

### 03. evaluation
Evaluate predicted wait-time by actual data.
```
./03-evaluate.py
```

## Author
[yago](https://github.com/ShimeiYago)
