WtPred
====

WtPred is a model to predict future wait-time of Disney Parks attractions in Japan.

## Description

There are many attractions in Disney Parks in Japan, but you have to wait in line for a long time in order to ride popular attractions. WtPred can predit future congestion degree of the park and wait-time each time by learning past. The future prediction results is published at "[wtchart.com](https://wtchart.com)".

## Performance

## Requirement

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
