import pandas as pd
import numpy as np
import datetime

from wtpred import preprocess
from wtpred import wtround
from wtpred.base_predictors import fbprophet
from wtpred.base_predictors import randomforest
from wtpred.base_predictors import rnn

from keras.models import model_from_json

class model:

    def __init__(self):
        # placeholder of models
        self.basemodels = {'mean':{}, 'std':{}, 'normal':{}}


    def fit(self, explanatory_eachday_df, explanatory_eachtime_df, waittimedf):
        waittimedf.columns = ['waittime']


        ##### 01. learn mean and std each day #####

        ### calucrate mean and standard-deviation each day ###
        meanstddf = preprocess.calucrate_mean_std_eachday(waittimedf)
            # columns -> [mean, std]


        for mean_or_std in ['mean', 'std']:

            ### decompose target-value into trend and resid ###
            detrended_df = fbprophet.de_trend(meanstddf[[mean_or_std]])
                # columns -> mean, trend, resid


            ### join and make dataset ###
            datasetdf = detrended_df[['resid']].join(explanatory_eachday_df, how='inner')


            ### feature selection ###
            features_list = preprocess.feature_selection(datasetdf)
            datasetdf = datasetdf[ ['resid']+features_list ]
      

            ### learn resid ###
            resid_predict_model = randomforest.make_model(datasetdf)


            ### learn error ###
            error_predict_model = rnn.make_model(datasetdf)

        
            ### save models ###
            self.basemodels[mean_or_std]['features'] = features_list
            self.basemodels[mean_or_std]['resid_model'] = resid_predict_model
            self.basemodels[mean_or_std]['error_model'] = error_predict_model


        ##### 02. learn mean and std #####

        ### normalize ###
        # Multi Index
        waittimedf.set_index([waittimedf.index.date, waittimedf.index], inplace=True)
        waittimedf.index.names = ['date', 'datetime']

        df = waittimedf.join(meanstddf, how='inner')

        # drop multiindex
        df.reset_index(level='date', inplace=True)
        df.drop(columns='date', inplace=True)

        df['normal'] = (df['waittime'] - df['mean']) / df['std']


        ### join and make dataset ###
        datasetdf = df[['normal']].join(explanatory_eachtime_df, how='inner')


        ### feature selection ###
        features_list = preprocess.feature_selection(datasetdf)
        datasetdf = datasetdf[ ['normal']+features_list ]


        ### learn normal value ###
        normal_predict_model = randomforest.make_model(datasetdf)


        ### save model ###
        self.basemodels['normal']['features'] = features_list
        self.basemodels['normal']['model'] = normal_predict_model



        ##### 03. count unique waittimes #####
        self.basemodels['roundlist'] = wtround.make_roundlist(waittimedf[['waittime']])



    def predict(self, past_waittime_df, explanatory_eachday_df, explanatory_eachtime_df, todate_str, upperdate_str):

        lowerdate_str_with_lookbucks = ( datetime.datetime.strptime(todate_str, '%Y-%m-%d') - datetime.timedelta(days=rnn.N_lookbucks) ).strftime('%Y-%m-%d')


        ##### 01. predict mean and std #####

        past_meanstd_df = preprocess.calucrate_mean_std_eachday(past_waittime_df)

        predicted_meanstd_df = explanatory_eachday_df[[]][lowerdate_str_with_lookbucks:upperdate_str]
        for mean_or_std in ['mean', 'std']:

            ### predict trend ###
            predicted_trenddf = fbprophet.predict_trend(past_meanstd_df[[mean_or_std]], upperdate_str)
            predicted_trenddf = predicted_trenddf[lowerdate_str_with_lookbucks:upperdate_str]
        

            ### predict resid ###
            features_list = self.basemodels[mean_or_std]['features']
            explanatorydf = preprocess.adapt_features(explanatory_eachday_df[lowerdate_str_with_lookbucks:upperdate_str], features_list)

            resid_predict_model = self.basemodels[mean_or_std]['resid_model']
            predicted_residdf = randomforest.predict(resid_predict_model, explanatorydf)
            predicted_residdf.columns = ['resid']


            ### join ###
            df = predicted_trenddf.join(predicted_residdf, how='inner')


            ### trend + resid ###
            df['predicted_value'] = df['trend'] + df['resid']


            ### modify nextday error ###
            error_predict_model = self.basemodels[mean_or_std]['error_model']
            predicted_nextday_error = rnn.predict_nextday_error(error_predict_model, past_meanstd_df[[mean_or_std]], df[['predicted_value']], todate_str)
            df.loc[todate_str, 'predicted_value'] = df.loc[todate_str, 'predicted_value'] + predicted_nextday_error


            predicted_meanstd_df[mean_or_std] = df['predicted_value']


        predicted_meanstd_df = predicted_meanstd_df[todate_str:upperdate_str]



        ##### 02. predict normal-value #####

        # explanatory
        features_list = self.basemodels['normal']['features']
        explanatorydf = preprocess.adapt_features(explanatory_eachtime_df[todate_str:upperdate_str], features_list)

        # predict
        normal_predict_model = self.basemodels['normal']['model']
        predicted_normaldf = randomforest.predict(normal_predict_model, explanatorydf)
        predicted_normaldf.columns = ['normal']

        # predict interval
        err_lower, err_upper = randomforest.predict_interval(normal_predict_model, explanatorydf, percentile=98)
        predicted_normaldf['normal_lower'] = err_lower
        predicted_normaldf['normal_upper'] = err_upper



        ##### 03. join and de-normalize #####

        # Multi Index (normal)
        predicted_normaldf.set_index([predicted_normaldf.index.date, predicted_normaldf.index], inplace=True)
        predicted_normaldf.index.names = ['date', 'datetime']

        # join
        preddf = predicted_normaldf.join(predicted_meanstd_df, how='inner')

        # drop multiindex
        preddf.reset_index(level='date', inplace=True)
        preddf.drop(columns='date', inplace=True)

        # de-normalize
        preddf['value'] = preddf['normal'] * preddf['std'] + preddf['mean']
        preddf['value_lower'] = preddf['normal_lower'] * preddf['std'] + preddf['mean']
        preddf['value_upper'] = preddf['normal_upper'] * preddf['std'] + preddf['mean']
        preddf = preddf[['value', 'value_lower', 'value_upper']]



        ###### 04. round #####
        roundlist = self.basemodels['roundlist']
        preddf['value'] = wtround.round_waittime(list(preddf['value']), roundlist)
        preddf['value_lower'] = wtround.round_waittime(list(preddf['value_lower']), roundlist)
        preddf['value_upper'] = wtround.round_waittime(list(preddf['value_upper']), roundlist)


        return predicted_meanstd_df[['mean']], preddf



    def save(self):
        obj = {'mean':{}, 'std':{}, 'normal':{}}

        ### mean ###
        obj['mean']['features'] = self.basemodels['mean']['features']
        obj['mean']['resid_model'] = self.basemodels['mean']['resid_model']
        obj['mean']['error_kerasmodel_json'] = self.basemodels['mean']['error_model'].to_json()
        obj['mean']['error_kerasmodel_weights'] = self.basemodels['mean']['error_model'].get_weights()
    

        ### std ###
        obj['std']['features'] = self.basemodels['std']['features']
        obj['std']['resid_model'] = self.basemodels['std']['resid_model']
        obj['std']['error_kerasmodel_json'] = self.basemodels['std']['error_model'].to_json()
        obj['std']['error_kerasmodel_weights'] = self.basemodels['std']['error_model'].get_weights()


        ### normal ###
        obj['normal']['features'] = self.basemodels['normal']['features']
        obj['normal']['model'] = self.basemodels['normal']['model']


        ### roundlist ###
        obj['roundlist'] = self.basemodels['roundlist']
       

        return obj



    def load(self, obj):

        ### mean ###
        self.basemodels['mean']['features'] = obj['mean']['features']
        self.basemodels['mean']['resid_model'] = obj['mean']['resid_model']
        self.basemodels['mean']['error_model'] = model_from_json(obj['mean']['error_kerasmodel_json'])
        self.basemodels['mean']['error_model'].set_weights(obj['mean']['error_kerasmodel_weights'])
    

        ### std ###
        self.basemodels['std']['features'] = obj['std']['features']
        self.basemodels['std']['resid_model'] = obj['std']['resid_model']
        self.basemodels['std']['error_model'] = model_from_json(obj['std']['error_kerasmodel_json'])
        self.basemodels['std']['error_model'].set_weights(obj['std']['error_kerasmodel_weights'])
    

        ### normal ###
        self.basemodels['normal']['features'] = obj['normal']['features']
        self.basemodels['normal']['model'] = obj['normal']['model']


        ### roundlist ###
        self.basemodels['roundlist'] = obj['roundlist']
