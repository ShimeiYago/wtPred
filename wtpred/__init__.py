from wtpred import preprocess
from wtpred import base_predictors as bp

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
            detrended_df = bp.de_trend(meanstddf[[mean_or_std]])
                # columns -> mean, trend, resid


            ### join and make dataset ###
            datasetdf = detrended_df[['resid']].join(explanatory_eachday_df, how='inner')


            ### feature selection ###
            features_list = preprocess.feature_selection(datasetdf)
            datasetdf = datasetdf[ ['resid']+features_list ]
      

            ### learn resid ###
            resid_predict_model = bp.randomforest_model(datasetdf)


            ### learn error ###
            error_predict_model = bp.error_predict_model(datasetdf)

        
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
        normal_predict_model = bp.randomforest_model(datasetdf)


        ### save model ###
        self.basemodels['normal']['features'] = features_list
        self.basemodels['normal']['model'] = normal_predict_model



    def predict(self, past_waittime_df, explanatory_eachday_df, explanatory_eachtime_df):

        lowerdate_str = explanatory_eachday_df.index[0].strftime('%Y-%m-%d')
        upperdate_str = explanatory_eachday_df.index[-1].strftime('%Y-%m-%d')

        past_meandf = preprocess.calucrate_mean_std_eachday(past_waittime_df)[['mean']]
        df = bp.predict_trend(past_meandf, lowerdate_str, upperdate_str)

        print(df)

    
        # ### predict ###
        # ## mean
        # predicted_meandf = predict_meanstd(meanstddf, 'mean', args.id, explanatories_eachday_df, lowerdate=args.lower_date, upperdate=args.upper_date, display_result=True)
        # predicted_meandf['mean_lower'] = predicted_meandf['mean'].values
        # predicted_meandf['mean_upper'] = predicted_meandf['mean'].values
        # # predicted_meandf['mean_lower'] = predicted_meandf['mean'] - 10
        # # predicted_meandf['mean_upper'] = predicted_meandf['mean'] + 10
        #     # columns = ['mean', 'mean_lower', 'mean_upper']

        # ## std
        # predicted_stddf = predict_meanstd(meanstddf, 'std', args.id, explanatories_eachday_df, lowerdate=args.lower_date, upperdate=args.upper_date, display_result=False)
        # predicted_stddf['std_lower'] = predicted_stddf['std'].values
        # predicted_stddf['std_upper'] = predicted_stddf['std'].values
        #     # columns = ['std', 'std_lower', 'std_upper']

        # ## normal
        # predicted_normaldf = predict_eachtime_value(args.id, explanatories_eachtime_df, lowerdate=args.lower_date, upperdate=args.upper_date)
        # # predicted_normaldf['normal_lower'] = predicted_normaldf['normal'].values
        # # predicted_normaldf['normal_upper'] = predicted_normaldf['normal'].values
        #     # columns = ['normal', 'normal_lower', 'normal_upper']



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
