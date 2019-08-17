from wtpred import preprocess
from wtpred import base_predictors as bp

class model:

    def fit(self, explanatory_eachday_df, waittime_df):

        ### calucrate mean and standard-deviation each day ###
        meanstddf = preprocess.calucrate_mean_std_eachday(waittime_df)
            # columns -> [mean, std]


        self.models = {}

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


            ### save ###
            models = [features_list, resid_predict_model]
            self.models[mean_or_std] = models

