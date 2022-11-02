import pandas as pd
import numpy as np
import sklearn.model_selection
import shap
import catboost

class FeatureImpactHelper():
    def __init__(self, 
        data_df : pd.DataFrame, 
        scores : pd.Series, 
        test_size : float
    ):
        """
        This helper object performs shap on a dataset, and extends the 
        functionality of shap_values to evaluate feature impact.

        -- Parameters --
        data_df : pd.DataFrame
            Dataset of input parameter configurations.
        scores : pd.Series
            Categorization of each item in @data_df
        test_size : float
            % of test data used for the evaluation portion of the shap model
        """
        assert len(scores) == len(data_df.index)
        assert test_size > 0 and test_size < 1

        # Split dataset
        self._X_train, self._X_test, self._y_train, self._y_test = \
            sklearn.model_selection.train_test_split(
                data_df,
                scores,
                test_size=test_size,
                # random_state=1
            )
        
        # Shap values
        self._shap_values = self._init_shap_values()        

        # Select Feature Impact
        self._feature_impact = self._init_feature_impact()    
        return

    

    @property
    def shap_values(self) -> np.ndarray:
        """
        Shap values.
        Index -1 is the expected value.
        """
        return self._shap_values

    @property
    def X_train(self) -> pd.DataFrame:
        return self._X_train

    @property
    def feature_impact(self) -> pd.DataFrame:
        """
        Dataframe with feature impact information for all feature
        including impact weight and cumulative impact in [0,1]
        """
        return self._feature_impact



    def _init_shap_values(self) -> np.ndarray:
        """
        Construct, fit, and evaluate shap values using a catboost model

        Returns a shap values as a numpy array.
        """
        # Identify Categorical Features
        categorical_features = np.where(self._X_train.dtypes != float)[0]

        # Instantiate and fit the cat model.
        kwargs = {
            'iterations': 5000,
            'learning_rate': 0.01,
            'cat_features': categorical_features,
            'depth': 6, # 4-10 is reccomend
            'eval_metric': 'AUC',
            'verbose': 200,
            'od_type': "Iter",  # overfit detector
            'od_wait': 100,  # most recent best iteration to wait before stopping
            'random_seed': 1
        }

        cat_model = catboost.CatBoostClassifier(**kwargs)

        cat_model.fit(self._X_train, self._y_train,
              eval_set=(self._X_test, self._y_test),
              # True if we don't want to save trees
              # created after iteration with the best validation score
              use_best_model=True,
              plot=True
              )

        # Retrieve shap values
        shap_values = cat_model.get_feature_importance(
            catboost.Pool(
                self._X_test,
                label=self._y_test,
                cat_features=categorical_features
            ),
            type="ShapValues"
        )

        return shap_values

    
    def _init_feature_impact(self) -> pd.DataFrame:
        """
        Generates the feature impact dataframe 
        which includes weight of impact and cumulative impact in [0,1]
        """
        # Summarize impact
        impact = np.abs(self.shap_values[:, :-1]).sum(0)
        impact = impact/impact.sum()


        # Combine into a dataframe
        df = pd.DataFrame({
            "feature" : self.X_train.columns,
            "impact" : impact
        }) 
        
        # Sort by impact
        df.sort_values(
            by=['impact'], ascending=False, inplace=True)

        
        # Find the cumulative value
        df = df.assign(cumulative = [df["impact"].iloc[:i].sum() \
            for i in range(1,len(df.index)+1)])
        
        return df


    def summary_plot(self, plot_type : str):
        """
        Uses the shap module to generate dot, bar, or violin @plot_type
        """
        plot_type = plot_type.lower()
        assert plot_type in ["dot", "bar", "violin"]

        expected_value = self.shap_values[0, -1]
        shap_values = self.shap_values[:, :-1]

        shap.summary_plot(
            shap_values,
            self._X_test,
            show=False,
            plot_size=(20, 20),
            max_display=len(shap_values[0, :]),
            plot_type=plot_type
        )
        return