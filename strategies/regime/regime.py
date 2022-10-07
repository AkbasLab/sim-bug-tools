import os
from typing import Callable
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

import warnings


import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.structs as structs
import sim_bug_tools.exploration.brrt_v2.adherer as adherer
import simulator

import pandas as pd


class RegimeSUMO:
    def __init__(self, target_score_classifier : Callable[[pd.Series], bool]):
        # Hide warning for appending pd.Series
        warnings.simplefilter(action='ignore', category=FutureWarning)

        # Initialize the parameter manager
        self._parameter_manager = simulator.TrafficLightRaceParameterManager()

        # Is in target score classification method
        self._target_score_classifier = target_score_classifier
        
        # 
        self._params_df = None
        self._scores_df = None
        return

    def __PARAMETERS__(self):
        return

    @property
    def parameter_manager(self) -> simulator.TrafficLightRaceParameterManager:
        return self._parameter_manager

    @property
    def target_score_classifier(self) -> Callable[[pd.Series], bool]:
        return self._target_score_classifier

    @property
    def params_df(self) -> pd.DataFrame:
        return self._params_df
    
    @property
    def scores_df(self) -> pd.DataFrame:
        return self._scores_df


    def __PRIVATE_METHODS__(self):
        return

    def _flatten_tl_params_df(self, df : pd.DataFrame) -> pd.Series:
        """
        Flattens a TL parameters @df into a series.
        """
        data = {}
        for i in range(len(df.index)):
            data["TL_%s" % df["state"].iloc[i]] = df["dur"].iloc[i]
        return pd.Series(data)

    def _flatten_veh_params_df(self, df : pd.DataFrame) -> pd.Series:
        """
        Flattens a vehicle parameters @df into a series.
        """
        features = df.columns.to_list()[:-1]
        data = {}
        for i in range(len(df.index)):
            vid = df["veh_id"].iloc[i]
            for feat in features:
                data["AV%s_%s" % (vid, feat)] = df[feat].iloc[i]
            continue
        return pd.Series(data)

    def _log_params_and_scores(self, params : pd.Series, scores : pd.Series):
        
        return


    def __PUBLIC_METHODS__(self):
        return

    def run_test(self, params : dict) -> list[pd.Series, pd.Series]:
        """
        Runs a simulation test using with concrete values in @params.
        Returns parameters and test scores as two pandas Series'
        """

        # Simulation Test
        veh_params_df = params["veh"]["concrete"]
        tl_params_df = params["tl"]["concrete"]
        test = simulator.TrafficLightRaceTest(veh_params_df, tl_params_df)

        # Format the parameters dictionary into a pandas series
        flat_veh_s = self._flatten_veh_params_df(veh_params_df)
        flat_tl_s = self._flatten_tl_params_df(tl_params_df)
        params_s = flat_veh_s.append(flat_tl_s)

        # Log the score data
        try:
            self._params_df = self.params_df.append(
                params_s, ignore_index = True)
            self._scores_df = self.scores_df.append(
                test.scores, ignore_index = True)
        except AttributeError:
            # Dataframes are None since no tests have been performed,
            # They are initialized with this first test.
            self._params_df = pd.DataFrame([params_s])
            self._scores_df = pd.DataFrame([test.scores])

        return params_s, test.scores


    def __GLOBAL_EXPLORATION__(self):
        return

    def global_exploration(self, seq : sequences.Sequence):
        """
        Global Exploration module.
        This module ends when a test score is in the Target Score Specs.
        """
        print("GLOBAL EXPLORATION START.")

        # Global exploration loop.
        while True:
            point = seq.get_points(1)[0]
            params = self.parameter_manager.map_parameters(point)
            params, scores = self.run_test(params)

            # Exit Condition
            if self.target_score_classifier(scores):
                break
        print("GLOBAL EXPLORATION END.")
        return


    def __BOUNDARY_DETECTION__(self):
        return

    def boundary_detection(self):
        print("BOUNDARY DETECTION START.")
        d = 0.005
        r = 2
        angle = adherer.ANGLE_90
        num = 4

        adhf = adherer.BoundaryAdherenceFactory(
            self.target_score_classifier, d, angle, r, num
        )
        print("BOUNDARY DETECTION END.")
        return 


    def __LOCAL_SENSITIVITY_REDUCTION__(self):
        return

    def local_sensitivity_reduction(self):
        print("LOCAL SENSITIVITY REDUCTION START.")

        

        print("LOCAL SENSITIVITY REDUCTION END.")
        return


    def __LOCAL_EXPLOITATION__(self):
        return

    def local_exploitation(self):
        print("LOCAL EXPLOITATION START.")

        print("LOCAL EXPLOITATION END.")
        return