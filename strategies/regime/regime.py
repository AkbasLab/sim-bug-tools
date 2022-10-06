import os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

import warnings


import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.structs as structs
import simulator

import pandas as pd


class RegimeSUMO:
    def __init__(self):
        warnings.simplefilter(action='ignore', category=FutureWarning)

        # Initialize the parameter manager
        self._parameter_manager = simulator.TrafficLightRaceParameterManager()


        # Use random sequence as a placeholdr
        seq = sequences.RandomSequence(
            self.parameter_manager.domain,
            self.parameter_manager.axes_names
        )
        seq.seed = 222
        for i in range(10):
            seq.get_points(1)
        self.global_exploration(seq)

        return

    def __PARAMETERS__(self):
        return

    @property
    def parameter_manager(self) -> simulator.TrafficLightRaceParameterManager:
        return self._parameter_manager


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

        return params_s, test.scores


    def __GLOBAL_EXPLORATION__(self):
        return

    def is_in_target_score_specs(self, score : pd.Series):
        """
        Checks if a test score is in the Target Score Specs
        TODO: make this function overloadable.
        """
        return score["e_brake"] > 0 and score["e_brake"] < .5

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
            if self.is_in_target_score_specs(scores):
                break
        print("GLOBAL EXPLORATION END.")
        return


