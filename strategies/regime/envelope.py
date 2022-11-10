
import sim_bug_tools.structs as structs

import pandas as pd

class Envelope(structs.Domain):
    def __init__(self, config_df : pd.DataFrame):
        """
        An envelope is a searchable subset of the testes and untested parameter
        space. 
        """
        return