"""
Derived from Wang et al. "Safety Performance Boundary Identification of Highly
Automated Vehicles: A Surrogate Model-Based Gradient Descent Searching Approach" 
"""

import numpy as np
import tensorflow as tf

from sim_bug_tools.experiment import Experiment, ExperimentParams

# tf.Variable(np.random.normal(size=))


class TrainGD(Experiment):
    def __init__(self):
        super().__init__("Training Gradient Descent Model", "model", ".out")
