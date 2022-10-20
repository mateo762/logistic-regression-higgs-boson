import numpy as np


def load_data():
    """Load data and convert it to the metric system."""
    path_dataset = "data/sample-submission.csv"
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1, 2])
    id_ = data[:, 0]
    prediction = data[:, 1]
   
    return id_, prediction

