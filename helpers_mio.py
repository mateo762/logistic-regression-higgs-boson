import numpy as np
import csv


def load_data():
    """Load data and convert it to the metric system."""
    file = open("data/sample-submission.csv")
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    rows = []
    for row in csvreader:
        rows.append(row)
    rows
    file.close()

    return rows
