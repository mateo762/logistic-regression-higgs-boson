
# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from implementations import *

from cross_validation import *

from helpers import *

from utils import *

# load data.
particle, products, ids = load_csv_data('data/train.csv', sub_sample= True)

#remove some features among highly correlated groups
axis = 1 #rows (0), columns (1)
obj = [4,5,6,9,21,24,25,26,27,28,29,17,18]
products_reduced = np.delete(products,obj,axis)

#sample data
seed = 1
y = np.expand_dims(particle, axis=1)
y, x = sample_data(y, products_reduced, seed, size_samples=1000)

# convert the labels from {-1,1}  to {0,1}
change_labels(y)


# we handle the NaN(-999) in the data
x = handle_nans(x)

#standardize
x, mean_x, std_x = standardize(x)

tx = np.c_[np.ones((y.shape[0], 1)), x]


#perform the regression
degree = 3
lambda_ = 0.000081113
initial_w = np.zeros(((tx.shape[1] - 1) * degree + 1, 1))
#create expanded polynomial
exp_tx_reduced = build_poly(tx, degree)
(w, loss) = reg_logistic_regression(y, exp_tx_reduced, lambda_, initial_w, 1000, 0.02)


#test the results
test_y, test_x, ids_test = load_csv_data('data/test.csv', sub_sample= False)
nb_test_samples = test_y.shape[0]

#remove some features among highly correlated groups
axis = 1 #rows (0), columns (1)
obj = [4,5,6,9,21,24,25,26,27,28,29,17,18]
test_x = np.delete(test_x,obj,axis)


test_y = np.expand_dims(test_y, axis=1)

# convert the labels accordingly
change_labels(test_y)


# we handle the NaN(-999) in the data
test_x = handle_nans(test_x)


#standardize
test_x, _, _ = standardize(test_x)

#add 1 column
test_tx = np.c_[np.ones((test_y.shape[0], 1)), test_x]

#build polynomials
test_tx_exp = build_poly(test_tx, degree)

#do the predictions
test_predictions_probs = test_tx_exp @ w
test_predictions_rounded = [-1 if i < 0 else 1 for i in test_predictions_probs.T[0]]  

create_csv_submission(ids_test, test_predictions_rounded, "data/submission.csv")