# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy import stats

from implementations_tanguy import *

from helpers import *

# load data.
particle, products, ids = load_csv_data('data/train.csv', sub_sample= True)

# build sampled x and y.
seed = 1
y = np.expand_dims(particle, axis=1)
y, X = sample_data(y, products, seed, size_samples=1000)
x, mean_x, std_x = standardize(X)

tx = np.c_[np.ones((y.shape[0], 1)), x]

print(y.shape, x.shape)
np.seterr(divide = 'ignore') 
initial_w = np.zeros((tx.shape[1], 1))


start_time = time.time()
w_gd, loss_gd = mean_squared_error_gd(y, tx, initial_w, 10, 0.001)
print("loss:", np.sqrt(2*loss_gd[0][0]), "--- %s seconds ---" % (time.time() - start_time))