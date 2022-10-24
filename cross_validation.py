import numpy as np
from helpers import * 
import matplotlib.pyplot as plt
from implementations_tanguy import *

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.
    
    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """ 
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_least_squares(y, x, k_indices, k):
    """return the loss of ridge regression for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527597114, 0.33555914361295175)
    """
    test_indices = k_indices[k]

    train_indices_not_flat = []

    # add all groups of indices in the list
    for i in range(len(k_indices)):
        if i != k: 
            train_indices_not_flat.append(k_indices[i])
    #flatten the indices
    train_indices_flat = [e for sl in train_indices_not_flat for e in sl]

    test_x = [x[i] for i in test_indices]    

    test_y = [y[i] for i in test_indices]    

    train_x = [x[i] for i in train_indices_flat]

    train_y = [y[i] for i in train_indices_flat]

    train_tx = np.c_[np.ones((len(train_y), 1)), train_x]

    test_tx = np.c_[np.ones((len(test_y), 1)), test_x]


    w = least_squares(train_y, train_tx)[0]
    
    #rr_test = ridge_regression(test_y, poly_test, lambda_)


    loss_tr = compute_loss_mse(train_y, train_tx, w)
 
    loss_te = compute_loss_mse(test_y, test_tx, w)  

    loss_tr = np.sqrt(2*loss_tr)

    loss_te = np.sqrt(2*loss_te)

    return loss_tr, loss_te


def cross_validation_linear_gd(y, x, k_indices,k, max_iters, gamma):
    """return the loss of ridge regression for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527597114, 0.33555914361295175)
    """
    test_indices = k_indices[k]

    train_indices_not_flat = []

    # add all groups of indices in the list
    for i in range(len(k_indices)):
        if i != k: 
            train_indices_not_flat.append(k_indices[i])
    #flatten the indices
    train_indices_flat = [e for sl in train_indices_not_flat for e in sl]

    test_x = [x[i] for i in test_indices]    

    test_y = [y[i] for i in test_indices]    

    train_x = [x[i] for i in train_indices_flat]

    train_y = [y[i] for i in train_indices_flat]

    train_tx = np.c_[np.ones((len(train_y), 1)), train_x]

    test_tx = np.c_[np.ones((len(test_y), 1)), test_x]


    w = mean_squared_error_gd(train_y, train_tx, np.zeros((train_tx.shape[1], 1)), max_iters, gamma)[0]
    
    #rr_test = ridge_regression(test_y, poly_test, lambda_)


    loss_tr = compute_loss_mse(train_y, train_tx, w)
 
    loss_te = compute_loss_mse(test_y, test_tx, w)  

    loss_tr = np.sqrt(2*loss_tr)

    loss_te = np.sqrt(2*loss_te)

    return loss_tr, loss_te
