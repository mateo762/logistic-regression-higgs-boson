import numpy as np
from helpers import *
import matplotlib.pyplot as plt
from implementations import *


def cross_validation_visualization(lambds, rmse_tr, rmse_te):
    """visualization the curves of rmse_tr and rmse_te."""
    plt.semilogx(lambds, rmse_tr, marker=".", color="b", label="train error")
    plt.semilogx(lambds, rmse_te, marker=".", color="r", label="test error")
    plt.xlabel("lambda")
    plt.ylabel("r mse")
    # plt.xlim(1e-4, 1)
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")


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
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def separate_data(tx, y, k_indices, k):
    # separates data between train and test sets using k folds and k indices
    test_indices = k_indices[k]

    train_indices_not_flat = []

    # add all groups of indices in the list
    for i in range(len(k_indices)):
        if i != k:
            train_indices_not_flat.append(k_indices[i])
    # flatten the indices
    train_indices_flat = [e for sl in train_indices_not_flat for e in sl]

    test_tx = np.array([tx[i] for i in test_indices])

    test_y = np.array([y[i] for i in test_indices])

    train_tx = np.array([tx[i] for i in train_indices_flat])

    train_y = np.array([y[i] for i in train_indices_flat])

    return train_tx, train_y, test_tx, test_y


def cross_validation_least_squares(y, tx, k_indices, k):
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
    train_tx, train_y, test_tx, test_y = separate_data(tx, y, k_indices, k)

    w, loss_tr = least_squares(train_y, train_tx)

    # rr_test = ridge_regression(test_y, poly_test, lambda_)

    loss_te = compute_loss_mse(test_y, test_tx, w)

    loss_tr = np.sqrt(2 * loss_tr)

    loss_te = np.sqrt(2 * loss_te)

    return loss_tr, loss_te


def cross_validation_linear_gd(y, tx, k_indices, k):
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
    train_tx, train_y, test_tx, test_y = separate_data(tx, y, k_indices, k)

    w, loss_tr = mean_squared_error_gd(
        train_y, train_tx, np.zeros((train_tx.shape[1], 1)), 10000, 0.001
    )

    # rr_test = ridge_regression(test_y, poly_test, lambda_)

    loss_te = compute_loss_mse(test_y, test_tx, w)

    loss_tr = np.sqrt(2 * loss_tr)

    loss_te = np.sqrt(2 * loss_te)

    return loss_tr, loss_te


def cross_validation_logistic_regression(y, tx, initial_w, k_indices, k):
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

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np
    .array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527597114, 0.33555914361295175)
    """
    train_tx, train_y, test_tx, test_y = separate_data(tx, y, k_indices, k)



    w, loss_tr = logistic_regression(
        train_y, train_tx, initial_w, 5000, 1e-6
    )

    # rr_test = ridge_regression(test_y, poly_test, lambda_)

    loss_te = compute_loss_mse(test_y, test_tx, w)

    loss_tr = np.sqrt(2 * loss_tr)

    loss_te = np.sqrt(2 * loss_te)

    return loss_tr, loss_te


def cross_validation_ridge_regression(y, tx, k_indices, k, lambda_):
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

    train_tx, train_y, test_tx, test_y = separate_data(tx, y, k_indices, k)

    w, loss_tr = ridge_regression(train_y, train_tx, lambda_)

    loss_te = compute_loss_mse(test_y, test_tx, w)

    loss_tr = np.sqrt(2 * loss_tr)

    loss_te = np.sqrt(2 * loss_te)

    return loss_tr, loss_te


def find_best_lambda_ridge_regression(tx, y, k_fold, lambdas):
    """cross validation over regularisation parameter lambda.

    Args:
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """
    seed = 12
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # set the best rmse to a maximal value as a start
    best_rmse = 101
    best_lambda = 0
    for l in lambdas:
        rmse_tr_temp = 0.0
        rmse_te_temp = 0.0
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation_ridge_regression(y, tx, k_indices, k, l)

            rmse_tr_temp += loss_tr
            rmse_te_temp += loss_te
        # do the average
        rmse_tr_temp = rmse_tr_temp / k_fold
        rmse_te_temp = rmse_te_temp / k_fold

        if rmse_te_temp < best_rmse:
            best_rmse = rmse_te_temp
            best_lambda = l
        rmse_tr.append(rmse_tr_temp)
        rmse_te.append(rmse_te_temp)

    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    print(
        " the choice of lambda which leads to the best test rmse is %.5f with a test rmse of %.3f"
        % (best_lambda, best_rmse)
    )
    return best_lambda, best_rmse


def cross_validation_reg_logistic_regression(y, tx, initial_w, k_indices, k, lambda_):
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
    train_tx, train_y, test_tx, test_y = separate_data(tx, y, k_indices, k)

    w, loss_tr = reg_logistic_regression(train_y, train_tx, lambda_, initial_w, 1000, 0.0001)

    # rr_test = ridge_regression(test_y, poly_test, lambda_)

    loss_te = calculate_logistic_loss(test_y, test_tx, w)# + lambda_ * np.dot(w.T, w)

    loss_tr = np.sqrt(2 * loss_tr)# + lambda_ * np.dot(w.T, w)

    loss_te = np.sqrt(2 * loss_te)# + lambda_ * np.dot(w.T, w)

    return loss_tr, loss_te


def find_best_lambda_reg_logistic_regression(tx, y, initial_w, k_fold, lambdas):
    """cross validation over regularisation parameter lambda.

    Args:
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """
    seed = 12
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # set the best rmse to a maximal value as a start
    best_rmse = 101
    best_lambda = 0
    for l in lambdas:
        rmse_tr_temp = 0.0
        rmse_te_temp = 0.0
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation_reg_logistic_regression( y, tx, initial_w, k_indices, k, l)

            rmse_tr_temp += loss_tr
            rmse_te_temp += loss_te
        # do the average
        rmse_tr_temp = rmse_tr_temp / k_fold
        rmse_te_temp = rmse_te_temp / k_fold

        if rmse_te_temp < best_rmse:
            best_rmse = rmse_te_temp
            best_lambda = l
        rmse_tr.append(rmse_tr_temp)
        rmse_te.append(rmse_te_temp)

    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    print(
        " the choice of lambda which leads to the best test rmse is %.9f with a test rmse of %.3f"
        % (best_lambda, best_rmse)
    )
    return best_lambda, best_rmse
