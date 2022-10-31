"""Some utility functions for project 1."""
import csv
import numpy as np
import random


def change_labels(y):
    """maps the classification labels from {-1,1} to {0,1}"""
    for i in range(len(y)):
        if y[i][0] == -1:
            y[i][0] = 0


def handle_nans(products):
    """shifts the -999 values to the mean of the other defined values for a same column"""
    # We compute the mean of all values except -999 for each column, and store that in a dictionary
    mean_of_columns = {}
    for i in range(products.shape[1]):
        nb_other_than_nan = sum([1 if x != -999 else 0 for x in products[:, i]])
        mean_of_columns[i] = (
            sum([x if x != -999 else 0 for x in products[:, i]]) / nb_other_than_nan
        )

    # we create a new matrix with all the -999 set to zero
    products_nan_to_mean = np.zeros(products.shape)
    for i in range(products.shape[0]):
        for j in range(products.shape[1]):
            if products[i, j] == -999:
                products_nan_to_mean[i, j] = mean_of_columns[j]
            else:
                products_nan_to_mean[i, j] = products[i, j]

    return products_nan_to_mean


def build_poly(tx, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
        Expand the data tx to the input degree.
        for each row of tx, the coefficients will be repeated to the power i,
        for each i in the range of the degree, the column of 1's will not be repeated.

    >>> build_poly(np.array([[1,1,2,3], [1,2,2,2]])
        [[ 1.  1.  2.  3.  1.  4.  9.  1.  8. 27.]
        [ 1.  2.  2.  2.  4.  4.  4.  8.  8.  8.]]


    """

    nb_coeff = (tx.shape[1] - 1) * degree + 1
    matrix = np.zeros((tx.shape[0], nb_coeff))
    N = tx.shape[0]
    # for each row
    for i in range(N):
        # set the first term to 1
        matrix[i][0] = 1
        # for each degree, add exponentiated coefficients to the row i
        for j in range(1, degree + 1):
            for l in range(1, tx.shape[1]):
                matrix[i][(j - 1) * (tx.shape[1] - 1) + l] = np.power(tx[i][l], j)

    return matrix


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def create_batch_size_1(y, tx, N):
    """
    Generate a batch of size 1, it is more efficient than the batch_iter method
    """
    rand_idx = int(random.random() * N)
    # take a y and put this value in a matrix
    y_stoch = y[rand_idx]

    y_stoch = np.expand_dims(y_stoch, axis=1)

    # take a random sample features in the appropriate matrix form
    tx_stoch = np.reshape(tx[rand_idx, :], (1, tx.shape[1]))

    return tx_stoch, y_stoch


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
    """return the kth fold separated data for cross validation
    Args:
        y:          shape=(N,)
        tx:         shape=(N,D)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)


    Returns:
        the separated data, train_tx, train_y, test_tx, test_y.


    """

    # separates data between train and test sets using k folds and k indices
    # the test samples are the kth group
    test_indices = k_indices[k]

    train_indices_not_flat = []

    # add all groups of indices in the list
    for i in range(len(k_indices)):
        if i != k:
            train_indices_not_flat.append(k_indices[i])

    # flatten the indices
    train_indices_flat = [e for sl in train_indices_not_flat for e in sl]

    # construct the matrices by selecting elements thanks to the different indices lists
    test_tx = np.array([tx[i] for i in test_indices])

    test_y = np.array([y[i] for i in test_indices])

    train_tx = np.array([tx[i] for i in train_indices_flat])

    train_y = np.array([y[i] for i in train_indices_flat])

    return train_tx, train_y, test_tx, test_y


def accuracy_logistic(tx, y, w):
    """Computes the accuracy of weights w given test predictions y and samples tx
    for a classification problem of y in {0,1}.

    Args:
        y:      shape=(N,)
        tx:     shape=(N,D)
        w:      shape=(D,)

    Returns:
        The accuracy a the number of correct predictions divided by the number
        of predictions
    """
    local_predictions = tx @ w

    # If a value x is negative, applying the sigmoid will give a probability
    # smaller than 0.5, hence a 0 prediction. Otherwise, it's a 1 prediction
    local_predictions = [0 if i < 0 else 1 for i in local_predictions.T[0]]

    N = len(y)
    # Get the accuracy
    accuracy = (N - sum(abs(local_predictions - y))) / N

    return accuracy
