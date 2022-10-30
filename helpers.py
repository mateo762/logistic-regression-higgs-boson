"""Some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def sample_data(y, x, seed, size_samples):
    """sample from dataset."""
    np.random.seed(seed)
    num_observations = y.shape[0]
    random_permuted_indices = np.random.permutation(num_observations)
    y = y[random_permuted_indices]
    x = x[random_permuted_indices]
    return y[:size_samples], x[:size_samples]


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(x, y):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


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
    y_stoch = [
        y[rand_idx],
    ]
    y_stoch = np.expand_dims(y_stoch, axis=1)

    # take a random sample features in the appropriate matrix form
    tx_stoch = np.reshape(tx[rand_idx, :], (1, tx.shape[1]))

    return tx_stoch, y_stoch
