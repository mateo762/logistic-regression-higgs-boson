import numpy as np
import random


""" LINEAR REGRESSION USING GRADIENT DESCENT"""
#############################################
def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1 / 2 * np.mean(e**2)


def compute_loss(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)


def compute_loss_mse(y, tx, w):

    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    N = len(y)
    error = y - (tx @ w)
    mse = (1 / (2 * N)) * (error.T @ error)

    return mse[0][0]


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """

    N = len(y)
    error_vector = y - tx @ w

    gradient = -(1 / N) * (tx.T @ error_vector)

    return gradient


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the final parameters a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        loss: the final loss
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    loss = 0.0

    for n_iter in range(max_iters):

        gradient = compute_gradient(y, tx, w)
        loss = compute_loss_mse(y, tx, w)

        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        # print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #     bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return (w, loss)


""" LINEAR REGRESSION USING STOCHASTIC GRADIENT DESCENT"""
#############################################


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


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


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    loss = 0.0
    for n_iter in range(max_iters):

        rand_idx = int(random.random() * N)
        # take a y and put this value in a matrix
        y_stoch = [y[rand_idx],]
        y_stoch = np.expand_dims(y_stoch, axis=1)

        # take a random sample features in the appropriate matrix form
        tx_stoch = np.reshape(tx[rand_idx, :], (1, tx.shape[1]))

        # compute a stochastic gradient and loss
        grad = compute_stoch_gradient(y_stoch, tx_stoch, w)
        # , _ = compute_stoch_gradient(y_stoch, tx_stoch, w)
        # update w through the stochastic gradient update
        w = w - gamma * grad
        # calculate loss
        loss = compute_loss_mse(y_stoch, tx_stoch, w)

        # print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        # bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return ws, loss


""" Least squares regression using normal equations"""
#############################################


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    gramMatrix = tx.T @ tx

    identity = np.identity(gramMatrix.shape[0])
    gram_inverse = np.linalg.solve(gramMatrix, identity)

    # inverse_g = np.linalg.inv(gramMatrix)
    w = gram_inverse @ (tx.T @ y)
    error = y - (tx @ w)
    N = len(y)

    loss = compute_loss_mse(y, tx, w)
    return (w, loss)


""" Ridge regression using normal equations"""
#############################################


def compute_loss_ridge(y, tx, w, lambda_):
    """Calculate the loss using ridge regression.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    loss = compute_loss_mse(y, tx, w) + lambda_ * (w.T @ w)
    return loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])

    """
    N = len(y)

    gram_m = tx.T @ tx

    lambda_identity = (lambda_ * 2 * N) * np.identity(gram_m.shape[0])

    w = (np.linalg.inv(gram_m + lambda_identity)) @ (tx.T @ y)

    loss = compute_loss_ridge(y, tx, w, lambda_)

    return (w, loss)


""" Logistic regression using gradient descent or SGD (y ∈ {0, 1}) """
#######################################################################


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    result = 1 / (1 + np.exp(-t))
    return result


def calculate_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(calculate_loss(y, tx, w), 8)
    1.52429481
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    N = len(y)

    result = 0.0
    for i in range(N):
        temp_prob = sigmoid(tx[i].T @ w)
        result += y[i] * np.log(temp_prob) + (1 - y[i]) * np.log(1 - temp_prob)
    loss = -(1 / N) * result[0]
    return loss


def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_gradient(y, tx, w)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """

    N = len(y)
    D = len(w)
    # We can find gradient with closed form
    gradient = (1 / N) * (tx.T @ (sigmoid(tx @ w) - y))

    return gradient


def calculate_logistic_hessian(y, tx, w):
    """return the Hessian of the loss function.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a hessian matrix of shape=(D, D)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_hessian(y, tx, w)
    array([[0.28961235, 0.3861498 , 0.48268724],
           [0.3861498 , 0.62182124, 0.85749269],
           [0.48268724, 0.85749269, 1.23229813]])
    """

    N = len(y)
    S = np.zeros((N, N))
    # we can find the second degree gradient for each entry of the hessian with closed form
    for i in range(N):
        temp = sigmoid(tx[i].T @ w)
        S[i][i] = temp * (1 - temp)

    hessian = (1 / N) * (tx.T @ (S @ tx))
    return hessian


def logistic_regression_loss_gradient_hessian(y, tx, w):
    """return the loss, gradient of the loss, and hessian of the loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)
        hessian: shape=(D, D)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> loss, gradient, hessian = logistic_regression(y, tx, w)
    >>> round(loss, 8)
    0.62137268
    >>> gradient, hessian
    (array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]]), array([[0.28961235, 0.3861498 , 0.48268724],
           [0.3861498 , 0.62182124, 0.85749269],
           [0.48268724, 0.85749269, 1.23229813]]))
    """

    loss = calculate_logistic_loss(y, tx, w)

    hessian = calculate_logistic_hessian(y, tx, w)

    gradient = calculate_logistic_gradient(y, tx, w)

    return loss, gradient, hessian


def learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step of Newton's method.
    Return the loss and updated w.
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar
    Returns:
        loss: scalar number
        w: shape=(D, 1)
    >>> y = np.c_[[0., 0., 1., 1.]]
    >>> np.random.seed(0)
    >>> tx = np.random.rand(4, 3)
    >>> w = np.array([[0.1], [0.5], [0.5]])
    >>> gamma = 0.1
    >>> loss, w = learning_by_newton_method(y, tx, w, gamma)
    >>> round(loss, 8)
    0.71692036
    >>> w
    array([[-1.31876014],
           [ 1.0590277 ],
           [ 0.80091466]])
    """

    loss, gradient, hessian = logistic_regression_loss_gradient_hessian(y, tx, w)

    identity = np.identity(hessian.shape[0])
    hessian_inverse = np.linalg.solve(hessian, identity)
    w = w - (gamma * (hessian_inverse @ gradient))
    return loss, w


def learning_by_logistic_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> gamma = 0.1
    >>> loss, w = learning_by_gradient_descent(y, tx, w, gamma)
    >>> round(loss, 8)
    0.62137268
    >>> w
    array([[0.11037076],
           [0.17932896],
           [0.24828716]])
    """

    loss = compute_loss_mse(y, tx, w)
    gradient = calculate_logistic_gradient(y, tx, w)

    new_w = w - gamma * gradient
    return loss, new_w


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    threshold = 1e-8
    losses = []
    w = initial_w
    loss= 0.0

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        # loss, w = learning_by_newton_method(y, tx, w, gamma)
        loss, w = learning_by_logistic_gradient_descent(y, tx, w, gamma)
        # log info
        # if iter % 1 == 0:
        #   print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))

        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    # visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_newton_method", True)
    # print("loss={l}".format(l=calculate_loss(y, tx, w)))
    return (w, loss)


def learning_by_penalized_logistic_gradient_descent(y, tx, w, lambda_, gamma):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        new_w: shape=(D, 1)

    """

    loss = compute_loss_mse(y, tx, w) + lambda_ * np.dot(w.T, w)
    gradient = calculate_logistic_gradient(y, tx, w) + 2 * lambda_ * w

    new_w = w - gamma * gradient
    return loss, new_w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Do steps of gradient descent until the difference between the losses is below threshold, using the penalized logistic regression.
    Return the final loss and w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        lambda_: scalar
        initial_w:  shape=(D, 1)
        max_iters: scalar
        gamma: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    """
    threshold = 1e-8
    losses = []
    w = initial_w
    loss= 0.0
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        # loss, w = learning_by_newton_method(y, tx, w, gamma)
        loss, w = learning_by_penalized_logistic_gradient_descent(
            y, tx, w, lambda_, gamma
        )
        # log info
        # if iter % 1 == 0:
        #   print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))

        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    # visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_newton_method", True)
    # print("loss={l}".format(l=calculate_loss(y, tx, w)))
    return (w, loss)
