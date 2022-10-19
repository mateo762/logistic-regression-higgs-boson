def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1 / 2 * np.mean(e ** 2)


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

def compute_loss_ridge(y, tx, w,lambda_):
    """Calculate the loss using ridge regression.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)+lambda_*np.dot(w.T,w)

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using Gradient Descent (GD).

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: a scalar denoting the model parameters as numpy arrays of shape (2), for the last iteration of GD
        loss: a scalar denoting the loss value for the last iteration of GD
    """
    # Define parameters
    ws = [initial_w]
    
    for n_iter in range(max_iters):
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # update w by gradient descent
        w = w - gamma * grad

        print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: a scalar denoting the model parameters as numpy arrays of shape (2), for the last iteration of SGD
        loss: a scalar denoting the loss value for the last iteration of SGD
    """

    # Define parameters
    w = initial_w

    for n_iter in range(max_iters):
        
        rand_idx = int(random.random() * len(y))
        y_stoch=y[rand_idx]
        tx_stoch=tx[rand_idx,:]
        # compute a stochastic gradient and loss
        grad, _ = compute_stoch_gradient(y_stoch, tx_stoch, w)
        # update w through the stochastic gradient update
        w = w - gamma * grad
        # calculate loss
        loss = compute_loss(y_stoch, tx_stoch, w)
    

        print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w,loss



"""
Least Square
"""

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
    D = tx.shape[1] #returns number of columns
    N = tx.shape[0]
    c = 2*N
    w = np.zeros(D)
    
    w = np.linalg.solve(np.dot(tx.T,tx),np.dot(tx.T,y))
    err = y-np.dot(tx,w)
    loss = 1/c*np.dot(err.T,err)
    
    return w,loss


"""
Ridge Regression
"""
def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    N=len(y)
    D=tx.shape[1]
    lambda_1=2*lambda_*N
    w = np.linalg.solve(np.dot(tx.T,tx)+lambda_1*np.eye(D),np.dot(tx.T,y))
    loss = compute_loss_ridge(y, tx, w,lambda_)
    
    
    return w, loss


