import numpy as np
from helpers import *
from utils import *
import matplotlib.pyplot as plt
from implementations import *


def cross_validation_visualization(lambds, rmse_tr, rmse_te):
    """visualization the curves of rmse_tr and rmse_te in function of the lamnda."""
    plt.semilogx(lambds, rmse_tr, marker=".", color="b", label="train error")
    plt.semilogx(lambds, rmse_te, marker=".", color="r", label="test error")
    plt.xlabel("lambda")
    plt.ylabel("r mse")
    # plt.xlim(1e-4, 1)
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")


def cross_validation_visualization_gamma(lambds, rmse_tr, rmse_te):
    """visualization the curves of rmse_tr and rmse_te in fucnction of the gamma."""
    plt.semilogx(lambds, rmse_tr, marker=".", color="b", label="train error")
    plt.semilogx(lambds, rmse_te, marker=".", color="r", label="test error")
    plt.xlabel("gamma")
    plt.ylabel("r mse")
    # plt.xlim(1e-4, 1)
    plt.title("cross validation for gamma")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")






def cross_validation_least_squares(y, tx, k_indices, k):
    """return the losses of least squares for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        tx:         shape=(N,D)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)


    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    """
    # separate data
    train_tx, train_y, test_tx, test_y = separate_data(tx, y, k_indices, k)

    # compute the weights and train loss
    w, loss_tr = least_squares(train_y, train_tx)

    # compute the test loss
    loss_te = compute_loss_mse(test_y, test_tx, w)
    # rmse
    loss_tr = np.sqrt(2 * loss_tr)
    loss_te = np.sqrt(2 * loss_te)

    return loss_tr, loss_te


def cross_validation_linear_gd(y, tx, k_indices, k):
    """return the losses of linear gd for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        tx:         shape=(N,D)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)


    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    """
    # separate data
    train_tx, train_y, test_tx, test_y = separate_data(tx, y, k_indices, k)

    # compute the weights and train loss
    w, loss_tr = mean_squared_error_gd(
        train_y, train_tx, np.zeros((train_tx.shape[1], 1)), 10000, 0.001
    )

    # compute the test loss
    loss_te = compute_loss_mse(test_y, test_tx, w)

    # rmse
    loss_tr = np.sqrt(2 * loss_tr)
    loss_te = np.sqrt(2 * loss_te)

    return loss_tr, loss_te


def cross_validation_logistic_regression(y, tx, initial_w, k_indices, k):
    """return the losses of logistic_regression gd for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        tx:         shape=(N,D)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)


    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    """
    accuracies = []
    losses = []

    for i in range(10):
        # separate data
        train_tx, train_y, test_tx, test_y = separate_data(tx, y, k_indices, k)
        w, loss_tr = logistic_regression(train_y, train_tx, initial_w, 1000, 0.02)
        loss_te = calculate_logistic_loss(test_y, test_tx, w)
        accuracies.append(accuracy_logistic(test_tx, test_y[:, 0], w))
        losses.append(loss_te)

    print("Accuracy, ", np.mean(accuracies), " +/- ", np.std(accuracies))
    print("loss: ", np.mean(losses), " +/- ", np.std(losses))

    # loss_tr = np.sqrt(2 * loss_tr)

    # loss_te = np.sqrt(2 * loss_te)
    # print(test_y)

    return loss_tr, loss_te


def cross_validation_logistic_regression_full(y, tx, initial_w, k_fold):
    """return the average test rmse and average accuracy for a full cross validation on k_fold
        prints the average accuracy and the standard deviation of accuracy
    Args:
        y:          shape=(N,)
        tx:          shape=(N,D)
        initial_w:  shape=(D, 1)
        k_fold: integer, the number of folds

    Returns:
        test rmse and average accuracy.

    """
    accuracies = []
    losses = []
    # build the indices
    k_indices = build_k_indices(y, k_fold, 1)

    # we perfom a logistic regression for each fold
    for k in range(k_fold):

        # separate the data
        train_tx, train_y, test_tx, test_y = separate_data(tx, y, k_indices, k)

        # compute train weight and train loss
        w, loss_tr = logistic_regression(train_y, train_tx, initial_w, 1000, 0.02)
        # compute test loss
        loss_te = calculate_logistic_loss(test_y, test_tx, w)

        accuracy = accuracy_logistic(test_tx, test_y[:, 0], w)
        accuracies.append(accuracy)
        losses.append(loss_te)

    # show the average accuracy and the standard deviation
    accuracy_mean = np.mean(accuracies)
    print("Accuracy, ", accuracy_mean, " +/- ", np.std(accuracies))

    loss_te = np.sqrt(2 * np.mean(losses))

    return loss_te, accuracy_mean


def cross_validation_logistic_regression_polynomial_exp_full(y, tx, degree, k_fold):
    """return the average test rmse and average accuracy for a full cross validation of
        polynomially expanded logistic regression on k_fold.
       print the average accuracy and the standard deviation of accuracy
    Args:
        y:          shape=(N,)
        tx:          shape=(N,D)
        degree:  int, degree of polynomial expansion
        k_fold: integer, the number of folds

    Returns:
        test rmse and average accuracy.

    """
    accuracies = []
    losses = []

    # build the indices
    k_indices = build_k_indices(y, k_fold, 1)

    # initialize w
    initial_w = np.zeros(((tx.shape[1] - 1) * degree + 1, 1))

    # we perfom a logistic regression for each fold
    for k in range(k_fold):

        # separate the data
        train_tx, train_y, test_tx, test_y = separate_data(tx, y, k_indices, k)

        # create the polynomials
        train_exp_tx = build_poly(train_tx, degree)
        test_exp_tx = build_poly(test_tx, degree)

        # compute train weight and train loss
        w, loss_tr = logistic_regression(
            train_y, train_exp_tx, initial_w, 1000, 0.02
        )

        # compute test loss
        loss_te = calculate_logistic_loss(test_y, test_exp_tx, w)

        accuracy = accuracy_logistic(test_tx, test_y[:, 0], w)
        accuracies.append(accuracy)
        losses.append(loss_te)

    # show the average accuracy and the standard deviation
    accuracy_mean = np.mean(accuracies)
    print("Accuracy, ", np.mean(accuracies), " +/- ", np.std(accuracies))

    loss_te = np.sqrt(2 * loss_te)

    return loss_te, accuracy_mean


def cross_validation_ridge_regression(y, tx, k_indices, k, lambda_):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        tx:          shape=(N,D)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

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
    # perform cross validation for each lambda
    for l in lambdas:
        rmse_tr_temp = 0.0
        rmse_te_temp = 0.0
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation_ridge_regression(y, tx, k_indices, k, l)
            rmse_tr_temp += loss_tr
            rmse_te_temp += loss_te
        # compute the average of losses
        rmse_tr_temp = rmse_tr_temp / k_fold
        rmse_te_temp = rmse_te_temp / k_fold
        # update the best loss and lambda
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


def cross_validation_reg_logistic_regression_exp(
    y, tx, degree, initial_w, k_indices, k, lambda_
):
    """return the loss of polynomially expanded logistic regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        tx:          shape=(N,D)
        degree:     scalar, cf. build_poly()
        initial_w:   shape=(N, (D - 1) *degree +1)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()

    Returns:
        train, test rmse and accuracy
    """

    # separate data
    train_tx, train_y, test_tx, test_y = separate_data(tx, y, k_indices, k)

    # build  polynomials
    train_exp_tx = build_poly(train_tx, degree)
    test_exp_tx = build_poly(test_tx, degree)

    w, loss_tr = reg_logistic_regression(
        train_y, train_exp_tx, lambda_, initial_w, 10000, 0.02
    )

    loss_te = calculate_logistic_loss(
        test_y, test_exp_tx, w
    )  # + lambda_ * np.dot(w.T, w)

    loss_tr = np.sqrt(2 * loss_tr)  # + lambda_ * np.dot(w.T, w)

    loss_te = np.sqrt(2 * loss_te)  # + lambda_ * np.dot(w.T, w)

    accuracy = accuracy_logistic(test_exp_tx, test_y[:, 0], w)

    return loss_te, loss_tr, accuracy


def find_best_lambda_reg_logistic_regression_exp(tx, y, degree, k_fold, lambdas):
    """cross validation of polynomially expanded logistic regression over regularisation parameter lambda.

    Args:
        y:          shape=(N,)
        tx:          shape=(N,D)
        degree:     scalar, cf. build_poly()
        initial_w:   shape=(N, (D - 1) *degree +1)
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
        best_accuracy : scalar, the associated accuracy for the best lambda
    """
    seed = 12
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []

    initial_w = np.zeros(((tx.shape[1] - 1) * degree + 1, 1))
    # set the best rmse to a maximal value as a start
    best_rmse = 101
    best_lambda = 0
    best_accuracy = 0.0
    acccuracy_averages = []
    # perform cross validation for each lambda
    for l in lambdas:
        rmse_tr_temp = 0.0
        rmse_te_temp = 0.0
        accuracies = []
        for k in range(k_fold):

            loss_te, loss_tr, accuracy = cross_validation_reg_logistic_regression_exp(
                y, tx, degree, initial_w, k_indices, k, l
            )

            rmse_tr_temp += loss_tr
            rmse_te_temp += loss_te
            accuracies.append(accuracy)
        avg_accuracy_temp = np.mean(accuracies)
        print(
            "Accuracy for lambda ",
            l,
            ": ",
            avg_accuracy_temp,
            " +/- ",
            np.std(accuracies),
        )

        # compute the average of losses
        rmse_tr_temp = rmse_tr_temp / k_fold
        rmse_te_temp = rmse_te_temp / k_fold

        # update the best loss and lambda
        if rmse_te_temp < best_rmse:
            best_rmse = rmse_te_temp
        if avg_accuracy_temp > best_accuracy:
            best_accuracy = avg_accuracy_temp
            best_lambda = l

        rmse_tr.append(rmse_tr_temp)
        rmse_te.append(rmse_te_temp)
        acccuracy_averages.append(avg_accuracy_temp)
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    print(
        " the choice of lambda which leads to the best test accuracy is %.9f with an accuracy of %.3f"
        % (best_lambda, best_accuracy)
    )
    return best_lambda, best_rmse


def cross_validation_logistic_regression_gamma(
    tx, y, initial_w, k_indices, k, iterations, gamma
):
    """return the loss of logistic regression for a fold corresponding to k_indices
       prints the accuracy average of 10 different optimisations with the same parameters

    Args:
        tx:         shape=(N,D)
        y:          shape=(N,)
        initial_w:  shape=(N, D)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        iterations: scalar, nb of iterations
        gamma: scalar, the learning rate

    Returns:
        train, test rmse and accuracy

    """
    accuracies = []
    # separate data
    train_tx, train_y, test_tx, test_y = separate_data(tx, y, k_indices, k)
    # we perform 10 optimizations over the same parameters
    rmse_tr = []
    rmse_te = []

    for i in range(10):
        # get weights and  train loss
        w, loss_tr = logistic_regression(
            train_y, train_tx, initial_w, iterations, gamma
        )
        # compute test loss
        loss_te = compute_loss_mse(test_y, test_tx, w)
        # add the current accuracy
        accuracies.append(accuracy_logistic(test_tx, test_y[:, 0], w))
        rmse_tr.append(np.sqrt(2 * loss_tr))
        rmse_te.append(np.sqrt(2 * loss_te))

    accuracy = np.mean(accuracies)
    print("Accuracy, ", accuracy, " +/- ", np.std(accuracies))

    return np.mean(rmse_tr), np.mean(rmse_te), accuracy


def find_best_gamma_logistic_regression(tx, y, initial_w, gammas):
    """cross validation  of logistic regression over learning rate parameter gamma.
       NOTE: for simplicity's sake we are performing a 2 fold, with constant k

    Args:
        tx:         shape=(N,D)
        y:          shape=(N,)
        initial_w:  shape=(N, D)
        gammas: shape = (p, ) where p is the number of values of gamma to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """
    l = 0.0
    iterations = 1000
    seed = 12
    # split data in 2 folds
    k_indices = build_k_indices(y, 2, seed)
    k = 1
    # define lists to store the loss of training data and test data

    # set the best rmse to a maximal value as a start
    best_rmse = 101
    best_gamma = 0
    best_accuracy = 0.0
    rmse_tr = []
    rmse_te = []
    accuracies_means = []
    for g in gammas:
        rmse_tr_temp = 0.0
        rmse_te_temp = 0.0
        (
            rmse_tr_temp,
            rmse_te_temp,
            accuracy,
        ) = cross_validation_logistic_regression_gamma(
            tx, y, initial_w, k_indices, k, iterations, g
        )
        print("test rmse : ", rmse_te_temp, " gamma ", g)
        #    rmse_tr_temp += loss_tr
        #    rmse_te_temp += loss_te
        # do the average
        # rmse_tr_temp = rmse_tr_temp / k_fold
        # rmse_te_temp = rmse_te_temp / k_fold
        if rmse_te_temp < best_rmse:
            best_rmse = rmse_te_temp
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_gamma = g

        accuracies_means.append(accuracy)
        rmse_tr.append(rmse_tr_temp)
        rmse_te.append(rmse_te_temp)
    cross_validation_visualization_gamma(gammas, rmse_tr, rmse_te)
    print(
        " the choice of gamma which leads to the best test rmse is %.9f with an accuracy of %.7f"
        % (best_gamma, best_accuracy)
    )
    return best_gamma, best_rmse
