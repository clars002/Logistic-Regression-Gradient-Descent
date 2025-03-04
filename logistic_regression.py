# This file contains the mathematical functions for gradient descent with logistic regression
import numpy as np


# Calculate loss in matrix-vector form
def logreg_loss(X: np.array, y: np.array, w: np.array):
    Xw = np.matmul(X, w)
    return np.sum(np.log(1 + np.exp(Xw))) - np.matmul(y.transpose(), Xw)


# Calculate output probabilities (based on equation for p)
def probability(X, w):
    Xw = np.matmul(X, w)
    p = np.exp(Xw)
    p = p / (1 + p)

    return p


# Calculate gradient
def logreg_gradient(X: np.array, y: np.array, w: np.array):
    p = probability(X, w)

    return np.matmul(X.transpose(), (p - y))


# Gradient descent algorithm
def logreg_grad_descent(
    w_0: np.array, X: np.array, y: np.array, step: float, iterations: int
):
    w_k = w_0
    losses = [0] * iterations
    for i in range(iterations):
        search_dir = -logreg_gradient(X, y, w_k)
        w_k = w_k + (step * search_dir)
        cur_loss = logreg_loss(X, y, w_k)
        losses[i] = cur_loss
    return w_k, losses


# Output predictions based on given threshold
def predict(p_vals: np.array, threshold: float):
    predictions = p_vals.copy()
    predictions[predictions >= threshold] = 1
    predictions[predictions < threshold] = 0

    return predictions
