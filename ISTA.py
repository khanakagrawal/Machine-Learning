from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from utils import problem


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
    
    """

    b = bias - 2 * eta * np.sum((X @ weight + bias) - y)

    gradient = X.T @ ((X @ weight + bias) - y)
    w = weight - 2 * eta * gradient

    w = np.sign(w) * np.maximum(np.abs(w) - 2 * eta * _lambda, 0)
    
    return (w, b)


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized SSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    return np.sum((X @ weight + bias - y) ** 2) + _lambda * np.sum(np.abs(weight))


@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float = 0.00001,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: float = None
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (float, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You will also have to keep an old copy of bias for convergence criterion function.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])

    if start_bias is None:
        start_bias = 0.0
    
    old_w = np.copy(start_weight)
    old_b = start_bias

    w, b = step(X, y, start_weight, start_bias, _lambda, eta)

    while not convergence_criterion(w, old_w, b, old_b, convergence_delta):
        old_b = b
        old_w = w
        w, b = step(X, y, old_w, old_b, _lambda, eta)

    return (w, b)


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight and bias has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compare it to convergence delta.
    It should also calculate the maximum absolute change between the bias and old_b, and compare it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of gradient descent.
        old_w (np.ndarray): Weight from previous iteration of gradient descent.
        bias (float): Bias from current iteration of gradient descent.
        old_b (float): Bias from previous iteration of gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight and bias has not converged yet. True otherwise.
    """
    w_diff = np.max((abs(old_w - weight)))
    b_diff = abs(old_b - bias)

    if w_diff <= convergence_delta and b_diff <= convergence_delta:
        return True
    
    return False


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """

    # generate data
    n, d, k = 500, 1000, 100
    w = np.zeros(d)
    w[:k] = np.arange(1, k + 1) # weight vector (first k values are nonzero)
    X = np.random.normal(0, 1, (n, d)) # feature matrix, normal distribution
    scaler = StandardScaler()
    X = scaler.fit_transform(X) # standardize data
    e = np.random.normal(0, 1, n) # noise
    y = X.dot(w) + e #  calculate y

    # calculate max lambda
    _lambda = np.max(2*abs(X.T.dot(y-np.mean(y))))

    nonzero_counts = []
    lambda_values = []
    fdr = []
    tpr = []

    while True:
        weights, bias = train(X, y, _lambda)
        nonzero_count = np.sum(weights != 0)

        incorrect_nonzeros = np.sum((weights != 0) & (w == 0)) 
        correct_nonzeros = np.sum((weights != 0) & (w != 0)) 
        fdr_val = incorrect_nonzeros / nonzero_count if nonzero_count > 0 else 0
        tpr_val = correct_nonzeros / k  

        nonzero_counts.append(nonzero_count)
        lambda_values.append(_lambda)
        fdr.append(fdr_val)
        tpr.append(tpr_val)

        if nonzero_count >= 100:  
            break  # stop when we get close to 100 nonzero weights

        _lambda *= 0.80  # reduce lambda iteratively

    # plot results
    plt.figure(figsize=(8, 6))
    plt.plot(lambda_values, nonzero_counts, marker='o', linestyle='-')
    plt.xscale("log")
    plt.xlabel('log($\\lambda$)')
    plt.ylabel("# of nonzero weights")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(fdr, tpr)
    plt.xlabel("FDR")
    plt.ylabel("TPR")
    plt.show()
    



if __name__ == "__main__":
    main()
