"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        # You can add additional fields
        self.mean=None
        self.variance=None

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        l = len(X)
        XD: np.ndarray = np.zeros((l, degree))
        for d in range (1, degree + 1):
            for n in range(0, l):
                XD[n, d-1] = X[n]**d
        
        return XD

        # raise NotImplementedError("Your Code Goes Here")

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You will need to apply polynomial expansion and data standardization first.
        """
        n = len(X)
        XD = self.polyfeatures(X, self.degree)
        self.mean = np.mean(XD, axis=0)
        self.variance = np.std(XD, axis=0)

        # demean data (standardizing)
        XD = (XD - self.mean) /  self.variance

        X_ = np.c_[np.ones([n, 1]), XD]

        n, d = X_.shape
        # remove 1 for the extra column of ones we added to get the original num features
        d = d - 1

        # construct reg matrix
        reg_matrix = self.reg_lambda * np.eye(d + 1)
        reg_matrix[0, 0] = 0

        # analytical solution (X'X + regMatrix)^-1 X' y
        self.weight = np.linalg.solve(X_.T @ X_ + reg_matrix, X_.T @ y)


    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        n = len(X)

        XD = self.polyfeatures(X, self.degree)

        # demean data (standardizing)
        XD = (XD - self.mean) /  self.variance

        # add 1s column
        X_ = np.c_[np.ones([n, 1]), XD]

        # predict
        return X_.dot(self.weight)



@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    mse: float = 0
    for i in range(0, len(a)):
        mse += (a[i] - b[i])**2

    mse = mse / len(a)
    return mse
    


@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)
    # Fill in errorTrain and errorTest arrays
    from polyreg import PolynomialRegression
    model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)

    for i in range(1, n):
        model.fit(Xtrain[0:(i+1)], Ytrain[0:(i+1)])
        train_prediction = model.predict(Xtrain[0:(i+1)])
        errorTrain[i] = mean_squared_error(Ytrain[0:(i+1)], train_prediction)
        test_prediction = model.predict(Xtest[0:(i+1)])
        errorTest[i] = mean_squared_error(Ytest[0:(i+1)], test_prediction)
        

    return errorTrain, errorTest
    # raise NotImplementedError("Your Code Goes Here")
