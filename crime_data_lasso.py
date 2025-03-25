if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    y_train = df_train['ViolentCrimesPerPop']
    x_train = df_train.drop('ViolentCrimesPerPop', axis=1)
    y_test = df_test['ViolentCrimesPerPop']
    x_test = df_test.drop('ViolentCrimesPerPop', axis=1)

    _lambda = np.max(2*abs(x_train.T.dot(y_train-np.mean(y_train))))

    nonzero_counts = []
    lambda_values = []
    features = ["agePct12t29", "pctWSocSec", "pctUrban", "agePct65up", "householdsize"]
    f_weights = {f: [] for f in features}
    train_error = []
    test_error = []

    w = np.zeros(x_train.shape[1])

    while _lambda >= 0.01:
        w, bias = train(x_train, y_train,  _lambda=_lambda, start_weight=w)
        nonzero_count = np.sum(w != 0) 

        nonzero_counts.append(nonzero_count)
        lambda_values.append(_lambda)

        for feat in features:
            idx = x_train.columns.get_loc(feat)
            f_weights[feat].append(w[idx])

        train_sse = np.sum((x_train @ w + bias - y_train) ** 2)
        test_sse = np.sum((x_test @ w + bias - y_test) ** 2)

        train_error.append(train_sse)
        test_error.append(test_sse)

        _lambda *= 0.50  # reduce lambda iteratively

    # plot results
    plt.figure(figsize=(8, 6))
    plt.plot(lambda_values, nonzero_counts)
    plt.xscale("log")
    plt.xlabel('log($\\lambda$)')
    plt.ylabel("# of nonzero weights")
    plt.show()

    plt.figure(figsize=(10, 6))
    for f in features:
        plt.plot(lambda_values, f_weights[f], label=f)

    plt.xscale("log")
    plt.xlabel("log($\\lambda$)")
    plt.ylabel("weight")
    plt.legend()
    plt.show()

    # when lambda = 30
    w, bias = train(x_train, y_train,  _lambda=30, start_weight=w)
    feature_names = x_train.columns
    feature_coefficients = dict(zip(feature_names, w))
    most_positive_feature = max(feature_coefficients, key=feature_coefficients.get)
    most_negative_feature = min(feature_coefficients, key=feature_coefficients.get)
    print(f"largest positive Lasso coefficient: {most_positive_feature}")
    print(f"most negative Lasso coefficient: {most_negative_feature}")

    plt.figure(figsize=(8, 6))
    plt.plot(lambda_values, train_error, label="train squared error")
    plt.plot(lambda_values, test_error, label="test squared error")
    plt.xscale("log") 
    plt.xlabel("log($\\lambda$)")
    plt.ylabel("squared error")
    plt.legend()
    plt.show()

    


if __name__ == "__main__":
    main()
