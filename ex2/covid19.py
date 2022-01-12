import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fit_linear_regression(X, y):
    X_dagger = np.linalg.pinv(X)

    w = np.matmul(X_dagger.T, y)

    Sigma = np.linalg.svd(X, compute_uv=False)

    return w, Sigma


def load_data(path):
    return pd.read_csv(path, index_col=False)



def q_19(path):
    log_detected = []
    data = load_data(path)

    for num in data["detected"]:
        log_detected.append(np.log(num))

    data["log_detected"] = log_detected

    return data


def q_20_21(path):
    data = q_19(path)

    X = np.array(data["day_num"]).reshape(len(data["day_num"]), 1)
    y = np.array(data["log_detected"]).reshape(len(data["log_detected"]), 1)

    w, sigma = fit_linear_regression(X.T, y)

    y_hat = np.matmul(X, w)
    y_hat_log = np.exp(y_hat)


    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title("log_detected as a function of day_num")
    plt.xlabel("day_num")
    plt.ylabel("log_detected")
    ax.plot(data["day_num"], data["log_detected"], ".")
    ax.plot(data["day_num"], y_hat)

    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title("detected as a function of day_num")
    plt.xlabel("day_num")
    plt.ylabel("log_detected")
    ax.plot(data["day_num"], data["detected"], ".")
    ax.plot(data["day_num"], y_hat_log)

    plt.show()


# q_19("C:\IML programing\covid19_israel.csv")
# q_20_21("C:\IML programing\covid19_israel.csv")
