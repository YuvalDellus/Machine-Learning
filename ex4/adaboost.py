"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np
import matplotlib.pyplot as plt
import ex4_tools


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None] * T  # list of base learners
        self.w = np.zeros(T)  # weights
        self.D = 0

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        D = np.array([1 / y.size] * y.size)

        for iteration in range(self.T):
            h = self.WL(D, X, y)

            predicted = h.predict(X)
            mistakes = np.where(predicted != y)
            epsilon = np.sum(D[mistakes])
            w = 0.5 * np.log(1 / epsilon - 1)

            D = np.exp(-y * w * predicted) * D  # can be faster by calculate e^w_t before
            D /= np.sum(D)

            self.h[iteration] = h
            self.w[iteration] = w

        self.D = D

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        predict = np.zeros(X.shape[0])

        for i, h in enumerate(self.h):
            if i == max_t:
                break
            predict += h.predict(X) * self.w[i]

        y_hat = np.sign(predict)

        return y_hat

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the correct predictions when predict only with max_t weak learners (float)
        """
        predicted = self.predict(X, max_t)
        errors = np.sum(predicted != y)

        return errors / len(y)


def plot_q_10(data_size, noise_rate=0):
    error_train = []
    error_test = []
    X, y = ex4_tools.generate_data(data_size, noise_rate)
    X_test, y_test = ex4_tools.generate_data(200, noise_rate)
    WL = ex4_tools.DecisionStump
    ada = AdaBoost(WL, 500)

    ada.train(X, y)

    for council_size in range(500):
        # print(council_size / learner_num * 100, "%")

        ada.predict(X, council_size)
        error_train.append(ada.error(X, y, council_size))

        ada.predict(X_test, council_size)
        error_test.append(ada.error(X_test, y_test, council_size))

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title("Error rate over size of council")
    ax.plot(error_train, label="Error over train Data")
    ax.plot(error_test, label="Error over test Data")
    plt.legend()
    plt.xlabel("Size of council")
    plt.ylabel("Error rate")

    plt.show()


def q_11(data_size, noise_rate=0):
    learner_size = [5, 10, 50, 100, 200, 500]
    X, y = ex4_tools.generate_data(data_size, noise_rate)
    X_test, y_test = ex4_tools.generate_data(200, noise_rate)
    WL = ex4_tools.DecisionStump
    index = 1

    for size in learner_size:
        ada = AdaBoost(WL, size)
        ada.train(X, y)
        plt.subplot(2, 3, index)
        ex4_tools.decision_boundaries(ada, X_test, y_test, size)
        index += 1

    plt.show()


def q_12(data_size, noise_rate=0):
    error_test = []
    learners = []
    learner_size = [5, 10, 50, 100, 200, 500]
    X, y = ex4_tools.generate_data(data_size, noise_rate)
    X_test, y_test = ex4_tools.generate_data(200, noise_rate)
    WL = ex4_tools.DecisionStump

    for size in learner_size:
        ada = AdaBoost(WL, size)
        ada.train(X, y)
        learners.append(ada)
        error_test.append(ada.error(X_test, y_test, size))

    min_error = min(error_test)
    ada = learners[error_test.index(min_error)]
    best_learner_size = learner_size[error_test.index(min_error)]

    ex4_tools.decision_boundaries(ada, X, y, best_learner_size)
    plt.suptitle("Error: %s" % min_error, y=1)
    plt.show()


def q_13(data_size, noise_rate=0):
    X, y = ex4_tools.generate_data(data_size, noise_rate)
    WL = ex4_tools.DecisionStump
    ada = AdaBoost(WL, 500)
    ada.train(X, y)
    D = ada.D / np.max(ada.D) * 10

    ex4_tools.decision_boundaries(ada, X, y, num_classifiers=500, weights=D)
    plt.suptitle("Training set with size proportional to its weight", y=1)
    plt.show()


def q_14(data_size, noise_rate):
    for noise in noise_rate:
        plot_q_10(data_size, noise)
        q_11(data_size, noise)
        q_12(data_size, noise)
        q_13(data_size, noise)



# plot_q_10(5000, 0.4)
# q_11(5000)
# q_12(5000)
# q_13(5000)
# q_14(5000, [0.01, 0.4])
