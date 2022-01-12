import numpy as np
import matplotlib.pyplot as plt
import models


def draw_points(m):
    cov = np.identity(2)
    mean = [0, 0]
    vec = np.array([0.3, -0.5])
    ones = np.array([1] * m)
    minus_ones = ones * (-1)
    check = True

    while (check):  # check not all entries in y are same

        X = np.random.multivariate_normal(mean, cov, m)
        y = np.sign(vec.dot(X.T) + 0.1)
        check = np.all(np.subtract(y, ones)) or np.all(np.subtract(y, minus_ones))

    return X, y



def plot(data, true_hypothsis_data, perceptron_data, svm_data, m):
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title("Data analysis over %s samples" % m)
    ax.plot(data)
    ax.plot(true_hypothsis_data)
    ax.plot(perceptron_data)
    ax.plot(svm_data)

    plt.show()


def categorized_data(X, y):
    '''
    seperates the dots where y = -+1
    '''
    y_positive_indexes = np.where(y == 1)
    y_positive = [i[1] for i in X[y_positive_indexes]]
    x_positive = [i[0] for i in X[y_positive_indexes]]

    y_negative_indexes = np.where(y == -1)
    y_negative = [i[1] for i in X[y_negative_indexes]]
    x_negative = [i[0] for i in X[y_negative_indexes]]

    return x_positive, y_positive, x_negative, y_negative


def generate_hyperPlane(X, w, noise=0.):
    x = np.linspace(np.min(X), np.max(X))
    a = -w[0] / w[1]
    y = a * x - noise / w[1]

    return x, y


def question_9():
    for m in [5, 10, 15, 25, 70]:
        X, y = draw_points(m)
        x_true, y_true = generate_hyperPlane(X, [0.3, -0.5], 0.1)

        x_positive, y_positive, x_negative, y_negative = categorized_data(X, y)

        perceptron_data = models.active_Perceptron(X.T, y)
        x_perceptron, y_perceptron = generate_hyperPlane(X, [perceptron_data[1], perceptron_data[2]],
                                                         perceptron_data[0])

        svm_data = models.active_SVM(X.T, y)
        x_svm, y_svm = generate_hyperPlane(X, [svm_data.model[0], svm_data.model[1]], svm_data.intercept)

        fig = plt.figure()
        ax = fig.add_subplot()
        plt.title("Data analysis over %s samples" % m)
        ax.plot(x_positive, y_positive, ".")
        ax.plot(x_negative, y_negative, ".")
        ax.plot(x_true, y_true, color="red", label="True hypothesis")
        ax.plot(x_perceptron, y_perceptron, color="yellow", label="Perceptron hypothesis")
        ax.plot(x_svm, y_svm, color="green", label="SVM hypothesis")
        plt.legend()

        plt.show()


def question_10():

    perceptron_acuuracies_per_m = []
    svm_accuracies_per_m = []
    lda_accuracies_per_m = []

    sample_num = [5, 10, 15, 25, 70]

    for m in sample_num:
        print(m)

        perceptron_acuuracies = []
        svm_accuracies = []
        lda_accuracies = []

        for round in range(500):
            print(round / 500 * 100, "%")

            X_train, y_train = draw_points(m)
            X_test, y_test = draw_points(10000)

            perceptron_data = models.test_Perceptron(X_train.T, y_train, X_test.T, y_test)
            perceptron_acuuracies.append(perceptron_data.accuracy)
            print("perceptron done")


            svm_data = models.test_SVM(X_train.T, y_train, X_test.T, y_test)
            svm_accuracies.append(svm_data.accuracy)
            print("SVM done")

            lda_data = models.test_LDA(X_train.T, y_train, X_test.T, y_test)
            lda_accuracies.append(lda_data.accuracy)
            print("LDA done\n")

        perceptron_acuuracies_per_m.append(np.mean(perceptron_acuuracies))
        svm_accuracies_per_m.append(np.mean(svm_accuracies))
        lda_accuracies_per_m.append(np.mean(lda_accuracies))

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title("mean accuracy over number of samples")
    print(perceptron_acuuracies_per_m)
    ax.plot(sample_num, perceptron_acuuracies_per_m, color="yellow", label="Perceptron hypothesis")
    ax.plot(sample_num, svm_accuracies_per_m, color="green", label="SVM hypothesis")
    ax.plot(sample_num, lda_accuracies_per_m, color="red", label="LDA hypothesis")


    plt.legend()

    plt.show()


question_10()

