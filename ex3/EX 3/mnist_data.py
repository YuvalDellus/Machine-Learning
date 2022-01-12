import numpy as np
import matplotlib.pyplot as plt
import mnist
import models


def load_process_data():
    x_train, y_train = mnist.train_images(), mnist.train_labels()
    x_test, y_test = mnist.test_images(), mnist.test_labels()

    return x_train, y_train, x_test, y_test


def draw_data(data, labels, num_of_samples):
    indexes = np.random.randint(0, data.shape[0], num_of_samples)

    return data[indexes], labels[indexes]


def filtered_data(num_1=0, num_2=1):
    x_train, y_train, x_test, y_test = load_process_data()

    num_1_indexes_test = np.where(y_test == num_1)
    num_2_indexes_test = np.where(y_test == num_2)

    num_1_indexes_train = np.where(y_train == num_1)
    num_2_indexes_train = np.where(y_train == num_2)

    all_indexes_train = np.concatenate((num_1_indexes_train[0], num_2_indexes_train[0]))
    all_indexes_test = np.concatenate((num_1_indexes_test[0], num_2_indexes_test[0]))

    return x_train[all_indexes_train], y_train[all_indexes_train], x_test[all_indexes_test], y_test[all_indexes_test]


def question_12():
    x_train, y_train, x_test, y_test = load_process_data()

    zeros_indexes = np.where(y_train == 0)
    ones_indexes = np.where(y_train == 1)

    for i in range(3):
        img = x_train[zeros_indexes[0][i]].reshape((28, 28))
        plt.imshow(img, cmap="Greys")
        plt.show()

        img = x_train[ones_indexes[0][i]].reshape((28, 28))
        plt.imshow(img, cmap="Greys")
        plt.show()


def rearrange_data(X):
    return np.reshape(X, (X.shape[0], 784))


def question_13():
    x_train, y_train, x_test, y_test = load_process_data()
    img = x_train
    return rearrange_data(img)


def question_14():

    logistic_acuuracies_per_m = []
    decisionTree_accuracies_per_m = []
    svm_accuracies_per_m = []
    k_nearest_accuracies_per_m = []

    sample_num = [50, 100, 300, 500]

    x_main_train, y_main_train, x_main_test, y_main_test = filtered_data()

    for m in sample_num:
        print(m)

        logistic_acuuracies = []
        decisionTree_accuracies = []
        svm_accuracies = []
        k_nearest_accuracies = []

        for round in range(50):
            print(round / 50 * 100, "%")

            x_train, y_train, = draw_data(x_main_train, y_main_train, 1000)
            x_test, y_test = draw_data(x_main_test, y_main_test, 1000)

            x_train = rearrange_data(x_train)
            x_test = rearrange_data(x_test)

            logistic = models.Logistic()
            logistic.fit(x_train.T, y_train)
            logistic.predict(x_test.T)
            logistic.score(x_test, y_test, 0)
            logistic_acuuracies.append(logistic.accuracy)
            print("Logistic done")

            decisionTree = models.DecisionTree(30)
            decisionTree.fit(x_train.T, y_train)
            decisionTree.predict(x_test.T)
            decisionTree.score(x_test, y_test, 0)
            decisionTree_accuracies.append(decisionTree.accuracy)
            print("DecisionTree done")

            svm = models.SVM(100)
            svm.fit(x_train.T, y_train)
            svm.predict(x_test.T)
            svm.score(x_test, y_test, 0)
            svm_accuracies.append(svm.accuracy)
            print("SVM done")

            k_nearest = models.KNearestNeighbors()
            k_nearest.fit(x_train.T, y_train)
            k_nearest.predict(x_test.T)
            k_nearest.score(x_test, y_test, 0)
            k_nearest_accuracies.append(k_nearest.accuracy)
            print("K-nearest done")


        logistic_acuuracies_per_m.append(np.mean(logistic_acuuracies))
        svm_accuracies_per_m.append(np.mean(svm_accuracies))
        decisionTree_accuracies_per_m.append(np.mean(decisionTree_accuracies))
        k_nearest_accuracies_per_m.append(np.mean(k_nearest_accuracies))



    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title("mean accuracy over number of samples")
    ax.plot(sample_num, logistic_acuuracies_per_m, color="blue", label="logistic hypothesis")
    ax.plot(sample_num, svm_accuracies_per_m, color="yellow", label="SVM hypothesis")
    ax.plot(sample_num, decisionTree_accuracies_per_m, color="green", label="DecisionTree hypothesis")
    ax.plot(sample_num, k_nearest_accuracies_per_m, color="red", label="K-nearest hypothesis")

    plt.legend()

    plt.show()


# x_train, y_train, x_test, y_test = load_process_data()
#
# x_sample, y_sample = draw_data(x_train, y_train,20)
#
#
# img =x_sample[3].reshape((28, 28))
# plt.imshow(img, cmap="Greys")
# plt.show()
#
# print(y_sample[3])

question_14()
