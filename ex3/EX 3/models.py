import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# -------------------------------------- Perceptron ----------------------------------------------------

class Perceptron:

    def __init__(self):
        self.model = 0
        self.accuracy = 0
        self.predicted_y = 0

    def add_ones_row(self, X):

        ones = np.ones(X.shape[1]).reshape([1, X.shape[1]])
        X = np.concatenate((ones, X), axis=0)

        return X

    def fit(self, X, y):

        X = self.add_ones_row(X)
        w = np.zeros(X.shape[0])

        while (True):
            wX = w.dot(X)

            ywX = [np.array(np.multiply(y, wX))]

            index = int(np.argmin(ywX))

            if ywX[0][index] > 0:
                self.model = w
                break

            w += (y[index] * X.T[index])

    def predict(self, X):

        labels = []
        X = self.add_ones_row(X)

        predict_vec = self.model.dot(X)

        for i in predict_vec:
            if i < 0:
                labels.append(-1)
            else:
                labels.append(1)

        self.predicted_y = labels

        return labels

    def score(self, X, y, y_minus_label=-1):

        FP = 0  # False Positive
        TP = 0  # True Positive
        FN = 0  # False Negative
        TN = 0  # True Negative
        P = 0  # number of positive
        N = 0  # number of negative
        dict = {}

        for i in range(len(y)):
            if y[i] == 1:
                P += 1
                if self.predicted_y[i] == 1:
                    TP += 1
                else:
                    FN += 1

            if y[i] == y_minus_label:
                N += 1
                if self.predicted_y[i] == 1:
                    FP += 1
                else:
                    TN += 1

        num_samples = X.shape[1]
        error = (FP + FN) / (P + N)
        accuracy = (TP + TN) / (P + N)
        precision = 0 if TP == 0 else TP / (TP + FP)
        recall = TP / P

        FPR = FP / num_samples
        TPR = TP / num_samples
        FNR = FN / num_samples

        dict["num_samples"] = num_samples
        dict["error"] = error
        dict["accuracy"] = accuracy
        dict["FPR"] = FPR
        dict["TPR"] = TPR
        # dict["FNR"] = FNR
        dict["precision"] = precision
        dict["recall"] = recall

        self.accuracy = accuracy

        return dict


# -------------------------------------- LDA ----------------------------------------------------

class LDA:

    def __init__(self):
        self.avg_plus_ones = 0
        self.avg_minus_ones = 0
        self.mean_data_plus_ones = 0
        self.mean_data_minus_ones = 0
        self.cov_mat = 0
        self.accuracy = 0
        self.predicted_y = 0

    def fit(self, X, y):
        num_of_plus_ones = max(np.sum(y == 1), 1)
        num_of_minus_ones = max(np.sum(y == -1), 1)
        self.avg_plus_ones = num_of_plus_ones / y.size
        self.avg_minus_ones = num_of_minus_ones / y.size
        self.mean_data_plus_ones = X[:, np.where(y == 1)[0]].sum(axis=1) / num_of_plus_ones
        self.mean_data_minus_ones = X[:, np.where(y == -11)[0]].sum(axis=1) / num_of_minus_ones
        self.cov_mat = np.cov(X)

    def predict(self, X):

        cov_mat_inv = np.linalg.inv(self.cov_mat)

        delta_plus = X.T.dot(cov_mat_inv).dot(self.mean_data_plus_ones) - 0.5 * self.mean_data_plus_ones.T.dot(
            cov_mat_inv).dot(self.mean_data_plus_ones) + np.log(self.avg_plus_ones)

        delta_minus = X.T.dot(cov_mat_inv).dot(self.mean_data_minus_ones) - 0.5 * self.mean_data_minus_ones.T.dot(
            cov_mat_inv).dot(self.mean_data_minus_ones) + np.log(self.avg_minus_ones)

        delta_plus_max = np.array([1])
        delta_minus_max = np.array([-1])

        prediction = np.where(delta_minus < delta_plus, delta_plus_max, delta_minus_max)

        self.predicted_y = prediction

        return prediction

    def score(self, X, y, y_minus_label=-1):

        FP = 0  # False Positive
        TP = 0  # True Positive
        FN = 0  # False Negative
        TN = 0  # True Negative
        P = 0  # number of positive
        N = 0  # number of negative
        dict = {}

        for i in range(len(y)):
            if y[i] == 1:
                P += 1
                if self.predicted_y[i] == 1:
                    TP += 1
                else:
                    FN += 1

            if y[i] == y_minus_label:
                N += 1
                if self.predicted_y[i] == 1:
                    FP += 1
                else:
                    TN += 1

        num_samples = X.shape[1]
        error = (FP + FN) / (P + N)
        accuracy = (TP + TN) / (P + N)
        precision = 0 if TP == 0 else TP / (TP + FP)
        recall = TP / P

        FPR = FP / num_samples
        TPR = TP / num_samples
        FNR = FN / num_samples

        dict["num_samples"] = num_samples
        dict["error"] = error
        dict["accuracy"] = accuracy
        dict["FPR"] = FPR
        dict["TPR"] = TPR
        # dict["FNR"] = FNR
        dict["precision"] = precision
        dict["recall"] = recall

        self.accuracy = accuracy

        return dict


# -------------------------------------- SVM ----------------------------------------------------

class SVM:

    def __init__(self, C=1e10):
        self.svm = SVC(C=C, kernel='linear')
        self.model = 0
        self.y = 0
        self.y_predict = 0
        self.intercept = 0
        self.accuracy = 0
        self.predicted_y = 0

    def fit(self, X, y):

        w = self.svm.fit(X.T, y)

        self.y = y
        self.model = self.svm.coef_[0]
        self.intercept = self.svm.intercept_[0]

    def predict(self, X):

        self.predicted_y = self.svm.predict(X.T)

        return self.predicted_y

    def score(self, X, y, y_minus_label=-1):

        FP = 0  # False Positive
        TP = 0  # True Positive
        FN = 0  # False Negative
        TN = 0  # True Negative
        P = 0  # number of positive
        N = 0  # number of negative
        dict = {}

        for i in range(len(y)):
            if y[i] == 1:
                P += 1
                if self.predicted_y[i] == 1:
                    TP += 1
                else:
                    FN += 1

            if y[i] == y_minus_label:
                N += 1
                if self.predicted_y[i] == 1:
                    FP += 1
                else:
                    TN += 1

        num_samples = X.shape[1]
        error = (FP + FN) / (P + N)
        accuracy = (TP + TN) / (P + N)
        precision = 0 if TP == 0 else TP / (TP + FP)
        recall = TP / P

        FPR = FP / num_samples
        TPR = TP / num_samples
        FNR = FN / num_samples

        dict["num_samples"] = num_samples
        dict["error"] = error
        dict["accuracy"] = accuracy
        dict["FPR"] = FPR
        dict["TPR"] = TPR
        # dict["FNR"] = FNR
        dict["precision"] = precision
        dict["recall"] = recall

        self.accuracy = accuracy

        return dict


# -------------------------------------- Logistic ----------------------------------------------

class Logistic:

    def __init__(self):
        self.logistic = LogisticRegression(solver='liblinear')
        self.model = 0
        self.y = 0
        self.predicted_y = 0

    def fit(self, X, y):
        w = self.logistic.fit(X.T, y)

        self.y = y
        self.model = w

    def predict(self, X):
        self.predicted_y = self.logistic.predict(X.T)

        return self.predicted_y

    def score(self, X, y, y_minus_label=-1):

        FP = 0  # False Positive
        TP = 0  # True Positive
        FN = 0  # False Negative
        TN = 0  # True Negative
        P = 0  # number of positive
        N = 0  # number of negative
        dict = {}

        for i in range(len(y)):
            if y[i] == 1:
                P += 1
                if self.predicted_y[i] == 1:
                    TP += 1
                else:
                    FN += 1

            if y[i] == y_minus_label:
                N += 1
                if self.predicted_y[i] == 1:
                    FP += 1
                else:
                    TN += 1

        num_samples = X.shape[1]
        error = (FP + FN) / (P + N)
        accuracy = (TP + TN) / (P + N)
        precision = 0 if TP == 0 else TP / (TP + FP)
        recall = TP / P

        FPR = FP / num_samples
        TPR = TP / num_samples
        FNR = FN / num_samples

        dict["num_samples"] = num_samples
        dict["error"] = error
        dict["accuracy"] = accuracy
        dict["FPR"] = FPR
        dict["TPR"] = TPR
        # dict["FNR"] = FNR
        dict["precision"] = precision
        dict["recall"] = recall

        self.accuracy = accuracy

        return dict


# -------------------------------------- DecisionTree ----------------------------------------------

class DecisionTree:

    def __init__(self, depth=1):
        self.decision_tree = DecisionTreeClassifier(max_depth=depth)
        self.model = 0
        self.y = 0
        self.predicted_y = 0

    def fit(self, X, y):

        w = self.decision_tree.fit(X.T, y)

        self.y = y
        self.model = w

    def predict(self, X):

        self.predicted_y = self.decision_tree.predict(X.T)

        return self.predicted_y

    def score(self, X, y, y_minus_label=-1):

        FP = 0  # False Positive
        TP = 0  # True Positive
        FN = 0  # False Negative
        TN = 0  # True Negative
        P = 0  # number of positive
        N = 0  # number of negative
        dict = {}

        for i in range(len(y)):
            if y[i] == 1:
                P += 1
                if self.predicted_y[i] == 1:
                    TP += 1
                else:
                    FN += 1

            if y[i] == y_minus_label:
                N += 1
                if self.predicted_y[i] == 1:
                    FP += 1
                else:
                    TN += 1

        num_samples = X.shape[1]
        error = (FP + FN) / (P + N)
        accuracy = (TP + TN) / (P + N)
        precision = 0 if TP == 0 else TP / (TP + FP)
        recall = TP / P

        FPR = FP / num_samples
        TPR = TP / num_samples
        FNR = FN / num_samples

        dict["num_samples"] = num_samples
        dict["error"] = error
        dict["accuracy"] = accuracy
        dict["FPR"] = FPR
        dict["TPR"] = TPR
        # dict["FNR"] = FNR
        dict["precision"] = precision
        dict["recall"] = recall

        self.accuracy = accuracy

        return dict


# -------------------------------------- KNearestNeighbors ----------------------------------------------------

class KNearestNeighbors:

    def __init__(self):
        self.nearest_neighbor = KNeighborsClassifier(n_neighbors=30)
        self.model = 0
        self.y = 0
        self.accuracy = 0
        self.predicted_y = 0

    def fit(self, X, y):
        w = self.nearest_neighbor.fit(X.T, y)

        self.y = y
        self.model = w

    def predict(self, X):
        self.predicted_y = self.nearest_neighbor.predict(X.T)

        return self.predicted_y

    def score(self, X, y, y_minus_label=-1):

        FP = 0  # False Positive
        TP = 0  # True Positive
        FN = 0  # False Negative
        TN = 0  # True Negative
        P = 0  # number of positive
        N = 0  # number of negative
        dict = {}

        for i in range(len(y)):
            if y[i] == 1:
                P += 1
                if self.predicted_y[i] == 1:
                    TP += 1
                else:
                    FN += 1

            if y[i] == y_minus_label:
                N += 1
                if self.predicted_y[i] == 1:
                    FP += 1
                else:
                    TN += 1

        num_samples = X.shape[1]
        error = (FP + FN) / (P + N)
        accuracy = (TP + TN) / (P + N)
        precision = 0 if TP == 0 else TP / (TP + FP)
        recall = TP / P

        FPR = FP / num_samples
        TPR = TP / num_samples
        FNR = FN / num_samples

        dict["num_samples"] = num_samples
        dict["error"] = error
        dict["accuracy"] = accuracy
        dict["FPR"] = FPR
        dict["TPR"] = TPR
        # dict["FNR"] = FNR
        dict["precision"] = precision
        dict["recall"] = recall

        self.accuracy = accuracy

        return dict


# -------------------------------------- Activisions ----------------------------------------------


def form_data(featuers=10, sample_size=5):
    y = np.random.binomial(1, 0.35, (1, sample_size))

    for index in range(len(y[0])):
        if (y[0][index] == 0):
            y[0][index] = -1

    y = np.array(y[0])

    X = np.random.normal(loc=5, size=(sample_size, featuers)).T

    return X, y


def active_Perceptron(X="-1", y=None):
    if type(X) == str:
        X, y = form_data(10)

    model = Perceptron()
    model.fit(X, y)
    model.predict(X)
    model.score(X, y)
    return model


def active_LDA(X, y):
    # X, y = form_data(50)
    model = LDA()
    model.fit(X, y)
    model.predict(X)
    model.score(X, y)
    return model


def active_SVM(X, y):
    # X, y = form_data(50)
    svm = SVM()
    svm.fit(X, y)
    svm.predict(X)
    svm.score(X, y)
    return svm


def test_Perceptron(X_training, y_training, X_testing, y_testing):
    model = Perceptron()
    model.fit(X_training, y_training)
    model.predict(X_testing)
    model.score(X_testing, y_testing)
    return model


def test_LDA(X_training, y_training, X_testing, y_testing):
    model = LDA()
    model.fit(X_training, y_training)
    model.predict(X_testing)
    model.score(X_testing, y_testing)
    return model


def test_SVM(X_training, y_training, X_testing, y_testing, C=None):
    model = SVM()
    model.fit(X_training, y_training)
    model.predict(X_testing)
    model.score(X_testing, y_testing)
    return model


def test_Logistic(X_training, y_training, X_testing, y_testing):
    model = Logistic()
    model.fit(X_training, y_training)
    model.predict(X_testing)
    model.score(X_testing, y_testing)
    return model


def test_DecisionTree(X_training, y_training, X_testing, y_testing, depth=None):
    model = DecisionTree(depth)
    model.fit(X_training, y_training)
    model.predict(X_testing)
    model.score(X_testing, y_testing)
    return model
