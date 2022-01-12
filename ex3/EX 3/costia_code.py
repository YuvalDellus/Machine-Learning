import numpy as np


class pro():

    def __init__(self):
        self.model = 0
        self.w = 0
        self.acc = 0

    def fit(self, X, y):
        extra_row = np.array((np.repeat(1, X.shape[1])))
        X = np.vstack((extra_row, X))
        X = X * y
        w = np.zeros_like(X.T[0, :])

        while (True):
            XTw = X.T @ w
            index = np.argmin(XTw)
            added_product = X.T[index, :]
            if np.amin(XTw) > 0:
                break
            w = np.add(w, added_product)

        self.model = X
        self.w = w

    def predict(self, X):
        extra_row = np.array((np.repeat(1, X.shape[1])))
        X = np.vstack((extra_row, X))

        prediction = X.T @ self.w

        return np.sign(prediction)

    def score(self, X, y):
        dict = {}

        p = np.sum(y == 1)
        n = np.sum(y == -1)

        predicted = self.predict(X)
        tp = np.nonzero((y > 0) & (y == predicted))[0].size
        fp = np.nonzero((y < 0) & ~(y == predicted))[0].size
        tn = np.nonzero((y < 0) & (y == predicted))[0].size
        fn = np.nonzero((y > 0) & ~(y == predicted))[0].size

        dict["num_samples"] = X.shape[1]
        dict["error"] = (fp + fn) / (p + n)
        dict["accuracy"] = (tp + tn) / (p + n)
        dict["FPR"] = fp / n
        dict["TPR"] = tp / (p + n)
        dict["precision"] = tp / (tp + fp)
        dict["recall"] = tp / p

        self.acc = (tp + tn) / (p + n)

        return dict


def active_Perceptron(X="-1", y=None):
    model = pro()
    model.fit(X, y)
    model.predict(X)
    # print(test.score(X, y))
    return model.model


def test_Perceptron(X_training, y_training, X_testing, y_testing):
    model = pro()
    model.fit(X_training, y_training)
    model.predict(X_testing)
    model.score(X_testing, y_testing)
    return model


x_train = np.array([[2.21469809, 0.18414353, 1.58183177, 1.60251354, -0.9116918],
                    [-1.22012343, -0.43593549, 2.70281907, 0.06539615, 0.13730875]])
y_train = np.array([1., 1., -1., 1., -1.])

x_test = np.array([[2.53195871e+00, 1.10046851e-01, -1.13577822e-03, 1.08546134e+00, -2.50084770e+00, -8.36300113e-01,
                    5.98013011e-01, 1.60480953e+00, -4.76271997e-01, -1.27674670e+00],
                   [-2.76508911e+00, -1.17663856e+00, -1.44934922e+00, -2.67507925e+00, 7.61061749e-01, 1.54165722e+00,
                    2.78953862e+00, -1.72657390e+00, -6.65335689e-01, -1.95178058e-01]])

y_test = np.array([-1., 1., 1., 1., -1., -1., -1., 1., -1., -1.])

perceptron = pro()
perceptron.fit(x_train, y_train)
perceptron.predict(x_test)
perceptron.score(x_test, y_test)
accuracy = perceptron.acc
print(x_train)
print("\n--------------------------------")
print(y_train)
print("\n--------------------------------")
print(x_test)
print("\n--------------------------------")
print(y_test)
print("\n--------------------------------")
print(accuracy)
