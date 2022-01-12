import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import ceil


def fit_linear_regression(X, y):
    X_dagger = np.linalg.pinv(X)

    w = np.matmul(X_dagger.T, y)

    Sigma = np.linalg.svd(X, compute_uv=False)

    return w, Sigma


def predict(X, w):
    return np.matmul(X.T, w)


def mse(response_v, prediction_v):
    lenght = len(response_v)
    square_sum = []

    for i in range(lenght):
        square_sum.append(np.power(response_v[i] - prediction_v[i], 2))

    return np.mean(square_sum)


def load_data(path):
    rows_set = set()
    data = pd.read_csv(path, index_col=False)
    data = data.drop(columns=['id', 'date', 'lat', 'long', 'sqft_living15', 'sqft_lot15'])

    data = data.fillna(0)

    titles = data.columns.values
    RowToDelete = []
    RowsToDelete = []

    for title in titles:  # finds all rows which has Null value in them

        RowToDelete += list(data[data[title].apply(np.isnan)].index)

        if title == "price":
            RowToDelete += list(data[data[title] <= 0].index)

        elif title == "sqft_living" or title == "sqft_lot":
            RowToDelete += list(data[data[title] <= 0].index)

        elif title == "floor":
            RowToDelete += list(data[data[title] < 1].index)

        elif title == "view":
            RowToDelete += list(data[data[title] < 0].index)
            RowToDelete += list(data[data[title] > 4].index)

        elif title == "condition":
            RowToDelete += list(data[data[title] < 1].index)
            RowToDelete += list(data[data[title] > 5].index)

        elif title == "grade":
            RowToDelete += list(data[data[title] < 1].index)
            RowToDelete += list(data[data[title] > 13].index)

        elif title == "sqft_above" or title == "sqft_basement":
            RowToDelete += list(data[data[title] < 0].index)

        RowsToDelete += RowToDelete

    for row in RowsToDelete:
        rows_set.add(row)

    # print(rows_set)
    data.drop(rows_set, inplace=True)
    data.reset_index(inplace=True, drop=True)  # squeeze the rows to fill the blanks of the droped ones

    price_vector = data.pop('price')

    dummies = pd.get_dummies(data['zipcode'])
    data.drop(columns=['zipcode'], inplace=True)
    data = data.join(dummies)
    # print(data.T)
    return data.T, price_vector


def plot_singular_values(singular_arr):
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title("the MSE over the test set as a function of p%")
    plt.xlabel("p%")
    plt.ylabel("MSE")
    ax.plot(singular_arr)

    plt.show()


def feature_evaluation(design_matrix, response_vector):
    titles = design_matrix.T.columns.values
    sorted_titles = []

    for title in titles:
        if type(title) is str:
            sorted_titles.append(title)

    for title in sorted_titles:
        fig = plt.figure()
        ax = fig.add_subplot()

        if title == "yr_renovated":  # screen out all the house which wasn't renovated to get a clearer results
            minimum = min(i for i in design_matrix.T[title].to_numpy() if i > 0)
            max = sorted(design_matrix.T[title].to_numpy())[-1]
            ax.set_xlim(int(minimum), int(max))

        pearson_correlation = np.cov(design_matrix.T[title], response_vector) / (
                    np.std(design_matrix.T[title]) * np.std(response_vector))

        plt.title("price over %s" % title + "\n" + "Pearson Correlation : %s" % pearson_correlation[0][1])
        plt.ylabel("price")
        plt.xlabel(title)
        ax.plot(design_matrix.T[title], response_vector, ".")
        plt.show()


def q_15(path):
    processed_data, prices_vector = load_data(path)
    vec, sigma = fit_linear_regression(processed_data, prices_vector)
    plot_singular_values(sigma)


def q_16(path):
    processed_data, prices_vector = load_data(path)
    x_train, x_test, y_train, y_test = train_test_split(processed_data.T.to_numpy(), prices_vector.to_numpy())

    results_mse = []

    for p in range(1, 101):
        # print(p)
        percent = ceil((p / 100) * y_train.shape[0])

        vec_train, sigma_train = fit_linear_regression(x_train[0:percent].T, y_train[0:percent])

        predictes = predict(x_test.T, vec_train)
        results_mse.append(mse(y_test, predictes))

    plot_singular_values(results_mse)


def q_17(path):
    processed_data, prices_vector = load_data(path)
    feature_evaluation(processed_data, prices_vector)


# q_15("C:\IML programing\kc_house_data.csv")
# q_16("C:\IML programing\kc_house_data.csv")
# q_17("C:\IML programing\kc_house_data.csv")
