import numpy as np
import matplotlib.pyplot as plt

COIN_TOSSES = 1000
REVERSALS = 100000
P = 0.25

data = np.random.binomial(1, 0.25, (REVERSALS, COIN_TOSSES))
epsilon = [0.5, 0.25, 0.1, 0.01, 0.001]


def calculate_mean_for_vec_i(i):
    '''
    calculates the mean of a specific vector i
    :param i: the vector in the data of which it's mean we want
    :return: the estimated means along the vector
    '''

    # print(i/REVERSALS * 100, "%")  # visual Aid

    estimated_means = [0]

    for j in range(1, COIN_TOSSES):
        estimated_means.append(np.average(data[i][0:j]))

    return estimated_means


def draw_means_plot():  # question a
    '''
    calculates and plot the means of the first 5 vectors
    '''

    for i in range(5):
        plt.plot(range(0, 1000), calculate_mean_for_vec_i(i))

    plt.title("Estimated mean over many tosses")
    plt.xlabel("Number of tosses")
    plt.ylabel("Mean")
    plt.show()


def calculate_Chebyshev(epsilon):
    '''
    calculates the Chebyshev bound under different epsilons
    :param epsilon: the specific epsilon used as bound parameter
    :return: a list with all the bounds along the vector
    '''
    chebyshev_bonuds = []
    epsilon_power = epsilon ** 2

    for toss in range(COIN_TOSSES):
        bound = 1 / (4 * (toss + 1) * epsilon_power)  # as learned in lecture 1
        if bound > 1: bound = 1
        chebyshev_bonuds.append(bound)

    return chebyshev_bonuds


def calculate_Hoeffding(epsilon):  # question b
    '''
    calculates the Hoeffding bound under different epsilons
    :param epsilon: the specific epsilon used as bound parameter
    :return: a list with all the bounds along the vector
    '''

    hoeffding_bonuds = []
    power_epsilon = epsilon ** 2

    for num in range(COIN_TOSSES):
        bound = 2 * np.math.exp(-2 * num * power_epsilon)
        if bound > 1: bound = 1
        hoeffding_bonuds.append(bound)

    return hoeffding_bonuds


def plot_question_b(question_c=False):
    '''
    manage the hoeffding and chebyshev bonuds, and plot them together side by side.
    :param question_c: in case we wanr to plot the percentage like in question c, this
            parameter should be checked True.
    :return: a graph with hoeffding and chebyshev bonuds ploted on.
    '''

    if question_c:
        means = []
        for i in range(REVERSALS):
            means.append(calculate_mean_for_vec_i(i))

    for e in epsilon:

        hoeffding_bonuds = calculate_Hoeffding(e)
        chebyshev_bonuds = calculate_Chebyshev(e)

        plt.plot(range(0, COIN_TOSSES), hoeffding_bonuds, label="Hoeffding bound")
        plt.plot(range(0, COIN_TOSSES), chebyshev_bonuds, label="Chebyshev bound")

        plot_title = "Hoeffding and Chebyshev over many tosses with epsilon = " + str(e) + ":"

        if question_c:
            plot_question_c(e, means)
            plot_title = "Percentage, " + plot_title

        plt.title(plot_title)
        plt.xlabel("Number of tosses")
        plt.ylabel("bound")
        plt.legend()
        plt.show()

        plt.cla()
        plt.clf()

    return


def plot_question_c(e, means):
    '''
    calculate the percentage of means the stands in the condition of greater than epsilon
    :param e: the specific epsilon used as bound parameter
    :param means: a list of all the means along all the vector in the data matrix.
    '''

    percentage_list = [0] * COIN_TOSSES
    # print("-------------------------------------- ", e, " -------------------------------------")

    for vec_index, vec in enumerate(means):

        # print(vec_index/REVERSALS * 100, "%")  # visual Aid

        for toss_index, toss in enumerate(vec):

            if abs(toss - P) >= e:
                percentage_list[toss_index] += 1 / REVERSALS

    plt.plot(range(0, COIN_TOSSES), percentage_list, label="Percentage")

    return
