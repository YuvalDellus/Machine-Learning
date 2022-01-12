import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr


mean = [0, 0, 0]
cov = np.eye(3)
x_y_z = np.random.multivariate_normal(mean, cov, 50000).T

S = np.matrix('0.1, 0, 0; 0, 0.5, 0; 0, 0, 2')


def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q


def plot_3d(x_y_z):
    '''
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    plt.title("Transformed Matrix after multiplying in orthogonal matrix")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def plot_2d(x_y):
    """
    plot points in 2D
    :param x_y_z: the points. numpy array with shape: 2 X num_samples
                  (first dimension for x, y coordinate)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter([x_y[0]], [x_y[1]], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.show()


def question_11():
    '''
    generates random matrix with 50000 entries
    '''

    x_y_z_11 = np.random.multivariate_normal(mean, cov, 50000).T
    plot_3d(x_y_z_11)

    return


def question_12():
    '''
    multiply the data matrix by scalar matrix.
    '''

    x_y_z_12 = S * np.random.multivariate_normal(mean, cov, 50000).T
    plot_3d(x_y_z_12)

    return


def question_13():
    '''
    multiply the streched data matrix by orthogonal matrix.
    '''

    Q = get_orthogonal_matrix(3)
    x_y_z_13 = Q * S * np.random.multivariate_normal(mean, cov, 50000).T
    plot_3d(x_y_z_13)

    return


def question_14():
    '''
    project the matrix on the 2D plain
    :return:
    '''

    projection_operator = np.matrix('1, 0, 0; 0, 1, 0; 0, 0, 0')
    Q = get_orthogonal_matrix(3)
    x_y_14 = projection_operator * Q * S * np.random.multivariate_normal(mean, cov, 50000).T

    plot_2d(x_y_14)

    return


def question_15():
    '''
    filter from the projected matrix only the points that stand in the condition -0.4 < z < 0.1
    '''

    projection_operator = np.matrix('1, 0, 0; 0, 1, 0; 0, 0, 0')
    Q = get_orthogonal_matrix(3)
    x_y_15 = projection_operator * Q * S * np.random.multivariate_normal(mean, cov, 50000).T

    final = x_y_15[0:2, np.where((-0.4 < x_y_15[2]) & (x_y_15[2] < 0.1))]
    plot_2d(final)

    return


def main():
    pass

main()
