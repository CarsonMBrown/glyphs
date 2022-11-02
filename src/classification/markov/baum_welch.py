import numpy as np
import pandas as pd


def forward(V, a, b, initial_distribution):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, V[0]]

    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            alpha[t, j] = alpha[t - 1] @ a[:, j] * b[j, V[t]]

    return alpha


def backward(V, a, b):
    beta = np.zeros((V.shape[0], a.shape[0]))

    # setting beta(T) = 1
    beta[V.shape[0] - 1] = np.ones((a.shape[0]))

    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]) @ a[j, :]

    return beta


def baum_welch(O, a, b, initial_distribution, n_iter=100):
    # http://www.adeveloperdiary.com/data-science/machine-learning/derivation-and-implementation-of-baum-welch-algorithm-for-hidden-markov-model/
    M = a.shape[0]
    T = len(O)
    for n in range(n_iter):
        # estimation step
        alpha = forward(O, a, b, initial_distribution)
        beta = backward(O, a, b)
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            # joint probability of observed data up to time t @ transition prob *
            # emisssion prob at t+1 @ joint probab of observed data from at t+1
            denominator = (alpha[t, :].T @ a * b[:, O[t + 1]].T) @ beta[t + 1, :]
            for i in range(M):
                numerator = alpha[t, i] * a[i, :] * b[:, O[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator
        gamma = np.sum(xi, axis=1)
        # maximization step
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, O == l], axis=1)
        b = np.divide(b, denominator.reshape((-1, 1)))
    return a, b


data = pd.read_csv('data_python.csv.txt')

V = data['Visible'].values

# Transition Probabilities
a = np.ones((2, 2))
a = a / np.sum(a, axis=1)

# Emission Probabilities
b = np.array(((1, 3, 5), (2, 4, 6)))
b = b / np.sum(b, axis=1).reshape((-1, 1))

# Equal Probabilities for the initial distribution
initial_distribution = np.array((0.5, 0.5))

n_iter = 100
a_model, b_model = baum_welch(V.copy(), a.copy(), b.copy(), initial_distribution.copy(), n_iter=n_iter)
print(f'Custom model A is \n{a_model} \n \nCustom model B is \n{b_model}')
