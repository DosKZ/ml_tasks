import numpy as np


def design_matrix(x, n):
    phi = (x ** 0).reshape(-1, 1)
    for i in range(1, n + 1):
        phi = np.concatenate((phi, x.reshape(-1, 1) ** i), axis=1)
    return phi
