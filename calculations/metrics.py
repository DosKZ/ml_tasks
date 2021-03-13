import numpy as np


def MSE(t, y_new):
    return ((t - y_new) ** 2).sum() / t.shape[0]


def RMSE(t, y_new):
    return np.sqrt(MSE(t, y_new))
