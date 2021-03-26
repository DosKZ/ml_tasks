import numpy as np


def train_valid_test(x, y, train_size=0.8, valid_size=0.1, test_size=0.1):
    data = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    np.random.shuffle(data)
    l_train = int(len(data) * train_size)
    l_valid = int(len(data) * valid_size)
    l_test = int(len(data) * test_size)
    return data[:l_train], data[l_train:l_train + l_valid], data[l_train + l_test:]


def split_x_y(data):
    return data[:, 0], data[:, 1]
