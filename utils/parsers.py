import numpy as np


def split_train_valid_test(x, t, gt, train_size=0.8, valid_size=0.1, test_size=0.1):
    data = np.concatenate((x.reshape(-1, 1), t.reshape(-1, 1)), axis=1)
    mask = np.array(
        sorted(np.concatenate((np.arange(len(data)).reshape(-1, 1), np.abs(data[:, 1] - gt).reshape(-1, 1)), axis=1),
               key=lambda x: x[1]))[::-1]

    data = data[np.rint(mask[:, 0]).astype(int).T]
    l_train = int(len(data) * train_size)
    l_valid = int(len(data) * valid_size)
    l_test = int(len(data) * test_size)
    return data[:l_train], data[l_train:l_train + l_valid], data[l_train + l_test:]


def split_x_y(data):
    return data[:, 0], data[:, 1]
