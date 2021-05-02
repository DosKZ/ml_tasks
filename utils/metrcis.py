def get_type(predict, true):
    if predict == 1 and true == 1:
        return 'TP'
    elif predict == 0 and true == 0:
        return 'TN'
    elif predict == 1 and true == 0:
        return 'FP'
    elif predict == 0 and true == 1:
        return 'FN'


def accuracy(TP, TN, n):
    return (TP + TN) / n


def precision(TP, FP):
    try:
        return TP / int(TP + FP)
    except ZeroDivisionError:
        return 1


def recall(TP, FN):
    try:
        return TP / (TP + FN)
    except ZeroDivisionError:
        return 0
