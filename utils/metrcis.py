
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
        print('Данные не содержат класс 1')
