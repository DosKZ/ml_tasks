import random


def random_classifier():
    return random.randint(0, 1)


def height_classifier(x, height):
    if x > height:
        return 1
    else:
        return 0
