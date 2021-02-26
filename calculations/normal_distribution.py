import numpy as np

def norm_dist(size, mu, sigma):
    return np.random.normal(mu, sigma, size)


if __name__=="__main__":
    X = norm_dist(1000,5,10)
    print(X)