import random as rand
from utils.file import write_to_file, read_from_file
from calculations.normal_distribution import norm_dist

file = "Mu n' Sigma.txt"

write_to_file(file, rand.randint(5, 87), rand.randint(60, 89))

mu, sigma = read_from_file(file)
print("μ = {0}\nσ = {1}".format(mu, sigma))

X = norm_dist(1000, mu, sigma)
print(X)