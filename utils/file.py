def write_to_file(filepath, mu, sigma):
    file = open(filepath, 'w')
    str = "mu: {0}\nsigma: {1}".format(mu, sigma)
    file.write(str)


def read_from_file(filepath):
    file = open(filepath, 'r').read().split("\n")
    temp = []
    for line in file:
        temp.append(line.split()[1])
    return int(temp[0]), int(temp[1])


if __name__=="__main__":
    write_to_file("test.txt",1,1)
    mu, sigma = read_from_file("test.txt")
    print(mu,sigma)
