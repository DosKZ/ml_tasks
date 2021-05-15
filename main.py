from sklearn import datasets
from Softmax import SoftmaxRegression

classifier = SoftmaxRegression(9)
digits = datasets.load_digits()
data = digits.data
labels = digits.target

train_data, train_labels, validation_data, validation_labels, test_data, test_labels = classifier.preprocess(data,
                                                                                                             labels)
batch_size = 18
iterations = 5000
save_weights = False
classifier.train(train_data, train_labels, validation_data, validation_labels, batch_size, iterations, save_weights)

load_weigths_from = 'weights.pickle'
classifier.test_classifier(test_data, test_labels, load_weigths_from)
