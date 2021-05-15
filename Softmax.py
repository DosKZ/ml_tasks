import pickle
import numpy as np
from matplotlib import pyplot as plt


class SoftmaxRegression:
    def __init__(self, classes_count=10, lr=0.001):
        self.classes_count = classes_count
        self.lr = lr

    def train(self, data, labels, val_data, val_labels, batch_size, iters, save_to_file=True):
        val_losses = []
        losses = []
        weights = np.zeros((data.shape[1], self.classes_count))
        biasies = np.zeros(self.classes_count)
        val_iters = np.arange(0, iters + 30, 30)
        for i in range(iters):
            if i in val_iters:
                val_losses.append(
                    self.__loss(self.__predict(val_data, weights, biasies), self.__code(val_labels), val_data.shape[0]))
            data_batch, labels_batch = self.__batch_loader(data, labels, batch_size, i)
            labels_batch = self.__code(labels_batch)
            predictions = self.__predict(data_batch, weights, biasies)
            softmax_predictions = self.__softmax(predictions)
            losses.append(self.__loss(predictions, labels_batch, batch_size))
            gradient = data_batch.T @ (softmax_predictions - labels_batch) / batch_size
            weights -= self.lr * gradient + self.lr * weights
            biasies -= self.lr * np.sum(softmax_predictions - labels_batch, axis=0) / batch_size

        if save_to_file:
            self.__save_weights(weights, biasies)
        # print(np.max(losses))
        self.__print_loss(losses, np.arange(len(losses)), title='Train loss')
        self.__print_loss(val_losses, val_iters[:-1], 'Validation loss')
        self.weights = weights
        self.biases = biasies

    def test_classifier(self, data, labels, filename=''):
        if filename:
            self.__load_weights(filename)
        prediction = self.__predict(data, self.weights, self.biases)
        classification = np.argmax(self.__softmax(prediction), axis=1)

        self.__confusion_matrix(classification, labels)
        print(f'Confusion matrix:\n{self.conf_matrix}\n')

        precisions, recalls = self.__precisions_recalls()
        print(f'Precisions:\n{precisions}\n')
        print(f'Recalls:\n{recalls}')

    def __code(self, gt):
        one_hot_vec = np.zeros((len(gt), self.classes_count))
        one_hot_vec[np.arange(len(one_hot_vec)), gt.astype(int)] = 1.0
        return one_hot_vec

    @staticmethod
    def __softmax(data):
        z = data - np.max(data, axis=1, keepdims=True)
        predictions_vec = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        return predictions_vec

    @staticmethod
    def __normalize(data):
        data[:, :-1] = ((data[:, :-1] - np.min(data[:, :-1])) / (np.max(data[:, :-1]) - np.min(data[:, :-1])))
        return data

    @staticmethod
    def __loss(y_pred, y_true, batch_size):
        z = y_pred - np.max(y_pred, axis=1, keepdims=True)
        loss = (-y_true * (z - np.log(np.exp(z).sum(axis=1, keepdims=True)))).sum(axis=0)
        return loss.mean() / batch_size

    @staticmethod
    def __train_val_test_split(data, train_size=0.8, valid_size=0.1, test_size=0.1):
        train_len = int(len(data) * train_size)
        valid_len = int(len(data) * valid_size)
        test_len = int(len(data) * test_size)

        train_data = data[: train_len]
        validation_data = data[train_len: train_len + valid_len]
        test_data = data[train_len + test_len:]

        train_labels = train_data[:, -1]
        validation_labels = validation_data[:, -1]
        test_labels = test_data[:, -1]

        train_data = train_data[:, :-1]
        validation_data = validation_data[:, :-1]
        test_data = test_data[:, :-1]

        return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

    def preprocess(self, data, labels):
        data = np.concatenate((data, labels.reshape(-1, 1)), axis=1)
        np.random.shuffle(data)
        data = self.__normalize(data)
        data = data[data[:, -1] < self.classes_count]
        train_data, train_labels, validation_data, validation_labels, test_data, test_labels = self.__train_val_test_split(
            data)
        return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

    @staticmethod
    def __batch_loader(data, labels, batch_size, iter):
        if iter != 0:
            if (int(data.shape[0] / batch_size)) < iter:
                iter = iter % int(data.shape[0] / batch_size)
        frst = iter * batch_size
        scnd = frst + batch_size
        return data[frst:scnd], labels[frst:scnd]

    @staticmethod
    def __save_weights(weights, biasies):
        dictionary = {'weights': weights, 'biasies': biasies}
        with open('weights.pickle', 'wb') as f:
            pickle.dump(dictionary, f)

    def __load_weights(self, filename):
        with open(filename, "rb") as f:
            dictionary = pickle.load(f)
        self.weights = dictionary['weights']
        self.biases = dictionary['biasies']

    @staticmethod
    def __predict(x, weights, biasies):
        return x @ weights + biasies

    def __confusion_matrix(self, y_pred, y_true):
        self.conf_matrix = np.zeros((self.classes_count, self.classes_count))
        for i in range(len(y_pred)):
            self.conf_matrix[int(y_pred[i]), int(y_true[i])] += 1

    def __precisions_recalls(self):
        precisions = []
        recalls = []
        for i in range(self.classes_count):
            precision = self.conf_matrix[i, i] / np.sum(self.conf_matrix[i, :])
            precisions.append(precision)
            recall = self.conf_matrix[i, i] / np.sum(self.conf_matrix[:, i])
            recalls.append(recall)
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        return precisions, recalls

    @staticmethod
    def __print_loss(losses, losses_range=None, title=None):
        plt.plot(losses_range, losses, linewidth=0.4)
        plt.title(title)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.show()
