import numpy as np
from matplotlib import pyplot as plt
import utils.metrcis as metrics
from utils.classifications import random_classifier, height_classifier

football_player_height = np.random.randn(500) * 20 + 160
basketball_player_height = np.random.randn(500) * 10 + 190

football_player_data = np.column_stack((football_player_height, np.zeros(500)))
basketball_player_data = np.column_stack((basketball_player_height, np.ones(500)))
data = np.concatenate((football_player_data, basketball_player_data))
np.random.shuffle(data)

height = 175

confusion_matrix = {'height': [[0, 0], [0, 0]],
                    'random': [[0, 0], [0, 0]]}

for i in range(len(data)):
    confusion_matrix['random'][random_classifier()][int(data[i, 1])] += 1
    confusion_matrix['height'][height_classifier(data[i, 0], height)][int(data[i, 1])] += 1

print(f"Random\n"
      f"Accuracy: {metrics.accuracy(confusion_matrix['random'][1][1], confusion_matrix['random'][0][0], data.shape[0])}\n"
      f"Precision: {metrics.precision(confusion_matrix['random'][1][1], confusion_matrix['random'][1][0])}\n"
      f"Recall: {metrics.recall(confusion_matrix['random'][1][1], confusion_matrix['random'][0][1])}\n")

print(f"Height {height}\n"
      f"Accuracy: {metrics.accuracy(confusion_matrix['height'][1][1], confusion_matrix['height'][0][0], data.shape[0])}\n"
      f"Precision: {metrics.precision(confusion_matrix['height'][1][1], confusion_matrix['height'][1][0])}\n"
      f"Recall: {metrics.recall(confusion_matrix['height'][1][1], confusion_matrix['height'][0][1])}")

precisions = []
recalls = []

for height in range(90, 231, 10):
    confusion_matrix = [[0, 0], [0, 0]]

    for i in range(len(data)):
        confusion_matrix[height_classifier(data[i, 0], height)][int(data[i, 1])] += 1

    precision = metrics.precision(confusion_matrix[1][1], confusion_matrix[1][0])
    recall = metrics.recall(confusion_matrix[1][1], confusion_matrix[0][1])
    precisions.append(precision)
    recalls.append(recall)

square = 0
for i in range(1, len(recalls)):
    square += (recalls[i] - recalls[i - 1]) * (precisions[i] + precisions[i - 1]) / 2
plt.plot(recalls, precisions)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Square:{square * -1}')
plt.show()
