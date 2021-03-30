import numpy as np
import matplotlib.pyplot as plt


def bar_plot(models_list, bin_count=1, size=(15, 6), rmse_list=['rmse_train', 'rmse_valid'],
             text=['обучающей', 'валидационной']):
    width = 0.2
    confidences = []
    for rmse in rmse_list:
        confidences.append([model[rmse] for model in models_list[:bin_count]])
    labels = []
    for i in range(bin_count):
        weights = models_list[i]['w']
        func_names = models_list[i]['model']
        reg = "".join([f"{w:.2f}*{name}+" for w, name in zip(weights[1:], func_names)]) + f"{weights[0]:.2f}"
        labels.append(reg)
    bin_positions = np.arange(len(confidences[0]))
    fig, ax = plt.subplots(figsize=size)
    bins_arts = []
    for i in range(len(confidences)):
        bins_arts.append(
            ax.bar(bin_positions - 0.1 * (-1) ** i, confidences[i], width, label=f"Точность на {text[i]} выборке"))
    plt.ylabel('Точность')
    plt.title('Точность различных архитектур')
    plt.xticks(bin_positions, labels)
    plt.legend(loc=3)
    for rect in bins_arts:
        for r in rect:
            height = r.get_height()
            plt.text(r.get_x() + r.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom')
    plt.show()
