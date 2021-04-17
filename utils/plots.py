import numpy as np
import matplotlib.pyplot as plt


def reg_plot(x, t, y, mse_list, wn, size=[2, 5, (10, 10)]):
    fig, axes = plt.subplots(size[0], size[1], figsize=size[2])
    k = 1
    for row in range(size[0]):
        for column in range(size[1]):
            axes[row, column].plot(x, t, 'bo', label='исходный', linewidth=0.5, markersize=0.3)
            axes[row, column].plot(x, y[k - 1], 'r-', label='полином', )
            axes[row, column].set_xlabel('X')
            axes[row, column].set_ylabel('Y')
            axes[row, column].legend(fontsize=5, loc='lower right')
            axes[row, column].set_title(f"{wn[k - 1]} итерация\n Ошибка: {np.rint(mse_list[k - 1]).astype(int)}")
            k += 1
    plt.subplots_adjust(wspace=1.2, hspace=0.5)
    plt.show()


def bar_plot(confidences, lambdas, bin_count=1, size=(15, 6)):
    width = 0.2
    labels = []
    bins_arts = []
    for i in range(bin_count):
        labels.append("".join(f"{lambdas[i]}"))
    bin_positions = np.arange(len(confidences[0]))
    fig, ax = plt.subplots(figsize=size)
    for i in range(len(confidences)):
        bins_arts.append(
            ax.bar(bin_positions - 0.1 * (-1) ** i, confidences[i], width))
    plt.ylabel('Ошибка')
    plt.title('Ошибки различных архитектур')
    plt.xticks(bin_positions, labels)
    plt.legend(loc=3)
    for rect in bins_arts:
        for r in rect:
            height = r.get_height()
            plt.text(r.get_x() + r.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom')
    plt.show()
