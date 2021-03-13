import numpy as np
import matplotlib.pyplot as plt

def reg_plot(x,t,y,rmse_list):
    fig, axes = plt.subplots(5, 4, figsize=(10, 10))
    n = 1
    for row in range(5):
        for column in range(4):
            axes[row, column].plot(x, t, 'b-', label='исходный',linewidth=0.3)
            axes[row, column].plot(x, y[n - 1], 'r-', label='gолином')
            axes[row, column].set_xlabel('X')
            axes[row, column].set_ylabel('Y')
            axes[row, column].legend(fontsize=5,loc='lower right')
            axes[row, column].set_title('Power {0} \n RMSE = {1}'.format(n, np.rint(rmse_list[n - 1])))
            n += 1
    plt.ylim((-50, 700))
    plt.subplots_adjust(wspace=0.5, hspace=1.5)
    plt.show()

def rmse_plot(rmse_list):
    x = np.arange(1, len(rmse_list)+1)
    y = np.array(rmse_list)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.ylim((0, 30000))
    ax.set_xlabel('Max power')
    ax.set_ylabel('RMSE')
    plt.show()