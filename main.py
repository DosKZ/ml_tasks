import numpy as np
from matplotlib import pyplot as plt
from calculations.metrics import MSE
from utils.Matrix import design_matrix
from utils.gradient_descent import grad_descent
from utils.parsers import split_x_y, split_train_valid_test
from utils.plots import bar_plot

# eps в критерии выхода
eps = 0.00001
eps0 = 0.000001
gamma = 0.005  # величина шага в градиентном спуске
points = 50  # количество данных всего
poly_deg = 7  # степень нашего полинома
lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 5, 10, 20, 30, 40, 50, 60, 70]

# генерация данных
x = np.linspace(0, 1, points)
gt = 10 * x * x + 20 * np.sin(10 * x)
err = 2 * np.random.randn(points)
err[8] -= 80
err[25] += 250
err[41] -= 200
t = gt + err

tr_data, val_data, test_data = split_train_valid_test(x, t, gt)
x_train, t_train = split_x_y(tr_data)
x_valid, t_valid = split_x_y(val_data)
x_test, t_test = split_x_y(test_data)

confidences = [[], []]
weights = []

for lmbd in lambdas:
    loss_value, w_grad = grad_descent(x_train, t_train, poly_deg, gamma, lmbd, eps, eps0)
    print(f"Lambda:{lmbd} | Final loss:{loss_value}")
    confidences[0].append(MSE(t_train, w_grad.dot(design_matrix(x_train, poly_deg).T)))
    confidences[1].append(MSE(t_valid, w_grad.dot(design_matrix(x_valid, poly_deg).T)))
    weights.append(w_grad)
bar_plot(confidences, lambdas, len(lambdas))

phi = design_matrix(x_train, poly_deg)
w = np.linalg.inv(np.dot(phi.T, phi)).dot(phi.T).dot(t_train)

index = sorted(enumerate(confidences[1]), key=lambda x: x[1])[0][0]
w_best = weights[index]

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x, t, 'bo', label='исходный', markersize=1)
ax1.plot(x, w.dot(design_matrix(x, poly_deg).T), 'r-', label='полином')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend(fontsize=5, loc='lower right')
ax1.set_title(
    f"Прямое решение\n"
    f"Ошибка на тестовой: {np.rint(MSE(t_test, w.dot(design_matrix(x_test, poly_deg).T))).astype(int)}"
)
ax2.plot(x, t, 'bo', label='исходный', markersize=1)
ax2.plot(x, w_best.dot(design_matrix(x, poly_deg).T), 'r-', label='полином')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.legend(fontsize=5, loc='lower right')
ax2.set_title(
    f"lambda:{lambdas[index]}\n"
    f"Ошибка на тестовой: {np.rint(MSE(t_test, w_best.dot(design_matrix(x_test, poly_deg).T))).astype(int)}"
)
plt.subplots_adjust(wspace=0.8)
plt.show()
