import numpy as np
from calculations.normal_distribution import norm_dist
from utils.plots import bar_plot
from calculations.metrics import RMSE
from itertools import combinations
from utils.parsers import train_valid_test, split_x_y


def func_from_str(f, x):
    return eval(f)


x = np.linspace(10 ** -10, 2 * np.pi, 1000)
y = 100 * np.sin(x) + 0.5 * np.exp(x) + 300
err = norm_dist(0, 100, 1000)
t = y + err

functions = np.array(['np.sin(x)', 'np.cos(x)', 'np.log(x)', 'np.exp(x)', 'np.sqrt(x)', 'x', 'x**2', 'x**3'])

tr_data, val_data, test_data = train_valid_test(x, t)
x_train, y_train = split_x_y(tr_data)
x_valid, y_valid = split_x_y(val_data)
x_test, y_test = split_x_y(test_data)

models_list = []

for i in range(1, 4):
    for comb in combinations(range(len(functions)), i):
        phi = np.power(x_train,0).reshape(-1, 1)
        for k in comb:
            phi = np.concatenate((phi, func_from_str(functions[k], x_train.reshape(-1, 1))), axis=1)
        w = np.linalg.inv(np.dot(phi.T, phi)).dot(phi.T).dot(y_train)
        new_y_train = (w * phi).sum(axis=1)
        rmse_train = RMSE(y_train, new_y_train)

        phi = np.power(x_valid, 0).reshape(-1, 1)
        for k in comb:
            phi = np.concatenate((phi, func_from_str(functions[k], x_valid.reshape(-1, 1))), axis=1)
        new_y_valid = (w * phi).sum(axis=1)
        rmse_valid = RMSE(y_valid, new_y_valid)

        model = {}
        model['model'] = functions[list(comb)]
        model['w'] = w
        model['rmse_train'] = rmse_train
        model['rmse_valid'] = rmse_valid
        models_list.append(model)
models_list = sorted(models_list, key=lambda x: x['rmse_valid'])
print(models_list[0])

comb=models_list[0]['model']
w=models_list[0]['w']
phi = np.power(x_test, 0).reshape(-1, 1)
for k in comb:
    phi = np.concatenate((phi, func_from_str(k, x_test.reshape(-1, 1))), axis=1)
new_y_test = (w * phi).sum(axis=1)
rmse_test = RMSE(y_test, new_y_test)

model={}
model['model'] = comb
model['w'] = w
model['rmse_train']=models_list[0]['rmse_train']
model['rmse_test'] = rmse_test



bar_plot(models_list,bin_count=3)
bar_plot([model], bin_count=1, size=(6, 4), rmse_list=['rmse_train','rmse_test'], text=['обучающей','тестовой'])
