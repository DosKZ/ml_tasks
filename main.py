import numpy as np
from calculations.normal_distribution import norm_dist
from utils.plots import reg_plot,rmse_plot
from calculations.metrics import RMSE

x = np.linspace(0, 2 * np.pi, 1000)
y = 100 * np.sin(x) + 0.5 * np.exp(x) + 300
err = norm_dist(0,100,1000)
t = y + err

phi = np.array([x ** 0]).T
rmse_list = []
y_new=[]

for i in range(1,21):
    phi = np.concatenate((phi, x.reshape(-1, 1) ** i), axis=1)
    w = np.linalg.inv(np.dot(phi.T, phi)).dot(phi.T).dot(t)
    y_new.append((w * phi).sum(axis=1))
    rmse_list.append(RMSE(t, y_new[i-1]))

reg_plot(x,t,y_new,rmse_list)

rmse_plot(rmse_list)
