import numpy as np

from calculations.metrics import MSE
from utils.Matrix import design_matrix
from utils.plots import reg_plot


def loss(x, t, w, n):
    return np.power(t - np.dot(w, design_matrix(x, n).T), 2).sum() / 2

def gradient(x, t, w, lamb, n):
    grad = np.dot((t - np.dot(w, design_matrix(x, n).T)), design_matrix(x, n))
    return -grad + lamb * w

def grad_descent(x, t, poly_deg, step, lmbd, eps, eps0):
    n=np.arange(0,4501,500)
    w_next = 10000 * np.random.rand(poly_deg + 1)
    cant_stop = True
    count = 0
    y_new=[]
    mse_list=[]
    while cant_stop:
        w_old = w_next
        if lmbd == 0 and np.where(n==count):
            y_new.append(w_old.dot(design_matrix(x, poly_deg).T))
            mse_list.append(MSE(t, w_old.dot(design_matrix(x, poly_deg).T)))
        w_next = w_old - step * gradient(x, t, w_old, lmbd, poly_deg)
        loss_value = loss(x, t, w_next, poly_deg)
        if np.linalg.norm(w_next - w_old) < eps * (np.linalg.norm(w_next) + eps0):
            cant_stop = False
        count += 1
    if lmbd==0:
        reg_plot(x,t,y_new,mse_list,n)
    return loss_value, w_next