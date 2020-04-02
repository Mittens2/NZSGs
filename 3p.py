import numpy as np
import random
import math
import matplotlib
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

# Hyperparameters
n = 10000
alpha = 0.003
d = 10

# Choose dynamics
dg = b, db
opt = oga

# Gradients of players for games

# Gradients for linear game
def db(x, y, z, i):
    if i == 0:
        return A_xz@z + A_xy@y
    elif i == 1:
        return -A_xy.T@x + A_yz@z
    else:
        return -A_xz.T@x - A_yz.T@y

# Gradients for strongly concave and smooth game
def dq(x, y, z, i):
    if i == 0:
        return -x + A_xy@y + A_xz@z
    elif i == 1:
        return -y - A_xy.T@x + A_yz@z
    else:
        return  -z - A_xz.T@x - A_yz.T@y

#  Dynamics

# GA dynamics
def ga(x, y, z, df, i):
    x_t = x[i] + alpha * df(x[i], y[i], z[i], 0)
    y_t = y[i] + alpha * df(x[i], y[i], z[i], 1)
    z_t = z[i] + alpha * df(x[i], y[i], z[i], 2)
    return x_t, y_t, z_t

# OGA dynamics
def oga(x, y, z, df, i):
    x_t =  x[i] + 2 * alpha * df(x[i], y[i], z[i], 0) - alpha * df(x[i-1], y[i-1], z[i-1], 0)
    y_t =  y[i] + 2 * alpha * df(x[i], y[i], z[i], 1) - alpha * df(x[i-1], y[i-1], z[i-1], 1)
    z_t =  z[i] + 2 * alpha * df(x[i], y[i], z[i], 2) - alpha * df(x[i-1], y[i-1], z[i-1], 2)
    return x_t, y_t, z_t

# Averaging
def avg(x):
    x_bar = np.zeros(x.shape)
    for i in range(len(x)):
        x_bar[i] = (x[i] + x_bar[i - 1] * i) / (i + 1)
    return x_bar

# Initialize parameters
x = np.zeros((n, d))
y = np.zeros((n, d))
z = np.zeros((n, d))
x[0] = np.random.uniform(low=-1, high=1, size=d)
y[0] = np.random.uniform(low=-1, high=1, size=d)
z[0] = np.random.uniform(low=-1, high=1, size=d)
x[1] = x[0]
y[1] = y[0]
z[1] = z[0]
A_xy = np.random.rand(d,d)
A_xz = np.random.rand(d,d)
A_yz = np.random.rand(d,d)
c_x = np.random.uniform()
c_y = np.random.uniform()
c_z = np.random.uniform()

fig = plt.figure()
ax1 = fig.add_subplot(211, projection='3d')
fig.subplots_adjust(hspace=.5)
ax2 = fig.add_subplot(212)

for i in range(0, n - 1):
    x_t, y_t, z_t = opt(x, y, z, dg, i)
    x[i + 1] = x_t
    y[i + 1] = y_t
    z[i + 1] = z_t
x_bar, y_bar, z_bar = avg(x), avg(y), avg(z)
x_plot, y_plot, z_plot = np.sum(x[1:] ** 2, axis=1), np.sum(y[1:] ** 2, axis=1), np.sum(z[1:] ** 2, axis=1)
ax1.plot(xs=x_plot, ys=y_plot, zs=z_plot, label='last')
ax1.plot(xs=np.sum(x_bar[1:] ** 2, axis=1), ys=np.sum(y_bar[1:] ** 2, axis=1), zs=np.sum(z_bar[1:] ** 2, axis=1), label='avg', color='orange')
ax2.plot(np.arange(n-1), np.sum(x[1:] ** 2, axis=1) + np.sum(y[1:] ** 2, axis=1) + np.sum(z[1:] ** 2, axis=1), label='last')
ax2.plot(np.arange(n-1), np.sum(x_bar[1:] ** 2, axis=1) + np.sum(y_bar[1:] ** 2, axis=1) + np.sum(z_bar[1:] ** 2, axis=1), label='avg')

ax1.locator_params(nbins=5)
ax1.title.set_text('3D trajectory')
ax1.legend(loc='upper left')
ax1.set_xlabel('x1', labelpad=10)
ax1.set_ylabel('x2')
ax1.set_zlabel('x3')
ax2.legend(loc='upper right')
ax2.title.set_text('Distance to Equillibrium')
ax2.set_xlabel('iterates')
ax2.set_ylabel('sum L2')

plt.show()
