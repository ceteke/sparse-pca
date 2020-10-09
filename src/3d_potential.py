import numpy as np
import h5py

import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.signal
from mpl_toolkits.mplot3d import Axes3D

omega = 1
rho = 0.05
tau = 0.5
beta = 0.27


def potential(x, Q, xi):
    v1 = 0.5 * tau * ((0.5 * tau + 1) * omega * np.square(Q) + 0.5 * tau)
    v2 = x - (omega * Q / ((0.5 * tau + 1) * omega * np.square(Q) + 0.5 * tau)) * xi

    return v1 * np.square(v2)

# f_oja = h5py.File("../data/oja_mc.jld", "r")
#
# Xi_oja = f_oja['Xi_mc'].value[:, 0]
# Q_oja = f_oja['Q_mc'].value[:-1, 0]
#
# q_sampled = Q_oja[np.arange(0, len(Q_oja), 10)]
# np.save('../data/example_Q.npy', q_sampled)

Q_oja = np.load('../data/example_Q.npy')

xs = np.linspace(-10, 10, 1500)
xi = 1/np.sqrt(rho)
# xi = 0

potentials = np.zeros((len(Q_oja), len(xs)))

Qs, Xs = np.meshgrid(Q_oja, xs)
Z = potential(Xs, Qs, xi)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(Qs, Xs, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()