import numpy as np
from metrics import cossim
from numba import jit


def online(model, phi, tau, T, return_X=False, progress=None):
    xi, _ = model(1)
    p = xi.shape[-1]

    x = np.random.randn(1, p)*0.5 + 1/np.sqrt(2)
    Q = np.zeros((p*T+1))
    Q[0] = cossim(x, xi)

    if return_X:
        X = np.zeros((p * T + 1, p))
        X[0] = x

    prox_fun = lambda x: x-phi(x)/p

    iterator = range(p*T)
    iterator = progress(iterator) if progress else iterator

    for t in iterator:
        _, y_k = model(1)
        x_g = x + (tau/p) * np.matmul(np.matmul(x, y_k.T), y_k)
        prox_x = prox_fun(x_g)
        x = np.sqrt(p) * prox_x / np.linalg.norm(prox_x)
        Q[t+1] = cossim(x, xi)

        if return_X:
            X[t+1] = x

    if return_X:
        return X, Q

    return Q
