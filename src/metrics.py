import numpy as np

def cossim(x, y):
    x = np.squeeze(x)
    y = np.squeeze(y)
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))