import numpy as np


def spiked_model(omega, sigma=1, xi=None, rho=None, p=None):
    '''
    Returns spiked model, if xi is not given randomly samples a xi.
    '''

    if not xi:
        assert rho is not None, "If xi is not given provide rho"
        assert p is not None, "If xi is not given provide p"
        xi = random_xi(rho, p)

    xi = xi.reshape(1, -1)

    def gen(N):
        C = np.random.randn(N, 1)
        A = np.random.randn(N, p) * sigma

        return xi, np.sqrt(omega/p) * C * xi + A

    return gen

def random_xi(rho, p):
    return np.random.binomial(n=1, p=rho, size=p) / np.sqrt(rho)

