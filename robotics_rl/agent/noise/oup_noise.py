""" Ornstein-Uhlenbeck Process for random noise
"""
import torch


class OrnsteinUhlenbeckProcess(object):

    def __init__(self, theta, sigma, dim, mu=0):
        self._theta = theta
        self.sigma = sigma
        self._mu = mu
        self._state = mu
        self._dim = dim

    def noise(self):
        v = self._theta*(self._mu - self._state) + \
            self.sigma*torch.randn(self._dim)
        self._state += v
        return self._state

    def reset(self):
        self._state = self._mu

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    process = OrnsteinUhlenbeckProcess(0.15, 0.2, 1)
    signal = []
    for i in range(1000):
        signal.append(process.noise().item())
    plt.plot(signal)
    plt.show()