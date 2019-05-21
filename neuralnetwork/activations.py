import numpy as np


# Activation Functions
def relu():
    def activation(x):
        return np.maximum(x, 0)

    def derivative(x):
        return np.maximum(np.sign(x), 0)

    return activation, derivative


def sigmoid():
    def activation(x):
        return np.divide(1, 1 + np.exp(-x))

    def derivative(x, activation = activation):
        return activation(x) * (1 - activation(x))

    return activation, derivative


def sigmoid_10():
    def activation(x):
        return np.divide(1, 1 + np.exp(-10 * x))

    def derivative(x):
        return self.sigmoid_10(x) * (1 - self.sigmoid_10(x)) * 10

    return activation, derivative


def absolute():
    def activation(x):
        return np.abs(x)

    def derivative(x):
        return np.sign(x)

    return activation, derivative
