"""
2 Dimensional Blobs

"""

import numpy as np
import matplotlib.pyplot as plt


def load_data(size=200):
    """Creates the blobs dataset.

    # Returns
        Tuple of Numpy arrays: X, Y.
    """
    # Initialize the X's
    size = int(size)
    first_half = size // 2
    second_half = size - first_half
    
    X = np.ndarray(shape=(size,2))
    # Define 2 regions
    X[:,0] = np.append(-np.random.rand(first_half)-0.25,np.random.rand(second_half))  # Negative blob X
    X[:,1] = np.append(-np.random.rand(first_half)-0.25,np.random.rand(second_half))  # Negative blob Y
    # Define the Y's
    Y = np.append(np.zeros(first_half),np.ones(second_half))  # Labels

    return X, Y

def graph(X,Y):
    plt.scatter(X[:,0], X[:,1], c=Y, s=40, cmap=plt.cm.Spectral)