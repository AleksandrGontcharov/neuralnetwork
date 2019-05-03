"""
2 Dimensional Blobs

"""

import numpy as np
import matplotlib.pyplot as plt


def load_data(size=200):
    """Loads the MNIST dataset.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    # Initialize the X's
    size = int(size)
    half_size = int(size / 2)
    
    X_2D_blobs = np.ndarray(shape=(size,2))
    # Define 2 regions
    X_2D_blobs[:,0] = np.append(-np.random.rand(half_size)-0.25,np.random.rand(half_size))  # Negative blob X
    X_2D_blobs[:,1] = np.append(-np.random.rand(half_size)-0.25,np.random.rand(half_size))  # Negative blob Y
    # Define the Y's
    Y_2D_blobs = np.append(np.zeros(half_size),np.ones(half_size))  # Labels

    return X_2D_blobs, Y_2D_blobs

def graph(X,Y):
    plt.scatter(X[:,0], X[:,1], c=Y, s=40, cmap=plt.cm.Spectral)