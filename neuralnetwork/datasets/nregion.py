"""
One-dimensional dataset with n_regions

"""

import numpy as np
import matplotlib.pyplot as plt


def load_data(regions=2):
    """Loads the MNIST dataset.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    # Initialize the X's
    X_region = np.append(np.append(-np.random.rand(50)-0.25,np.random.rand(50)),np.random.rand(50)+1.25)
    Y_region = np.append(np.append(np.zeros(50),np.ones(50)),np.zeros(50))
    return X_region, Y_region

def graph(X,Y):
    plt.figure(figsize=(15,1))
    plt.scatter(X, np.zeros(X.shape), c=Y, cmap='flag', s = 30)
    plt.title('Binary Labeled Data to Classify\n')
    plt.show()