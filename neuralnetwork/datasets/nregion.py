"""
One-dimensional dataset with n_regions

"""

import numpy as np
import matplotlib.pyplot as plt


def load_data(regions=3):
    """Loads the MNIST dataset.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    # Initialize list of centers
    C = []
    X = np.random.rand(50)
    Y = np.zeros(50)
    for n in range(1,regions):
        X = np.append(X, np.random.rand(50) + 2*n)
        if n % 2 == 1:
            C.append(0.5 + 2*n)
        Y = np.append(Y, np.zeros(50) + (n % 2))
    X = X.reshape(X.shape[0],1)
    return X, Y, C

def graph(X,Y,decision=None):
    if decision == None:
        plt.figure(figsize=(19,1))
        plt.scatter(X[:,0], np.zeros(X[:,0].shape), c=Y, cmap='flag', s = 30)
        plt.title('Binary Labeled Data to Classify\n')
        plt.xlim(min(X[:,0])-1,max(X[:,0])+1)
        plt.show()
    else:
        plt.figure(figsize=(19,4))
        plt.scatter(X[:,0], np.zeros(X[:,0].shape), c=Y, cmap='flag', s = 30)

        plt.grid(alpha=.4,linestyle='--')
        X_line = np.arange(min(X[:,0])-1, max(X[:,0])+1, (max(X[:,0])-1 - min(X[:,0])-1)/1000)
        X_line_reshape = X_line.reshape(X_line.shape[0],1)
        Y_line = [decision(X_line_reshape[i])[0][0] for i in range(len(X_line_reshape))]
        plt.plot(X_line, Y_line)
        plt.title('Binary Labeled Data with Boundary \n')
        plt.ylim(-1,1)
        plt.xlim(min(X[:,0])-1, max(X[:,0])+1)
        x = np.array(range(100))

        plt.show()