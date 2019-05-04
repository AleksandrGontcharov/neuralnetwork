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
    X = np.append(np.append(-np.random.rand(50)-0.25,np.random.rand(50)),np.random.rand(50)+1.25)
    X = X.reshape(X.shape[0],1)
    Y = np.append(np.append(np.zeros(50),np.ones(50)),np.zeros(50))
    return X, Y

def graph(X,Y,decision=None):
    if decision == None:
        plt.figure(figsize=(15,1))
        plt.scatter(X[:,0], np.zeros(X[:,0].shape), c=Y, cmap='flag', s = 30)
        plt.title('Binary Labeled Data to Classify\n')
        plt.xlim(min(X[:,0])-1,max(X[:,0])+1)
        plt.show()
    else:
        plt.figure(figsize=(15,4))
        plt.scatter(X[:,0], np.zeros(X[:,0].shape), c=Y, cmap='flag', s = 30)

        plt.grid(alpha=.4,linestyle='--')
        X_line = np.arange(min(X[:,0])-1, max(X[:,0])+1, (max(X[:,0])-1 - min(X[:,0])-1)/50)
        X_line_reshape = X_line.reshape(X_line.shape[0],1)
        Y_line = [decision(X_line_reshape[i])[0][0] for i in range(len(X_line_reshape))]
        plt.plot(X_line, Y_line)
        plt.title('Binary Labeled Data with Boundary \n')
        plt.ylim(-1,1)
        plt.xlim(min(X[:,0])-1, max(X[:,0])+1)
        x = np.array(range(100))

        plt.show()