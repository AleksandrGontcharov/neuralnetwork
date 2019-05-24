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
    
    
def plot_decision_boundary(X, Y, predict):
    """predict: predict function from NN
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid


    Z = predict(np.c_[xx.ravel(), yy.ravel()])

    Z = np.where(Z > 0.5, 1, 0)


    #Z = mynetwork.decision(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')

    plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.Spectral)
    plt.show()