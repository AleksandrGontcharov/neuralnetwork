"""
One-dimensional dataset with n_regions

"""

import numpy as np
import matplotlib.pyplot as plt


def load_data(regions=3, validation = 0.20, points=50):
    """Create an n-region dataset with a list of centers and radii 
    corresponding the centers and radii of positively labeled examples.
    # Returns:  `X_train, Y_train, X_val, Y_val, C, R`.
    """

    # Initialize list of centers
    C = []
    X = np.random.rand(points)
    Y = np.zeros(points)
    for n in range(1,regions):
        X = np.append(X, np.random.rand(points) + 2*n)
        if n % 2 == 1:
            C.append(0.5 + 2*n)
        Y = np.append(Y, np.zeros(points) + (n % 2))
    X = X.reshape(X.shape[0],1)
    # Split into training and validation sets
    training_samples = int(len(X)*(1-validation))
    validation_samples = len(X) - training_samples

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train = X[:training_samples]
    Y_train = Y[:training_samples]

    X_val = X[training_samples: training_samples + validation_samples]
    Y_val = Y[training_samples: training_samples + validation_samples]



    return X_train, Y_train, X_val, Y_val, C

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