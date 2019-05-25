"""
2 Dimensional nested circles

"""


import numpy as np
import matplotlib.pyplot as plt


def load_data(size=300, factor=5):
    """Creates the circles dataset.

    # Returns
        Tuple of Numpy arrays: X, Y.
    """
    # Initialize the X's
  

    size_inside = size // 2
    size_outside = size - size_inside
    linspace_in = np.linspace(0, 2 * np.pi, size_inside, endpoint=False)
    linspace_out = np.linspace(0, 2 * np.pi, size_outside, endpoint=False)

    # Outer circle
    outer_circ_x = np.cos(linspace_out) * factor
    outer_circ_x = [(np.random.rand()+x) for x in outer_circ_x]
    outer_circ_y = np.sin(linspace_out)  * factor
    outer_circ_y = [(np.random.rand()+x) for x in outer_circ_y]

    outer_circ = np.ndarray(shape=(size_outside,2))
    outer_circ[:,0] = outer_circ_x
    outer_circ[:,1] = outer_circ_y


    # Inner circle
    inner_circ_x = np.cos(linspace_in)
    inner_circ_x = [(np.random.rand()+x) for x in inner_circ_x]
    inner_circ_y = np.sin(linspace_in) 
    inner_circ_y = [(np.random.rand()+x) for x in inner_circ_y]

    inner_circ = np.ndarray(shape=(size_inside,2))
    inner_circ[:,0] = inner_circ_x
    inner_circ[:,1] = inner_circ_y


    # create region and labels

    X = np.concatenate((inner_circ, outer_circ), axis=0)
    Y = np.append(np.ones(size_inside),np.zeros(size_outside))

    return X, Y




def load_data_multiple_circles(size=200, number_of_blobs = 2):
    """Creates the blobs dataset.

    # Returns
        Tuple of Numpy arrays: X, Y.
    """
    if number_of_blobs == 2:
        X, Y = load_data(size=size)
        # Initialize the X's
        x_min, x_max = X[:,0].min(), X[:,0].max()
        y_min, y_max = X[:,1].min(), X[:,1].max()
        x_length = x_max - x_min
        x_width = y_max - y_min

        X2 = X + 2*x_length
        X_new = np.concatenate((X,X2))
        Y_new = np.concatenate((Y,Y))
    elif number_of_blobs == 3:
        X, Y = load_data(size=size)
        # Initialize the X's
        # Initialize the X's
        x_min, x_max = X[:,0].min(), X[:,0].max()
        y_min, y_max = X[:,1].min(), X[:,1].max()
        x_length = x_max - x_min
        x_width = y_max - y_min

        X3 = X.copy()

        X3[:,0] = X3[:,0]+2*x_length


        X_new = np.concatenate((X,X3))
        Y_new = np.concatenate((Y,Y))

        X4 = X.copy()
        X4[:,1] = X4[:,1]+2*x_length

        X_new = np.concatenate((X_new,X4))
        Y_new = np.concatenate((Y_new,Y))
        
    elif number_of_blobs == 4:
        X, Y = load_data(size=size)    
        # Initialize the X's
        x_min, x_max = X[:,0].min(), X[:,0].max()
        y_min, y_max = X[:,1].min(), X[:,1].max()
        x_length = x_max - x_min
        x_width = y_max - y_min

        X3 = X.copy()

        X3[:,0] = X3[:,0]+2*x_length


        X_new = np.concatenate((X,X3))
        Y_new = np.concatenate((Y,Y))

        X4 = X_new.copy()
        X4[:,1] = X4[:,1]+2*x_length

        X_new = np.concatenate((X_new,X4))
        Y_new = np.concatenate((Y_new,Y_new))



    return X_new, Y_new

def graph(X,Y):
    plt.scatter(X[:,0], X[:,1], c=Y, s=40, cmap=plt.cm.Spectral)
    
  