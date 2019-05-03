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

def graph(X,Y):
    plt.scatter(X[:,0], X[:,1], c=Y, s=40, cmap=plt.cm.Spectral)
    
  