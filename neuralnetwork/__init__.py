import numpy as np

def input_data_check(X, Y):
    '''Given input data X and labels Y, prints a a message describing the dataset
    and returns True if data is in proper format.
    '''
    assert len(Y) == len(X), "Your X, Y datasets do not have the same number of examples"
    if len(X.shape) == 1:
        dimension = 1
    else:
        dimension = X.shape[1]
    print(f"{X.shape[0]} {dimension}-Dimensional examples with {len(np.unique(Y))} labels")
    return True