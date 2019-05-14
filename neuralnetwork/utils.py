from .network import Network
import numpy as np

def build_scoop_network_1D(C: list,R: list):
    ''' Build and initializes a one-dimensional neural network 
    from a given list of centers C and radii R.
    '''
    # Determine number of Neurons in Hidden Layers
    N = len(C)
    mynetwork = Network(1)
    mynetwork.add_layer(N, activation = 'absolute' )
    mynetwork.add_layer(N, activation = 'relu')
    mynetwork.add_layer(1, activation = 'sigmoid')
    mynetwork.summary()
    
    # Initialize parameters
    
    mynetwork.layers['L1']['weights'] = np.ones_like(mynetwork.layers['L1']['weights'])
    mynetwork.layers['L1']['biases'] = -np.array(C).reshape(mynetwork.layers['L1']['biases'].shape)

    mynetwork.layers['L2']['weights'] = -np.identity(mynetwork.layers['L2']['weights'].shape[0], dtype=None)
    mynetwork.layers['L2']['biases'] = np.array(R).reshape(mynetwork.layers['L2']['biases'].shape)+min(R)
    
    mynetwork.layers['L3']['weights'] = np.ones_like(mynetwork.layers['L3']['weights'])
    mynetwork.layers['L3']['biases'] = -min(R)
    
    return mynetwork




    
    