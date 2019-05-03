import numpy as np


class Layer:
    '''
    A building block for a neural network.
    Each layer can
    - Process input and get output: output = layer.forward(input)
    - Propagate gradients for backpass:
         grad_input = layer.backward(input, grad_output)

    If a layer has learnable parameters that are updated during layer.backward
    '''
    def __init__(self):
        '''Initialize parameters, if any. 
        This is a dummy layer so it does nothing'''
        pass

    def forward(self, input):
        '''A dummy layer just returns whatever it gets as input
        '''
        return input

    def backward(self, input, grad_output):
        '''
        '''

        num_units = input.shape[1]

        d_layer_d_input = np.eye(num_units)

        return np.dor(grad_output, d_layer_d_input)  # chain rule
