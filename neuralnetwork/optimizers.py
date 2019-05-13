# Optimizers
import numpy as np

#if you want to use no optimizer, then use momentum with beta = 0

def momentum():
    def optimizer(dW, dB, **kwargs):
        """ MOMENTUM OPTIMIZER
        Given dB, dW of the current mini batch
        and v_dW, v_dB of the previous gradient update,
        returns an updated v_dW, v_dB
        """
        V_dW = kwargs['V_dW']
        V_dB = kwargs['V_dB']
        beta = kwargs['beta']


        V_dW = beta*V_dW + (1-beta)*dW
        V_dB = beta*V_dB + (1-beta)*dB

        return V_dW, V_dB
    
    def weight_update(**kwargs):
        """ Rule on how to update the weights using this optimizer
        """
        weights = kwargs['weights']
        learning_rate = kwargs['learning_rate']
        V_d = kwargs['V_d']
        
        
        weights = weights - learning_rate * V_d
        
        return weights
    
    return optimizer, weight_update


def RMSprop():
    def optimizer(dW, dB, **kwargs):
        """ Given dB, dW of the current mini batch
        and v_dW, v_dB of the previous gradient update,
        returns an updated v_dW, v_dB
        """
        V_dW = kwargs['V_dW']
        V_dB = kwargs['V_dB']
        beta = kwargs['beta']


        V_dW = beta*V_dW + (1-beta)*np.square(dW)
        V_dB = beta*V_dB + (1-beta)*np.square(dB)

        return V_dW, V_dB
    
    def weight_update(**kwargs):
        """ Rule on how to update the weights using this optimizer
        """
        weights = kwargs['weights']
        learning_rate = kwargs['learning_rate']
        V_d = kwargs['V_d']
        gradient = kwargs['gradient']  # either dW or dB
        
        weights = weights - learning_rate * np.divide(gradient, np.sqrt(V_d)+10e-08)
        
        return weights
    
    return optimizer, weight_update