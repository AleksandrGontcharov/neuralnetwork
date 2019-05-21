import numpy as np

# Loss Functions              
def binary_crossentropy_loss():
    

    def loss_function(Y, Y_hat):

        """ Computes the binary crossentropy loss for X of shape (batch_size, dim)
        and Y of shape (batch_size, 1).

        # Arguments:
            Y: numpy.ndarray, with shape (num_examples, ) - binary labels
            Y_hat: numpy.ndarray, with shape (num_examples, ) - predicitions 
        """
        Y_hat = Y_hat.reshape(Y.shape)
        losses = -(np.multiply(Y, np.log(Y_hat)) + np.multiply(1 - Y, np.log(1 - Y_hat)))

        return np.average(losses)
    
    def loss_derivative(Y, Y_hat):

        """ Computes the binary crossentropy loss for X of shape (batch_size, dim)
        and Y of shape (batch_size, 1).

        # Arguments:
            Y: numpy.ndarray, with shape (num_examples, ) - binary labels
            Y_hat: numpy.ndarray, with shape (num_examples, ) - predicitions 
        """
        Y_hat = Y_hat.reshape(Y.shape)
        batch_size = Y.shape[0]
        
        derivative = -(np.multiply(Y, 1 / (Y_hat+1e-30)) - np.multiply(1 - Y, 1 / (1 - Y_hat+1e-30)))
        # added +1e-30 for numerical stability

        return derivative
    
    return loss_function, loss_derivative
