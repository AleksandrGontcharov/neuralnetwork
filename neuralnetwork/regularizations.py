import numpy as np

# Loss Functions              
def L2():
    
    def regularization_grads(grads, batch_size, lambd):

        """ Computes the binary crossentropy loss for X of shape (batch_size, dim)
        and Y of shape (batch_size, 1).

        # Arguments:
            grads: dict, 
            batch_size:int, 
            lambd: float,  
        """
        # Initialize gradients
        
         # Loop backwards through layers stopping at 1
        for key in grads.keys():
            grads[key]["dW"] += (lambd/batch_size)*grads[key]["dW"]
                      
        return grads
    
    def regularization_loss(mynetwork, batch_size, lambd):

        """ Computes the binary crossentropy loss for X of shape (batch_size, dim)
        and Y of shape (batch_size, 1).

        # Arguments:

        """
        loss = 0

        for key, layer in list(mynetwork.layers.items())[1:]:
            loss += np.sum(np.multiply(layer["weights"],layer["weights"]))
                      
        return (lambd/(2*batch_size))*loss
    
    return regularization_grads, regularization_loss
