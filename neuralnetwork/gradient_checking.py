import numpy as np
# This module can be used to check the back propagation implementation for bugs


def compute_grad_approximations(mynetwork, X_train, Y_train, epsilon = 1e-7):
    mynetwork.reinitialize_all_weights()
    loss_function = mynetwork.binary_crossentropy_loss
    fake_grads = {}
    for key, layer in list(mynetwork.layers.items())[1:]:
        fake_grads[key] = {}

        # Take care of the weights
        fake_grads[key]['dW'] = np.zeros_like(layer['weights'])
        row = layer['weights'].shape[0]
        col = layer['weights'].shape[1]

        for i in range(row):
            for j in range(col):
                # nudge the weights by epsilon plus
                mynetwork.layers[key]['weights'][i,j] += epsilon
                J_plus_epsilon = loss_function(X_train,Y_train)
                mynetwork.reinitialize_all_weights()
                # nudge the weights by epsilon minus
                mynetwork.layers[key]['weights'][i,j] -= epsilon
                J_minus_epsilon = loss_function(X_train,Y_train)
                mynetwork.reinitialize_all_weights()
                fake_grads[key]['dW'][i,j] =  (J_plus_epsilon - J_minus_epsilon) / (2*epsilon)

        fake_grads[key]['dW'] = fake_grads[key]['dW']


        # Take care of the biases
        fake_grads[key]['dB'] = np.zeros_like(layer['biases'])
        row = layer['biases'].shape[0]
        col = layer['biases'].shape[1]

        for i in range(row):
            for j in range(col):
                # nudge the weights by epsilon plus
                mynetwork.layers[key]['biases'][i,j] += epsilon
                J_plus_epsilon = loss_function(X_train,Y_train)
                mynetwork.reinitialize_all_weights()
                # nudge the weights by epsilon minus
                mynetwork.layers[key]['biases'][i,j] -= epsilon
                J_minus_epsilon = loss_function(X_train,Y_train)
                mynetwork.reinitialize_all_weights()
                fake_grads[key]['dB'][i,j] =  (J_plus_epsilon - J_minus_epsilon) / (2*epsilon)

        fake_grads[key]['dB'] = fake_grads[key]['dB']   
        
    return fake_grads

def concatenate_grads(grads, mynetwork):
    """After a backward pass returns grads
    (i.e. grads = mynetwork.backward(X,Y))
    this functions returns agiant vector containing
    all the partial derivatives of the network
    """

    full_weight_vector = np.zeros(shape=(0,1))
    # this ensures that the order of the vectors is preserved since layers is an orderedDict
    for key, layer in list(mynetwork.layers.items())[1:]:
        weights_flat = np.reshape(grads[key]['dW'], (grads[key]['dW'].shape[0]*grads[key]['dW'].shape[1],1))
        biases_flat  = np.reshape(grads[key]['dB'], (grads[key]['dB'].shape[0]*grads[key]['dB'].shape[1],1))
        full_weight_vector = np.concatenate((full_weight_vector, weights_flat), axis=0)
        full_weight_vector = np.concatenate((full_weight_vector, biases_flat), axis=0)
    return np.squeeze(full_weight_vector)


