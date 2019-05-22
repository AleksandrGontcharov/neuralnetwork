from collections import OrderedDict

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from .activations import relu
from .losses import binary_crossentropy_loss
from .optimizers import momentum


class Network:

    """A Network is a directed acyclic graph of layers."""

    def __init__(self, input_dim: int) -> None:
        """Creates a blank neural network
        
        # Arguments:
            input_dim: int, the dimension of the input layer
        """
        self._prefix = "L"
        self._input_dim = input_dim
        layer_name = self._prefix + "0"
        self.layers = OrderedDict({layer_name: {"neurons": input_dim}})

    def add_layer(self, number_of_neurons: int, activation=relu) -> None:

        """Adds a hidden layers with specified number of neurons and activation
        
        A layer includes: number of layers, weights, biases, and activation
        
        # Arguments:
            number_of_neurons: int, represents the number of neurons in the layer
            activation: function, specifies the activation function for the layer
                        the function is unpacked as follow
                        activation, derivative = activation()
                        where activation is the activation function
                        and derivative is its derivative
        """
        # TEMPORARY
        #np.random.seed(2)

        # Generate layer name
        layer_name = self._prefix + str(self.number_of_layers)
        prev_layer_name = self._prefix + str(self.number_of_layers - 1)

        # Initialize layer dictionary
        self.layers[layer_name] = {}

        # Add number of neurons
        self.layers[layer_name]["neurons"] = number_of_neurons

        # Randomly initialize weights and biases
        self.layers[layer_name]["weights"] = np.random.randn(
            number_of_neurons, self.layers[prev_layer_name]["neurons"]
        )
        self.layers[layer_name]["biases"] = np.random.randn(number_of_neurons, 1)

        # Add trainability matrices (used during weight update step of GD)
        self.layers[layer_name]["weights_trainable"] = np.ones_like(
            self.layers[layer_name]["weights"]
        )
        self.layers[layer_name]["biases_trainable"] = np.ones_like(
            self.layers[layer_name]["biases"]
        )
        
        #unpack the activation function and derivative
        activation_func, derivative = activation()

        # Add activation and derivative
        self.layers[layer_name]["activation"] = activation_func
        self.layers[layer_name]["derivative"] = derivative

    @property
    def number_of_layers(self):
        return len(self.layers)

    @property
    def largest_layer_size(self):
        return max([self.layers[key]["neurons"] for key in self.layers.keys()])
    
    def summary(self):
        """ Displays a network diagram in matplotlib
        """
        # Quick summary of Neural Network
        for key, layer in self.layers.items():
            if key == "L0":
                print(f"Layer: {key}, Neurons: {layer['neurons']}")
            else:
                print(
                    f"Layer: {key}, Neurons: {layer['neurons']}, Weights: {layer['weights'].shape}, Biases: {layer['biases'].shape}"
                )

        # Loop and graph each layer
        for key, layer in self.layers.items():
            x_value = key[-1]
            y_middle = self.largest_layer_size / 2
            # labels = self.layers[layer][0]
            number_of_neurons = self.layers[key]["neurons"]
            # Y = list(range(number_of_neurons))
            max_length = self.largest_layer_size
            frac = max_length / (number_of_neurons + 1)
            Y = [i * frac for i in range(1, number_of_neurons + 1)]

            X = [x_value for i in Y]
            fig = plt.scatter(X, Y, s=500)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.ylim(-1, self.largest_layer_size + 1)
            plt.xlim(-1, self.number_of_layers)

        plt.show()

    # Decision function for graphing

    def decision(self, x):
        """Perform a forward propagation on a single element x but 
        doesn't apply the final activation.
        
        # Arguments:
            x: float, represents the number of neurons in the layer
        """
        x = np.array([x], ndmin=2)

        neuron_outputs = self.forward(x)
        highest_layer_name = self._prefix + str(self.number_of_layers - 1)
        return neuron_outputs[highest_layer_name]["Z"].item(0)

    # Forward pass

    def forward(self, X):
        """ Forward propagates a dataset X with shape (num_examples, dim) and
        returns a dictionary containing all neuron activations.
        
        # Arguments:
            X: numpy.ndarray, with shape (num_examples, dim)
        """
        # get input dimension
        input_dim = self._input_dim

        # initialize output
        neuron_outputs = {}

        # transpose the input matrix
        layer_input = X.T

        # save input layer activations for layer L0
        input_layer_name = self._prefix + "0"
        neuron_outputs[input_layer_name] = {}
        neuron_outputs[input_layer_name]["A"] = layer_input

        # propagate through layers L1 to LN
        for key, layer in list(self.layers.items())[1:]:
            neuron_outputs[key] = {}
            # linear component Z = weights*previous activations + biases 
            neuron_outputs[key]["Z"] = (
                np.matmul(layer["weights"], layer_input) + layer["biases"]
            )
            # apply activation to Z: A = activation(Z)
            neuron_outputs[key]["A"] = layer["activation"](neuron_outputs[key]["Z"])

            # update the input for the next loop iteration
            layer_input = neuron_outputs[key]["A"]

        return neuron_outputs
        
    
    # Backward pass
                      
    def backward(self, X, Y, **kwargs):
        """ Do a backward pass for batch X and binary labels Y and return 
        a dictionary grads of gradients.
        
        # Arguments:
            X: numpy.ndarray, with shape (num_examples, dim) - input examples
            Y: numpy.ndarray, with shape (num_examples, ) - binary labels
        """
        #Unpack kwargs
        loss_function = kwargs['loss_function']


           
        # unpack loss function and derivative
        _, loss_derivative = loss_function()
                      

                      
        # Get the batch size
        batch = X.shape[0]
        
        # Compute dZ dW, dB for output layer
        neuron_outputs = self.forward(X)
        highest_layer_name = self._prefix + str(self.number_of_layers - 1)
        # Get final Z
        final_Z = neuron_outputs[highest_layer_name]["Z"]
        Y_hat   = neuron_outputs[highest_layer_name]["A"]
        # Get last activation derivative
        last_activation_derivative = self.layers[highest_layer_name]['derivative']

                      
        # Get dZ of final activation
        dZ = np.multiply(loss_derivative(Y = Y, Y_hat = Y_hat), last_activation_derivative(final_Z))
                      
        # Adding this if statement later might help the algorithm run faster and avoid numerical errors
#         if loss_function == binary_crossentropy_lossand and last_activation_derivative == 'sigmoid derivative':
#                 dZ = Y_hat - Y
                      
        # Initialize gradients
        grads = {}

        # Loop backwards through layers stopping at 1
        for layer_number in range(self.number_of_layers - 1, 0, -1):

            # Update layer names for iteration
            current_layer_name = self._prefix + str(layer_number)
            prev_layer_name = self._prefix + str(layer_number - 1)

            # Get previous layer's activations
            A = neuron_outputs[prev_layer_name]["A"]

            # Define dW and dB and update gradients
            grads[current_layer_name] = {}
            grads[current_layer_name]["dW"] = np.matmul(
                np.multiply(np.divide(1, batch), dZ), A.T
            )
            grads[current_layer_name]["dB"] = np.multiply(
                (1 / batch), np.sum(dZ, axis=1, keepdims=True)
            )

            # Calculate dZ for next loop until we reach 1
            if layer_number > 1:
                # Get current layer weights
                W = self.layers[current_layer_name]["weights"]

                # Get previous layer's activation derivative
                derivative = self.layers[prev_layer_name]["derivative"]
                # Calculate previous layer Z
                Z_prev = neuron_outputs[prev_layer_name]["Z"]
                # Calculate dZ for next loop
                dZ = np.multiply(np.matmul(W.T, dZ), derivative(Z_prev))
                 
                 
                      
        # Add regularization_grads to grads               
        regularization = kwargs['regularization']
        lambd = kwargs['lambd']
        if regularization  == None:
            return grads
        else:
            # Unpack regularization
            regularization_grads, _ = regularization()
            # Update gradients
            grads = regularization_grads(grads=grads, batch_size=batch, lambd=lambd)         
            return grads

    def fit(
        self,
        X_train,
        Y_train,
        batch_size,
        num_epochs,
        learning_rate,
        loss_function=binary_crossentropy_loss,
        validation_data=None,
        optimizer=momentum,
        beta=0.9,
        beta2=0.999,
        regularization=None,
        lambd = 0.1,
    ):
        """ Performs mini batch gradient descent
        
        # Arguments:
            X_train: numpy.ndarray, training data ndarray with shape: (total_examples, dim)
            Y_train: numpy.ndarray, binary labels with shape = (batch_size, )
            batch_size: int, the size of the batch, should be smaller than total_examples
            num_epochs: int, specifies the number of epochs to train the neural network
            learning_rate: float, specifies the learning rate for gradient descent
            validation data: tuple, contains (X_val, Y_val) as a validation set
            optimizer: func, the optimizer for gradient descent, found in optimizers.py 
            beta: float, this is used in many optimizers
            beta2: float, this is used in the adam optimizer
        """
                      
        # Unpack the loss function
        loss_func, _ = loss_function()
        
        # Update the loss function if regularization is used:
        reg_loss = 0              
        if regularization != None:
            _, regularization_loss = regularization()
            reg_loss =  regularization_loss(mynetwork=self, batch_size=batch_size, lambd=lambd)

        loss = loss_func(Y=Y_train, Y_hat=self.predict(X_train))+reg_loss
        acc = self.accuracy(X_train, Y_train)

        desc = f"Loss:{loss:2f} Acc:{acc:2f}"
        if not validation_data is None:
            val_loss = loss_func(
                Y = validation_data[1], Y_hat=self.predict(validation_data[0]))+reg_loss
            val_acc = self.accuracy(validation_data[0], validation_data[1])
            desc += f" val_loss:{val_loss:2f} val_acc:{val_acc:2f}"

        # initialize the optimized matrices to zeros
        # these are the gradients used for the weight update
        # this dictionary holds the right shapes to perform, momentum, RMSprop and adam

        optimized_grads = {
            key: {
                "V_dW": np.zeros_like(self.layers[key]["weights"]),
                "V_dB": np.zeros_like(self.layers[key]["biases"]),
                "S_dW": np.zeros_like(self.layers[key]["weights"]),
                "S_dB": np.zeros_like(self.layers[key]["biases"]),
            }
            for key in list(self.layers.keys())[1:]
        }
                      
        # Unpack the optimizer and weight update
        optimizer, weight_update = optimizer()

        iteration = 1
        for i in tqdm(range(num_epochs), desc=desc):
            for X, Y in self.batch_generator(X_train, Y_train, batch_size):
                # Get gradients
                grads = self.backward(X=X, Y=Y, loss_function=loss_function, regularization=regularization, lambd=lambd)
                      
                # propagate through layers LN to L1 and update weights
                for key, layer in reversed(list(self.layers.items())[1:]):

                    # get gradients for layer's weights and biases
                    dW = grads[key]["dW"]
                    dB = grads[key]["dB"]

                    # get optimized gradients
                    V_dW = optimized_grads[key]["V_dW"]
                    V_dB = optimized_grads[key]["V_dB"]
                    S_dW = optimized_grads[key]["S_dW"]
                    S_dB = optimized_grads[key]["S_dB"]

                    # update the dictionary entries of the optimized gradients
                    V_dW, V_dB, S_dW, S_dB = optimizer(
                        dW=dW,
                        dB=dB,
                        V_dW=V_dW,
                        V_dB=V_dB,
                        S_dW=S_dW,
                        S_dB=S_dB,
                        beta=beta,  # default for adam optimizer/momentum/RMSprop 0.9
                        network=self,
                        learning_rate=learning_rate,
                        key=key,
                        X=X,
                        Y=Y,
                        beta2=beta2,  # default for adam optimizer 0.999
                    )
                    # Update the dictionary
                    optimized_grads[key]["V_dW"] = V_dW
                    optimized_grads[key]["V_dB"] = V_dB
                    optimized_grads[key]["S_dW"] = S_dW
                    optimized_grads[key]["S_dB"] = S_dB

                    # Weight Update
                    self.layers[key]["weights"] -= np.multiply(self.layers[key]["weights_trainable"],weight_update(
                        weights=layer["weights"],
                        gradient=dW,
                        V_d=V_dW,
                        S_d=S_dW,
                        learning_rate=learning_rate,
                        beta=beta,
                        beta2=beta2,
                        iteration=iteration,
                    ))
                    self.layers[key]["biases"] -= np.multiply(self.layers[key]["biases_trainable"],weight_update(
                        weights=layer["biases"],
                        gradient=dB,
                        V_d=V_dB,
                        S_d=S_dB,
                        learning_rate=learning_rate,
                        beta=beta,
                        beta2=beta2,
                        iteration=iteration,
                    ))

                # update iteration number
                iteration += 1

    def accuracy(self, X, Y):
        """ Returns accuracy of the dataset X with labels Y
        
        # Arguments:
            X: numpy.ndarray, with shape (num_examples, dim) - input examples
            Y: numpy.ndarray, with shape (num_examples, ) - binary labels
        """
        Y_hat = self.predict(X)
        Y_hat_pred = [0 if y < 0.5 else 1 for y in Y_hat[0]]
        accuracy = Y_hat_pred == Y
        correct_guesses = list(accuracy).count(True)
        total = len(Y)
        return correct_guesses / total

    @staticmethod
    def batch_generator(X_train, Y_train, batch_size):
        num_batches = int(np.ceil(X_train.shape[0] / batch_size))
        # Indices for batches
        batches = [i * batch_size for i in range(num_batches)]
        batches.append(len(X_train))
        for index in range(1, len(batches)):

            # get batch indices
            low = batches[index - 1]
            high = batches[index]

            # Define mini batch
            X = X_train[low:high]
            Y = Y_train[low:high]

            yield X, Y
                      

    def predict(self, X):
        """Returns the final layer of the neural network for
        an input of size (batch_size, dim)
        
        # Arguments:
            X: numpy.ndarray, with shape (num_examples, dim) - input examples
        """
        neuron_outputs = self.forward(X)
        highest_layer_name = self._prefix + str(self.number_of_layers - 1)
        return neuron_outputs[highest_layer_name]["A"]

    # Trainability functions

    def train_biases_only(self):
        """Train only the bias weights"""
        for key, layer in list(self.layers.items())[1:]:
            self.layers[key]["weights_trainable"] = np.zeros_like(
                self.layers[key]["weights_trainable"]
            )
            self.layers[key]["biases_trainable"] = np.ones_like(
                self.layers[key]["biases_trainable"]
            )

    def train_slopes_only(self):
        """Train only the slope weight (i.e. layers['weights'])"""
        for key, layer in list(self.layers.items())[1:]:
            self.layers[key]["weights_trainable"] = np.ones_like(
                self.layers[key]["weights_trainable"]
            )
            self.layers[key]["biases_trainable"] = np.zeros_like(
                self.layers[key]["biases_trainable"]
            )

    def train_all(self):
        """Sets all parameters to trainable"""
        for key, layer in list(self.layers.items())[1:]:
            self.layers[key]["weights_trainable"] = np.ones_like(
                self.layers[key]["weights_trainable"]
            )
            self.layers[key]["biases_trainable"] = np.ones_like(
                self.layers[key]["biases_trainable"]
            )

    def train_none(self):
        """Sets all parameters to trainable"""
        for key, layer in list(self.layers.items())[1:]:
            self.layers[key]["weights_trainable"] = np.zeros_like(
                self.layers[key]["weights_trainable"]
            )
            self.layers[key]["biases_trainable"] = np.zeros_like(
                self.layers[key]["biases_trainable"]
            )

    def reset_weights_to_zero(self):
        """Set all weights to zero"""
        for key, layer in list(self.layers.items())[1:]:
            self.layers[key]["weights"] = np.zeros_like(self.layers[key]["weights"])
            self.layers[key]["biases"] = np.zeros_like(self.layers[key]["biases"])
                      
                      
    def reinitialize_all_weights(self):
        """Randomly initializes all the weights in the network
        """
        # TEMPORARY
        np.random.seed(5)
        for key, layer in list(self.layers.items())[1:]:
            self.layers[key]["weights"] = np.random.randn(self.layers[key]["weights"].shape[0], self.layers[key]["weights"].shape[1])
            self.layers[key]["biases"] =  np.random.randn(self.layers[key]["biases"].shape[0], self.layers[key]["biases"].shape[1])
                      
                      
    # Initialization Methods:
    def scoop_initialization(self, xmin, xmax, type='uniform'):
        """Initializes all the weights in a 1-D network
        """
        # length of interval
        length = xmax - xmin
        
        # Get number of centers:
        number_of_centers = self.layers['L1']['weights'].shape[0]              
        
        # define centers 
        C = [xmin + length* (i) /(number_of_centers-1) for i in range(number_of_centers)]
        # define radii
        R = [(length / (number_of_centers-1)) / 2 for i in range(number_of_centers)]
        # the slopes
        self.layers['L1']['weights'] = np.ones_like(self.layers['L1']['weights'])
        # the centers
        self.layers['L1']['biases'] = -np.array(C).reshape(self.layers['L1']['biases'].shape)
        # cutting off relationship connections
        self.layers['L2']['weights'] = -np.identity(self.layers['L2']['weights'].shape[0], dtype=None)
        # the radii
        self.layers['L2']['biases'] = np.array(R).reshape(self.layers['L2']['biases'].shape)
        self.layers['L3']['weights'] = np.ones_like(self.layers['L3']['weights'])
        self.layers['L3']['biases'] = np.zeros_like(self.layers['L3']['biases'])
                      