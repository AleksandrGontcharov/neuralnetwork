import itertools
import matplotlib.pyplot as plt
import numpy as np

class Network:
    """A Network is a directed acyclic graph of layers."""
    
    def __init__(self, input_layer_size, name="L"):
        """A network is an acyclic graph
        """
        
        
        self.prefix = name
        
        neurons = []
        edges =  None
        
        # Create initial of input neurons
        layer_number = 0
        neurons = [self.prefix + str(0) + "_" + str(i+1) for i in range(input_layer_size)]
        layer_name = self.prefix + str(0)
        activation = None
        weights = np.random.randn(0, 0)     
        biases = np.random.randn(0)
        biases = biases.reshape(biases.shape[0],0)
        self.layers = {layer_name: {'neurons': neurons, 'edges': edges, 'weights': weights, 'activation': activation, 'biases': biases}}
    
    def add_layer(self, number_of_neurons, connections = 'all', activation = 'relu'):
        """
        Allows you to add a hidden or output layer and choose an activation and define the connections to previous layer
        Connections is a list of tuples representing the connections to previous layer
        # Arguments:
            number_of_neurons: int, represents the number of neurons in the layer
            
        """
        

        
        # Previous Layer Information
        previous_layer_number = len(self.layers) - 1
        previous_layer_name = self.prefix + str(previous_layer_number)
        previous_list_of_neurons = self.layers[previous_layer_name]['neurons']
        previous_number_of_neurons = len(previous_list_of_neurons)

        # Create Layer Name
        layer_name = self.prefix + str(len(self.layers))
        
        # Creating Nodes 

        neurons = [layer_name + "_" + str(i+1) for i in range(number_of_neurons)]
        

        # Creating the edges
        if connections == 'all':
            # Create All edges
            edges = list(itertools.product(previous_list_of_neurons,neurons))
                                             
        # Initialize Layer Weights
        
        weights = np.random.randn(number_of_neurons, previous_number_of_neurons,)
        biases = np.random.randn(number_of_neurons)
        biases = biases.reshape(biases.shape[0],1)
        
        # Define activation function
        if activation == 'relu':
            activation = self.relu
            activation_derivative = self.relu_derivative
        elif activation == 'sigmoid':
            activation = self.sigmoid
            activation_derivative = self.sigmoid_derivative
        elif activation == 'absolute':
            activation = self.absolute
            activation_derivative = self.absolute_derivative

    
        
        self.layers[layer_name] =  {'neurons': neurons, 'edges': edges, 'weights': weights,  'biases': biases,'activation': activation, 'activation_derivative': activation_derivative}
        
    @property
    def number_of_layers(self):
        return len(self.layers.keys())
    
    @property
    def largest_layer_size(self):
        return max([len(self.layers[key]['neurons']) for key in self.layers.keys()])


    
    def summary(self):
        ''' Displays a network diagram in matplotlib
        '''
        # Quick summary of Neural Network
        for key, layer in self.layers.items():
            activation = str(layer['activation'])[22:].split(" ")[0]
            if activation =="":
                activation = None
            print(f"Layer: {key}, Neurons: {len(layer['neurons'])}, Activation: {activation}, Weights: {layer['weights'].shape}, Biases: {layer['biases'].shape}")
    
        # Loop and graph each layer
        
        for key, layer in self.layers.items():
            x_value = key[-1]
            y_middle = self.largest_layer_size / 2
            #labels = self.layers[layer][0]
            labels = self.layers[key]['neurons']
     

            number_of_neurons = len(labels)
           # Y = list(range(number_of_neurons))
            max_length = self.largest_layer_size
            frac = max_length/(number_of_neurons+1)
            Y = [i*frac for i in range(1,number_of_neurons+1)]

            
            X = [x_value for i in Y]
            fig = plt.scatter(X,Y, s = 500)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.ylim(-1,self.largest_layer_size+1)
            plt.xlim(-1,self.number_of_layers)

        plt.show()
        
                  
            
    # Activation Functions
                  
    def relu(self, x):
        return np.maximum(x,0)
    
    def relu_derivative(self, x):
        return np.maximum(np.sign(x),0)
    
    def sigmoid(self, x):
        return np.divide(1,1+np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    def absolute(self, x):
        return np.abs(x)
    
    def absolute_derivative(self, x):
        return np.sign(x)
    
    
    
    # Predict function for one sample
                  
    def predict(self,  X):
        '''
        Perform a forward propagation for one example and return the output
        X.shape should be (input_dimension,)
        '''
        input_dimension = len(self.layers[self.prefix + str(0)]['neurons'])

        assert X.shape == (input_dimension,),  f"The data point have shape ({input_dimension},)."
        
        # reshape the input to a column vector
        layer_input = X.reshape(X.shape[0],1)
        # propagate through every layer
        for key, layer in self.layers.items():
            # Skip the input layer
            if key[-1] == '0':
                pass
            else:
                # Get weights, biases and activation function
                weights = layer['weights']
                biases = layer['biases']
                activation = layer['activation']
                # Do the linear combination step to get z
                z = np.matmul(weights, layer_input) + biases
                # Apply the activation function
                output = activation(z)
                # redefine the input for the next loop iteration
                layer_input = output
        return output
                  
    # Decision Function for graphing

    def decision(self,  X):
        '''
        Perform a forward propagation but doesn't apply the final activation
        '''
        input_dimension = len(self.layers[self.prefix + str(0)]['neurons'])

        assert X.shape == (input_dimension,),  f"The data point have shape ({input_dimension},)."
        
        # reshape the input to a column vector
        layer_input = X.reshape(X.shape[0],1)
        # propagate through every layer
        for key, layer in self.layers.items():
            # Skip the input layer
            if key[-1] == '0':
                pass
            else:
                # Get weights, biases and activation function
                weights = layer['weights']
                biases = layer['biases']
                activation = layer['activation']
                # Do the linear combination step to get z
                z = np.matmul(weights, layer_input) + biases
                # Apply the activation function
                output = activation(z)
                # redefine the input for the next loop iteration
                layer_input = output
        return z
                  
    # Forward Propagation Step
                  
    def forward(self,  X):
        '''
        Perform a forward bass where X has shape (batch_size, dim)
        Returns an array of the last activations and a dictionary containing all neuron activations 
        before 
        '''
        input_dimension = len(self.layers[self.prefix + str(0)]['neurons'])
        assert input_dimension == X.shape[1], f"The data should be {input_dimension}-dimensional, you provided {X.shape[1]}-dim data"
        
        # Define activations
        neuron_outputs = {}
                  
        # reshape the input to a column vector
        layer_input = X.T
        # propagate through every layer
        for key, layer in self.layers.items():
            # Skip the input layer
            if key[-1] == '0':
                neuron_outputs[key] = {}
                neuron_outputs[key]['A'] = layer_input
            else: 
                # Get weights, biases and activation function
                weights = layer['weights']
                biases = layer['biases']
                activation = layer['activation']
                # Do the linear combination step to get z
                z = np.matmul(weights, layer_input) + biases
                # Apply the activation function
                output = activation(z)
                # Append values to dictionary
                neuron_outputs[key] = {}
                neuron_outputs[key]['Z'] = z
                neuron_outputs[key]['A'] = output
                # redefine the input for the next loop iteration
                layer_input = output
                  
                  
        
        return output, neuron_outputs
     
    # Backward Propagation Step
                  
    def backward(self,  X, Y):
        '''
        Assume final activation is sigmoid and loss is binary crossentropy
        Perform a forward bass where X has shape (batch_size, dim)
        Return an array
        
        '''
        # Can be updated later for other losses
                  
        # Get the batch size
        batch_size = X.shape[0]
                         
        # Compute dZ dW, dB for output layer
        Y_hat, neuron_outputs = self.forward(X)
                  
        # For now we assume that the loss function is binary crossentropy and final activation is sigmoid
        dZ = Y_hat - Y  
    
        # Initialize gradients        
        grads = {}
                  
        # Loop backwards through layers 
        for layer_number in range(len(self.layers)-1, 0, -1):

            # Update layer names for iteration  
            current_layer_name = self.prefix + str(layer_number)
            previous_layer_name = self.prefix + str(layer_number-1)
                  
            # Get previous layer's activations
            A = neuron_outputs[previous_layer_name]['A']

            # Define dW and dB and update gradients
            grads[current_layer_name] = {}
            grads[current_layer_name]['dW'] = np.matmul(np.multiply(np.divide(1, batch_size), dZ), A.T)
            grads[current_layer_name]['dB'] = np.multiply((1/batch_size),np.sum(dZ, axis=1, keepdims=True))
                  
            # Calculate dZ for next loop
            if layer_number >= 2:
                # Get current layer weights
                W = self.layers[current_layer_name]['weights']
                # Get previous layer's activation derivative
                activation_derivative = self.layers[previous_layer_name]['activation_derivative']
                # Calculate previous layer Z
                Z_prev = neuron_outputs[previous_layer_name]['Z']
                # Calculate dZ for next loop
                dZ = np.multiply(np.matmul(W.T, dZ), activation_derivative(Z_prev))
        
        return grads
                  
    # Loss function for a batch of samples
    
    def binary_crossentropy_loss(self, X, Y):
        """ Computes the loss for X of shape (batch_size, dim)
            and Y of shape (batch_size, 1)
        """
                  
        Y_hat, _ = self.forward(X)
        Y_hat = Y_hat.reshape(Y.shape)
        losses = -(np.multiply(Y,np.log(Y_hat)) + np.multiply(1-Y, np.log(1-Y_hat)))
        return np.average(losses)
                  
    
    def binary_crossentropy_loss_derivative(self, X, Y):
        """ The derivative of the binary crossentropy 
        (-Y)/Y_hat + (1 - Y)/(1-Y_hat)
        """
        Y_hat, _ = self.forward(X)
        Y_hat = Y_hat.reshape(Y.shape)
                  
        return  np.add(np.divide(-Y,Y_hat),np.divide(1-Y,1-Y_hat)).reshape(Y_hat.shape[0],1)
        

    def train(self, X, Y, learning_rate, num_epochs):
        '''
        Train your network on a given batch of X and y.
        '''
             
        
        for i in range(1, num_epochs):
            
            # Perform a forward and backwardpass with backward function      
            grads = self.backward(X, Y)
            # Update the weights for all layers
            for key, layer in self.layers.items():
                # Skip the input layer
                if key[-1] == '0':
                    pass
                else:
                    # get layers' weights and biases
                    W = layer['weights']
                    B = layer['biases']  

                    # get gradients for layer's weights and biases
                    dW = grads[key]['dW']       
                    dB = grads[key]['dB']   

                    # Perform the update
                    self.layers[key]['weights'] = W - learning_rate * dW  
                    self.layers[key]['biases'] = B - learning_rate * dB 


                
                  
            

    