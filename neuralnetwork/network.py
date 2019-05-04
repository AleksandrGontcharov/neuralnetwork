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
        biases = biases.reshape(biases.shape[0],1)
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
        elif activation == 'sigmoid':
            activation = self.sigmoid
        elif activation == 'absolute':
            activation = self.absolute
    
        
        self.layers[layer_name] =  {'neurons': neurons, 'edges': edges, 'weights': weights, 'activation': activation, 'biases': biases}
        
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
    
    def sigmoid(self, x):
        return np.divide(1,1+np.exp(-x))
    
    def absolute(self, x):
        return np.abs(x)
            
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
                  
    # Forward Propagation Algorith,
                  
    def forward(self,  X):
        '''
        Perform a forward bass where X has shape (batch_size, dim)
        Return an array
        '''
        input_dimension = len(self.layers[self.prefix + str(0)]['neurons'])
        assert input_dimension == X.shape[1], f"The data should be {input_dimension}-dimensional, you provided {X.shape[1]}-dim data"
        
                  
        # reshape the input to a column vector
        layer_input = X.T
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
                  
    # Loss function for a batch of samples
    
    def binary_crossentropy_loss(self, X, Y):
        """ Computes the loss for X of shape (batch_size, dim)
            and Y of shape (batch_size, 1)
        """
        number_of_samples = len(Y)
        Y_hat = self.forward(X)
        Y_hat = Y_hat.reshape(Y.shape)
        losses = -(np.multiply(Y,np.log(Y_hat)) + np.multiply(1-Y,np.log(1-Y_hat)))
        return np.average(losses)
                  
        

    def train(self,X,Y):
        pass
            
        
