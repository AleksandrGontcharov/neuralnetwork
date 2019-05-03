import itertools
import matplotlib.pyplot as plt


class Network:
    """A Network is a directed acyclic graph of layers."""
    
    def __init__(self, input_layer_size, name="L"):
        """A network is an acyclic graph
        """
        
        
        self.prefix = name
        
        neurons = []
        edges =  []
        
        # Create initial of input neurons
        layer_number = 0
        neurons = [self.prefix + str(0) + "_" + str(i+1) for i in range(input_layer_size)]
        layer_name = self.prefix + str(0)   
        self.layers = {layer_name: (neurons, edges)}
    
    def add_layer(self, number_of_neurons, connections = 'all'):
        """
        Allows you to add a hidden or output layer and choose an activation and define the connections to previous layer
        Connections is a list of tuples representing the connections to previous layer
        
        Example connections input: (assume previous layer has two neurons L1_1, L1_2)
        connections = [(1,1), (1,2), (2,2)]
        
        connections={ "I_1" : ["L1_1"],
                      "I_1" : ["L1_2"]
                      "I_2" : ["L1_2"]}
        """
        

            
        # Get highest layer number
        layer_number = len(self.layers)
        layer_name = self.prefix + str(layer_number)
        
        
        previous_layer_number = layer_number - 1
        previous_layer_name = self.prefix + str(previous_layer_number)
        
    
        #initialize nodes and edges
        neurons = []
        # Create initial nodes
        neurons = [layer_name + "_" + str(i+1) for i in range(number_of_neurons)]
        
        
        # Creating the edges
        
        # Write an assert statement here to ensure that the connections given are valid
        # Namely, that the first dimension is withint he limits of the available neurons
        
        if connections == 'all':
            previous_list_of_neurons = self.layers[previous_layer_name][0]
            current_list_of_neurons  = neurons
            edges = list(itertools.product(previous_list_of_neurons,current_list_of_neurons))
                                             
            
        self.layers[layer_name] =  (neurons, edges)
        
    @property
    def number_of_layers(self):
        return len(self.layers.keys())
    
    @property
    def largest_layer_size(self):
        return max([len(self.layers[key][0]) for key in self.layers.keys()])

    
    def summary(self):
        ''' Displays a network diagram in matplotlib
        '''
        # Loop and graph each layer
        for layer in self.layers.keys():
            x_value = layer[-1]
            y_middle = self.largest_layer_size / 2
            #step_
            Y = list(range(len(self.layers[layer][0])))
            X = [x_value for i in Y]

            fig = plt.scatter(X,Y, s = 1000)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.ylim(-1,self.largest_layer_size)
            plt.xlim(-1,self.number_of_layers+1)

        plt.show()
        
        
        
