import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

np.random.seed(15)

def create_data(points, classes):
    """
    create spiral data set 
    params
    ----------
    points (int) : number of points for each class
    classes (int) : number of distinct classes, evenly distributed

    returns
    ---------
    X (numpy array) : array of shape (points*classes, 2)
    y (numpy array) : 1d array of classes
    """
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number + 1))
        r = np.linspace(0.0, 1, points) # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number

    return X, y

class DenseLayer(object):
    """
    Densley connected layer.
    """
    def __init__(self, n_inputs, n_neurons, activation):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        try: 
            self.activation = activation.lower()
        except:
            self.activation = None
        self.outputlayer = False

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases
        
    def activate(self, inputs):
        self.output = self.activation_function(self.output) # activation 

    def activation_function(self, inputs): # still need leaky ReLU, parametric ReLU, swish
        # Remember input values
        self.activationinputs = inputs
        if self.activation == 'relu':
            return np.maximum(0, inputs)
        elif self.activation == 'sigmoid' or self.activation == 'logistic':
            return 1 / (1 + np.exp(-inputs))
        elif self.activation == 'softmax':
            # get unnormalized probabilities
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
            # normalize them for each sample
            probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            return probabilities
        elif self.activation == 'tanh' or self.activation == 'hyperbolic':
            tanh = (np.exp(inputs) - np.exp(-inputs))/(np.exp(inputs) + np.exp(-inputs))
            return tanh
        elif self.activation == None:
            return inputs
        else:
            Exception(f'{self.activation} is not a currently accepted activation function.')

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        # print(dvalues.shape, self.weights.T.shape, self.inputs.shape)
        self.dvalues = np.dot(dvalues, self.weights.T)
    
    def backwards_activation(self, dvalues):
        # Since we need to modify the original variable, let's make a copy of the values first
        self.dvalues = dvalues.copy()
        if self.activation == 'relu':
            # Zero out gradient where input values were negative 
            self.dvalues[self.activationinputs <= 0] = 0
        elif self.activation =='sigmoid' or self.activation == 'logistic':
            sig = 1 / (1 + np.exp(-self.activationinputs))
            self.dvalues = sig * (1 - sig)
        elif self.activation == 'softmax':
            pass
        elif self.activation == 'tanh' or self.activation == 'hyperbolic':
            tanh = (np.exp(self.activationinputs) - np.exp(-self.activationinputs))/(np.exp(self.activationinputs) + np.exp(-self.activationinputs))
            self.dvalues = 1 - tanh**2
        elif self.activation == None:
            pass 

    def params(self):
        return [self.n_inputs, self.n_neurons, self.activation]

class NeuralNetwork(object):
    """
    Network.

    """

    loss_list = ['mse', 'categorical crossentropy', 'binary crossentropy']
    optimizer_list = ['adam', 'sgd', 'adaguard', None]

    def __init__(self, architecture, loss = 'categorical crossentropy', optimizer = 'Adam'):
        """   
        params
        ----------
        architecture (dict) : dictionary object that includes each layer as a key and the structure of each layer as a sub
            dictionary as its value
        loss (str) : loss function specific to the network; current options are ['categorical crossentropy', 'binary crossentropy', 'mse']
        optimizer (str) : optimizer specific to the network; current options are ['sgd', 'adam']
        """
        self.architecture = architecture
        self.loss = loss.lower()
        try:
            self.optimizer = optimizer.lower()
        except:
            self.optimizer = optimizer
        self.check_params()
        self.iterations = 0

        # run .layers() at init to produce .nodes_list attribute
        self.layers()

    def check_params(self):
        """
        performs initial check of loss function and optimizer to make sure they are built into code before
        attempting to train data
        """
        if self.loss not in self.loss_list:
            raise Exception(f'{self.loss} is not a recognized loss function.')
        if self.optimizer not in self.optimizer_list:
            raise Exception(f'{self.optimizer} is not a recognized optimizer.')

    def layers(self):
        layer_list = []
        self.nodes_list = [] # used for visualization method
        self.activations_list = [None] # used for visualization method
        for idx, val in enumerate(self.architecture.values()):
            params = []
            for i in val.values():
                params.append(i)
            if idx == 0:
                self.nodes_list.append(params[0]) # add inputs of first layer as initial neurons
            self.nodes_list.append(params[1]) # add neurons from every layer including first layer
            self.activations_list.append(params[2])
            try:   
                layer_list.append(DenseLayer(params[0], params[1], params[2])) # build layers
            except:
                layer_list.append(DenseLayer(params[0], params[1], activation=None)) # if no activation is given, default to None
        layer_list[-1].outputlayer = True
        
        return layer_list

    def loss_function(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7) # clip predictions to avoid zeros
        samples = y_pred.shape[0]  
        if self.loss == 'categorical crossentropy':
            # Probabilities for target values - only if categorical labels
            if len(y_true.shape) == 1:
                y_pred = y_pred[range(samples), y_true]
            # Losses
            negative_log_likelihoods = -np.log(y_pred)
            # Mask values - only for one-hot encoded labels
            if len(y_true.shape) == 2:
                negative_log_likelihoods *= y_true
            # Overall loss
            data_loss = np.sum(negative_log_likelihoods) / samples
            return data_loss
        elif self.loss == 'binary crossentropy':
            m = y_pred.shape[1]
            y_true = y_true.reshape(-1, m)
            cost = -1 / m * (np.dot(y_true, np.log(y_pred).T) + np.dot(1 - y_true, np.log(1 - y_pred).T))
            return np.mean(cost)
        elif self.loss == 'mse' or self.loss == 'mean squared error':
            sum_square_error = 0.0
            for i in range(len(y_true)):
                sum_square_error += (y_true[i] - y_pred[i]) ** 2
            mean_square_error = 1.0 / len(y_true) * sum_square_error
            return np.mean(mean_square_error)
        else:
            raise Exception(f'{self.loss} is not a currently accepted loss function.')
    
    def accuracy_function(self, y_pred, y_true):
        predictions = np.argmax(y_pred, axis = 1)
        accuracy = np.mean(predictions == y_true)
        return accuracy

    def feedforward(self, layer_list, inputs, y_true):
        """
        params
        -----------
        layer_list (list) : 
        inputs (array) : 
        y_true (1d array) :
        iteration (int) :

        returns
        -----------
        layer.output (array) : DenseLayer().output from the final layer of the architecture
        """
        for layer in layer_list:
            # forward pass X parameter
            layer.forward(inputs)

            # apply layer-specific activation function
            layer.activate(layer.output)
            inputs = layer.output
            
            # data has passed completely through the network
            if layer.outputlayer == True:

                # apply network-specific loss function
                data_loss = self.loss_function(inputs, y_true)
                accuracy = self.accuracy_function(inputs, y_true)
                # append metrics lists
                self.data_loss_list.append(data_loss)
                self.accuracy_list.append(accuracy)

                # if the loss function, based on the current weights and biases produces a loss lower than the previous lowest
                # print at what iteration, what the log loss is, and what the accuracy is
                if data_loss < self.lowest_loss:
                    print('New set of weights found, iteration:', self.iterations, 'loss:', data_loss, 'acc:', accuracy)

                    # set new loss standard
                    self.lowest_loss = data_loss

                    # adjust the weights list and biases list with the current best weights and biases if such values produced
                    # the best log loss for our input data 
                    for ix, lay in enumerate(layer_list):
                        self.weights_list[ix] = lay.weights
                        self.biases_list[ix] = lay.biases

                # return final layer's output for back propogation through the network
                return layer.output

    def back_prop(self, dvalues, layer_list, y_true):
        # the derivative values first produced will be from reversing the loss function
        self.backward_loss(dvalues = dvalues, y_true = y_true)

        # establish first derivative values; once past the loss function, derivative values for a layer will come from the prior layer
        dvalues = self.dvalues

        for layer in layer_list[::-1]:
            # reverse the activation function for the layer using its derivative
            layer.backwards_activation(dvalues=dvalues)

            # take the derivative of the weighting
            layer.backward(dvalues = layer.dvalues)
            
            # establish input for next layer
            dvalues = layer.dvalues

    def backward_loss(self, dvalues, y_true):
        samples = dvalues.shape[0]
        self.dvalues = dvalues.copy()  # Copy so we can safely modify
        if self.loss == 'categorical crossentropy':
            self.dvalues[range(samples), y_true] -= 1
            self.dvalues = self.dvalues / samples
        elif self.loss == 'binary crossentropy':
            pass
        elif self.loss == 'mse' or self.loss == 'mean squared error':
            pass
        else:
            raise Exception(f'{self.loss} is not a currently accepted loss function.')

    def optimize(self, layer):
        """
        updates the weights and biases of each neuron upon the next pass based on derivative weights and biases received
        from back propogation
        """

        self.current_learning_rate = self.learning_rate

        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1. / (1. + self.decay * self.iterations))

        if self.optimizer == 'adam':
            pass
        elif self.optimizer == 'sgd':
            if self.momentum:
                # if layer does not contain momentum arrays, create them, filled with zeros
                if not hasattr(layer, 'weight_momentums'):
                    layer.weight_momentums = np.zeros_like(layer.weights)
                    layer.bias_momentums = np.zeros_like(layer.biases)
                # build weight updates with momentum--take previous updates multiplied by retain factor and update with current gradients
                weight_updates = ((self.momentum * layer.weight_momentums) - (self.current_learning_rate * layer.dweights))
                bias_updates = ((self.momentum * layer.bias_momentums) - (self.current_learning_rate * layer.dbiases))
                layer.weight_momentums = weight_updates                
                layer.bias_momentums = bias_updates
            else:
                # vanilla SGD optimizer (no momentum)
                weight_updates = -self.current_learning_rate * layer.dweights
                bias_updates = -self.current_learning_rate * layer.dbiases
            layer.weights += weight_updates
            layer.biases += bias_updates
        elif self.optimizer == 'adagrad':
            pass
        elif self.optimizer == None:
            pass
        else:
            pass

    def train(self, X, y_true, iterations = 1000, learning_rate = 1.0, decay = 0.0, momentum = None):
        """
        Using the power of loops, passes the input data (X), forward through the layers, assesses the data loss and the accuracy of
        the predictions relative to y_true, then back propogates through the layers using .backprop() method, then calls the .optimize()
        method to adjust the weights and biases of each neuron in each layer. Adjust iterations to increase/decrease training time; 
        adjust learning_rate and decay to adjust speed across iterations.

        params
        ----------
        X (array) : input data
        y_true (1d array) : "ground truth"; classes to compare outputs against
        iterations (int) : number of times to show input data to the network 
        learning_rate (float) : passed into .optimize() method; typically a float between 0.0 and 1.0; the degree to which dervative values
            affect the weights and biases of each neuron upon the next pass
        decay (float) : passed into .optimize() method; typically a float between 0.0 and 1.0 where 1 - decay is the degree to which the
            learning rate is multiplied with each iteration
        momentum (float) : 

        updates
        ----------

        """
        # first check for key shape Exception where final layer output does not match the number of possible classes
        output_neurons = self.layers()[-1].params()[1]
        class_count = len(Counter(y_true).keys())
        if  output_neurons != class_count:
            raise Exception(f'Neuron count in output layer ({output_neurons}) does not equal the number of classes ({class_count}).')

        self.data_loss_list = []
        self.accuracy_list = []
        self.learning_rate_list = []

        # establish helper variables: starting point for log loss, initial weights and biases
        self.lowest_loss = 1000
        self.weights_list = []
        self.biases_list = []
        layer_list = self.layers() # create list instance
        for layer in layer_list:
            self.weights_list.append(layer.weights)
            self.biases_list.append(layer.biases)
        
        # establish optimization metrics
        self.decay = decay
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # using a for loop, show the input data to architecture "iterations" number of times, slightly adjusting the weights
        # and biases with each pass
        for iteration in range(iterations):
            inputs = X

            # feed forward through layers
            dvalues = self.feedforward(layer_list = layer_list, inputs = inputs, y_true = y_true)
            
            # feed backward through layers
            self.back_prop(dvalues = dvalues, layer_list = layer_list, y_true = y_true)

            for idx, layer in enumerate(layer_list):
                layer.weights = self.weights_list[idx]
                layer.biases = self.biases_list[idx]
                self.optimize(layer)

            self.learning_rate_list.append(self.current_learning_rate)
            self.iterations += 1

        self.trained_network = layer_list

    def visualize_validation(self):
        """
        takes a trained network and displays plots for accuracy, loss, and learning rate against iteration count
        """
        if self.iterations == 0:
            raise Exception("Network hasn't been trained yet. Call the .train() method of an instance of the NeuralNetwork class first.")
        else:
            x = range(self.iterations)
            acc = round(self.accuracy_list[-1], 4)
            loss = round(self.data_loss_list[-1], 4)

            fig, ax = plt.subplots(3, sharex = True)
            ax[0].plot(x, self.accuracy_list)
            ax[0].title.set_text(f'Accuracy: {acc}')
            ax[1].plot(x, self.data_loss_list)
            ax[1].title.set_text(f'Data Loss: {loss}')
            ax[2].plot(x, self.learning_rate_list)
            ax[2].title.set_text('Learning Rate')
            plt.show()

    def predict(self, X):
        """
        """
        inputs = X
        # feed inputs forward through a trained network
        for layer in self.trained_network:
            # forward pass X parameter
            layer.forward(inputs)

            # apply layer-specific activation function
            layer.activate(layer.output)
            inputs = layer.output
        
        results = np.argmax(layer.output, axis = 1)
        return results

    def draw_network(self, ax, left, right, bottom, top, layer_sizes):
        """
        Draw a neural network cartoon using matplotilb.
        
        :usage:
            >>> fig = plt.figure(figsize=(12, 12))
            >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
        
        :parameters:
            - ax : matplotlib.axes.AxesSubplot
                The axes on which to plot the cartoon (get e.g. by plt.gca())
            - left : float
                The center of the leftmost node(s) will be placed here
            - right : float
                The center of the rightmost node(s) will be placed here
            - bottom : float
                The center of the bottommost node(s) will be placed here
            - top : float
                The center of the topmost node(s) will be placed here
            - layer_sizes : list of int
                List of layer sizes, including input and output dimensionality
        """
        n_layers = len(self.layers())
        layer_sizes = self.nodes_list
        v_spacing = (top - bottom)/float(max(layer_sizes))
        h_spacing = (right - left)/float(len(layer_sizes) - 1)
        
        # Input-Arrows
        layer_top_0 = v_spacing * (layer_sizes[0] - 1)/2. + (top + bottom)/2.
        for m in range(layer_sizes[0]):
            plt.arrow(left - 0.18, layer_top_0 - m * v_spacing, 0.12, 0, lw=1, head_width=0.01, head_length=0.02)
        
        # Nodes
        for (n, layer_size), func in zip(enumerate(layer_sizes), self.activations_list):
            layer_top = v_spacing * (layer_size - 1)/2. + (top + bottom)/2.
            # Activation Function headers
            if n != 0:
                plt.text(n * h_spacing + left - 0.025, layer_top + 0.05, func)
            for m in range(layer_size):
                circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing/4.,
                                    color='w', ec='k', zorder=4)
                if n == 0:
                    plt.text(left - 0.15, layer_top - m * v_spacing, r'$X_{' + str(m + 1) + '}$', fontsize=15)
                elif (n_layers == 3) & (n == 1):
                    plt.text(n * h_spacing + left + 0.00, layer_top - m * v_spacing + (v_spacing / 8. + 0.01 * v_spacing), r'$H_{' + str(m + 1) + '}$', fontsize=15)
                elif n == n_layers:
                    plt.text(n * h_spacing + left + 0.10, layer_top - m * v_spacing, r'$y_{' + str(m + 1) + '}$', fontsize=15)
                ax.add_artist(circle)
        
        # Edges
        for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layer_top_a = v_spacing * (layer_size_a - 1)/2. + (top + bottom)/2.
            layer_top_b = v_spacing * (layer_size_b - 1)/2. + (top + bottom)/2.
            for m in range(layer_size_a):
                for o in range(layer_size_b):
                    line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                    [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k')
                    ax.add_artist(line)
        
        # Output-Arrows
        layer_top_0 = v_spacing * (layer_sizes[-1] - 1)/2. + (top + bottom)/2.
        for m in range(layer_sizes[-1]):
            plt.arrow(right + 0.015, layer_top_0 - m * v_spacing, 0.16 * h_spacing, 0, lw=1, head_width=0.01, head_length=0.02)

    def visualize_network(self, savename = False):
        fig = plt.figure(figsize=(9,9))
        ax = fig.gca()
        ax.axis('off')
        self.draw_network(ax, 0.1, 0.9, 0.1, 0.9, self.nodes_list)
        if savename != False:
            if type(savename) != str:
                raise Exception('savename parameter needs to either be a string or set to False.')
            plt.savefig(savename + '.png')
        plt.show()
