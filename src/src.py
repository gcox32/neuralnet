import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import json
from typing import Optional
from dataclasses import dataclass

np.random.seed(15)
class DenseLayer(object):
    """
    Densley connected layer.
    """
    ACTIVATION_FUNCTIONS = {
        'relu': lambda x: np.maximum(0, x),
        'leaky_relu': lambda x: np.where(x > 0, x, x * 0.01),
        'elu': lambda x: np.where(x > 0, x, np.exp(x) - 1),
        'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        'tanh': lambda x: np.tanh(x),
        'softmax': lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True),
        None: lambda x: x
    }

    def __init__(self, n_inputs: int, n_neurons: int, activation: Optional[str] = None) -> None:
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        try: 
            self.activation = activation.lower()
        except:
            self.activation = None
        self.outputlayer = False

    def forward(self, inputs: list) -> None:
        """
        Forward pass of the layer.
        
        Parameters
        ----------
        inputs : ndarray
            Input values to the layer, shape (n_samples, n_inputs)
        
        Updates
        -------
        self.inputs : ndarray
            Stores input values for use in backpropagation
        self.output : ndarray
            Computed output before activation, shape (n_samples, n_neurons)
        """
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases
        
    def activate(self) -> None:
        """
        Applies the layer's activation function to the inputs.
        
        Parameters
        ----------
        inputs : ndarray
            Input values to be activated
            
        Updates
        -------
        self.output : ndarray
            Activated output values
        """
        self.output = self.activation_function(self.output) # activation 

    def activation_function(self, inputs: list) -> list:
        """
        Applies the specified activation function to the inputs

        -----
        Supported activation functions:
        - ReLU: max(0, x)
        - Leaky ReLU: x if x > 0 else 0.01x
        - ELU: x if x > 0 else (e^x - 1)
        - Sigmoid: 1/(1 + e^(-x))
        - Softmax: e^x_i/Σe^x_j
        - Tanh: (e^x - e^(-x))/(e^x + e^(-x))
        """
        # Remember input values
        self.activationinputs = inputs
        try:
            return self.ACTIVATION_FUNCTIONS[self.activation](inputs)
        except KeyError:
            raise ValueError(f'{self.activation} is not a currently accepted activation function.')

    def backward(self, dvalues: list) -> None:
        """
        Calculates gradients of weights, biases, and inputs.
        
        Parameters
        ----------
        dvalues : ndarray
            Gradient of the loss function with respect to layer outputs
            
        Updates
        -------
        self.dweights : ndarray
            Gradient of the loss with respect to weights
        self.dbiases : ndarray
            Gradient of the loss with respect to biases
        self.dvalues : ndarray
            Gradient of the loss with respect to layer inputs
        """
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)

        # Add L2 regularization if specified
        if hasattr(self, 'l2_lambda') and self.l2_lambda > 0:
            self.dweights += self.l2_lambda * self.weights

        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        # print(dvalues.shape, self.weights.T.shape, self.inputs.shape)
        self.dvalues = np.dot(dvalues, self.weights.T)
    
    def backwards_activation(self, dvalues: list) -> None:
        """
        Calculates gradients of activation function

        Notes
        -----
        Derivatives:
        - ReLU: 1 if x > 0 else 0
        - Leaky ReLU: 1 if x > 0 else 0.01
        - ELU: 1 if x > 0 else e^x
        - Sigmoid: s(x)(1 - s(x)) where s(x) is sigmoid
        - Tanh: 1 - tanh^2(x)
        """
        # Since we need to modify the original variable, let's make a copy of the values first
        self.dvalues = dvalues.copy()
        if self.activation == 'relu':
            # Zero out gradient where input values were negative 
            self.dvalues[self.activationinputs <= 0] = 0
        elif self.activation == 'leaky_relu':
            self.dvalues[self.activationinputs <= 0] *= 0.01
        elif self.activation == 'elu':
            self.dvalues[self.activationinputs <= 0] *= np.exp(self.activationinputs[self.activationinputs <= 0])
        elif self.activation == 'sigmoid' or self.activation == 'logistic':
            sig = 1 / (1 + np.exp(-self.activationinputs))
            self.dvalues = sig * (1 - sig)
        elif self.activation == 'tanh' or self.activation == 'hyperbolic':
            self.dvalues = 1 - np.tanh(self.activationinputs)**2
        elif self.activation == None:
            pass 

    def params(self) -> list:
        return [self.n_inputs, self.n_neurons, self.activation]

class NeuralNetwork(object):
    """
    Network.
    """

    VALID_LOSS_FUNCTIONS = {'mse', 'categorical crossentropy', 'binary crossentropy'}
    VALID_OPTIMIZERS = {'adam', 'sgd', 'adaguard', None}
    VALID_BATCH_MODES = {'batch', 'mini-batch', 'stochastic'}

    def __init__(self, architecture: dict, loss: str = 'categorical crossentropy', optimizer: str = 'Adam'):
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
        self.iterations = 0

        # run .layers() at init to produce .nodes_list attribute
        self.layers()
        
        self.check_params()

    def check_params(self) -> None:
        """
        performs initial check of loss function and optimizer to make sure they are built into code before
        attempting to train data
        """
        if self.loss not in self.VALID_LOSS_FUNCTIONS:
            raise Exception(f'{self.loss} is not a recognized loss function.')
        if self.optimizer not in self.VALID_OPTIMIZERS:
            raise Exception(f'{self.optimizer} is not a recognized optimizer.')
        for idx, (i, n) in enumerate(zip(self.inputs_list, self.nodes_list[1:])):
            if idx == 0:
                corr = i
            if i != corr:
                raise Exception(f'Check your architecture. Inputs of any non-first layer need to equal the neuron count of the prior layer. {i} inputs in layer {idx + 1} does not mesh with the {corr} neurons from layer {idx}.')
            corr = n    

    def layers(self) -> list:
        """
        Creates layer instances based on network architecture.
        """
        layer_list = []
        self.inputs_list = [] # used for .check_params() method at initialization
        self.nodes_list = [] # used for visualization method and .check_params() at initialization
        self.activations_list = [None] # used for visualization method
        for idx, val in enumerate(self.architecture.values()):
            params = []
            for i in val.values():
                params.append(i)
            if idx == 0:
                self.nodes_list.append(params[0]) # add inputs of first layer as initial neurons
            self.inputs_list.append(params[0])
            self.nodes_list.append(params[1]) # add neurons from every layer including first layer
            self.activations_list.append(params[2])
            try:   
                layer_list.append(DenseLayer(params[0], params[1], params[2])) # build layers
            except:
                layer_list.append(DenseLayer(params[0], params[1], activation=None)) # if no activation is given, default to None
        layer_list[-1].outputlayer = True
        
        return layer_list

    def loss_function(self, y_pred: list, y_true: list) -> float:
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7) # clip predictions to avoid zeros
        samples = y_pred.shape[0]  
        
        # Calculate base loss
        if self.loss == 'categorical crossentropy':
            if len(y_true.shape) == 1:
                y_pred = y_pred[range(samples), y_true]
            negative_log_likelihoods = -np.log(y_pred)
            if len(y_true.shape) == 2:
                negative_log_likelihoods *= y_true
            data_loss = np.sum(negative_log_likelihoods) / samples
        elif self.loss == 'binary crossentropy':
            m = y_pred.shape[1]
            y_true = y_true.reshape(-1, m)
            cost = -1 / m * (np.dot(y_true, np.log(y_pred).T) + np.dot(1 - y_true, np.log(1 - y_pred).T))
            data_loss = np.mean(cost)
        elif self.loss == 'mse' or self.loss == 'mean squared error':
            sum_square_error = 0.0
            for i in range(len(y_true)):
                sum_square_error += (y_true[i] - y_pred[i]) ** 2
            data_loss = 1.0 / len(y_true) * sum_square_error
        else:
            raise Exception(f'{self.loss} is not a currently accepted loss function.')
        
        # Add L2 regularization if specified
        reg_loss = 0
        if hasattr(self, 'l2_lambda') and self.l2_lambda > 0:
            for layer in self.trained_network:
                reg_loss += self.l2_lambda * np.sum(layer.weights * layer.weights)
        
        return data_loss + reg_loss
    
    def accuracy_function(self, y_pred: list, y_true: list) -> float:
        predictions = np.argmax(y_pred, axis = 1)
        accuracy = np.mean(predictions == y_true)
        return accuracy

    def feedforward(self, layer_list: list, inputs: list, y_true: list) -> list:
        """
        updates
        -----------
        layer.output (array) : DenseLayer().output from the final layer of the architecture
        """
        for layer in layer_list:
            # forward pass X parameter
            layer.forward(inputs)

            # apply layer-specific activation function
            layer.activate()
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

    def back_prop(self, dvalues: list, layer_list: list, y_true: list) -> None:
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

    def backward_loss(self, dvalues: list, y_true: list) -> None:
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

    @property
    def current_learning_rate(self) -> float:
        """
        Calculate current learning rate based on decay and iterations.
        
        Returns
        -------
        float
            Current learning rate value
        """
        if self.decay:
            return self.learning_rate * (1. / (1. + self.decay * self.iterations))
        return self.learning_rate

    def optimize(self, layer: DenseLayer) -> None:
        """
        Updates the weights and biases using the chosen optimizer
        """

        if self.optimizer == 'adam':
            # Adam parameters
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-7

            # Initialize momentum and cache if not exists
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                layer.bias_cache = np.zeros_like(layer.biases)
                layer.iteration = 0

            layer.iteration += 1

            # Update momentum with current gradients
            layer.weight_momentums = beta1 * layer.weight_momentums + (1 - beta1) * layer.dweights
            layer.bias_momentums = beta1 * layer.bias_momentums + (1 - beta1) * layer.dbiases
            
            # Get corrected momentum
            weight_momentums_corrected = layer.weight_momentums / (1 - beta1 ** layer.iteration)
            bias_momentums_corrected = layer.bias_momentums / (1 - beta1 ** layer.iteration)
            
            # Update cache with squared current gradients
            layer.weight_cache = beta2 * layer.weight_cache + (1 - beta2) * layer.dweights**2
            layer.bias_cache = beta2 * layer.bias_cache + (1 - beta2) * layer.dbiases**2
            
            # Get corrected cache
            weight_cache_corrected = layer.weight_cache / (1 - beta2 ** layer.iteration)
            bias_cache_corrected = layer.bias_cache / (1 - beta2 ** layer.iteration)

            # Vanilla SGD parameter update + normalization with square rooted cache
            layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + epsilon)
            layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + epsilon)
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
            eps = 1e-7 # epsilon, a very small value to keep from zeroing out
            # if layer does not contain cache arrays, create ones filled with zeros
            if not hasattr(layer, 'weight_cache'):
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)
            # update cache with squared current gradients
            layer.weight_cache += layer.dweights**2
            layer.bias_cache += layer.dbiases**2
            # vanilla sgd parameter update + normalization with square rooted cache
            layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + eps)
            layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + eps)
        elif self.optimizer == None:
            pass
        else:
            pass

    def train(self, X: list, y_true: list, X_val: Optional[list] = None, y_val: Optional[list] = None, l2_lambda: float = 0.0, iterations: int = 1000, 
            learning_rate=1.0, decay=0.0, momentum=None, 
            batch_mode='batch mode', batch_size=None, 
            patience=5):
        """
        Train the neural network with optional early stopping based on validation loss.
        
        Parameters
        ----------
        X : ndarray
            Training input data
        y_true : ndarray
            Training target values
        X_val : ndarray, optional
            Validation input data
        y_val : ndarray, optional
            Validation target values
        patience : int, default=5
            Number of epochs to wait for validation loss improvement before stopping
        """
        # Initialize training parameters
        self.batch_mode = batch_mode.lower()
        self.l2_lambda = l2_lambda
        
        # Initialize metrics lists
        self.data_loss_list = []
        self.accuracy_list = []
        self.learning_rate_list = []
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
        
        # establish batch parameters
        self.batch_mode = batch_mode.lower()
        self.l2_lambda = l2_lambda

        if self.batch_mode == 'stochastic':
            self.batch_size = 1
        elif self.batch_mode == 'batch mode':
            self.batch_size = X.shape[0]
        elif batch_size and type(batch_size) == int:
            self.batch_size = batch_size
        
        batch_list = []
        for idx in range(0, X.shape[0], self.batch_size):
            batch = X[idx:min(idx + self.batch_size, X.shape[0]), :]
            batch_list.append(batch)

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
        
        # Training loop
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
            
            # Validation pass if validation data is provided
            if X_val is not None and y_val is not None:
                val_output = self.feedforward(layer_list=layer_list, inputs=X_val, y_true=y_val)
                val_loss = self.loss_function(val_output, y_val)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model weights
                    self.best_weights = [layer.weights.copy() for layer in layer_list]
                    self.best_biases = [layer.biases.copy() for layer in layer_list]
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f'Early stopping triggered. Validation loss hasn\'t improved for {patience} epochs.')
                    print(f'Best validation loss: {best_val_loss:.4f}')
                    # Restore best weights
                    for layer, weights, biases in zip(layer_list, self.best_weights, self.best_biases):
                        layer.weights = weights
                        layer.biases = biases
                    break
            
            self.iterations += 1
        
        # Save final network state
        self.trained_network = layer_list

    def predict(self, X: list) -> list:
        """
        Generate predictions for input samples
        """
        inputs = X
        # feed inputs forward through a trained network
        for layer in self.trained_network:
            # forward pass X parameter
            layer.forward(inputs)

            # apply layer-specific activation function
            layer.activate()
            inputs = layer.output
        
        results = np.argmax(layer.output, axis = 1)
        return results

    def draw_network(self, ax: plt.Axes, left: float, right: float, bottom: float, top: float, layer_sizes: list) -> None:
        """
        Draw a neural network cartoon using matplotlib with proper spacing.
        """
        # Set figure aspect ratio to be square
        ax.set_aspect('equal', adjustable='box')
        
        # Calculate proper spacing
        n_layers = len(layer_sizes)
        max_neurons = max(layer_sizes)
        
        # Adjust vertical spacing based on number of neurons
        v_spacing = (top - bottom) / max(max_neurons + 1, 2)
        h_spacing = (right - left) / max(n_layers + 1, 2)
        
        # Scale node radius based on spacing
        node_radius = min(v_spacing, h_spacing) * 0.3
        
        # Draw nodes for each layer
        for n, (layer_size, activation_func) in enumerate(zip(layer_sizes, self.activations_list)):
            x = left + (n * h_spacing)
            
            # Center the layer vertically
            layer_height = layer_size * v_spacing
            layer_bottom = (top + bottom - layer_height) / 2
            
            # Draw activation function labels
            if n != 0:  # Skip input layer
                plt.text(x, top + node_radius/2, 
                        activation_func or 'linear',
                        ha='center', va='bottom',
                        fontsize=8)
            
            # Draw nodes
            for m in range(layer_size):
                y = layer_bottom + (m * v_spacing)
                circle = plt.Circle((x, y), node_radius, 
                                  color='white', ec='black', zorder=4)
                ax.add_artist(circle)
                
                # Add labels
                if n == 0:  # Input layer
                    plt.text(x - node_radius*2, y, f'$X_{{{m+1}}}$', 
                            ha='right', va='center')
                elif n == n_layers - 1:  # Output layer
                    plt.text(x + node_radius*2, y, f'$y_{{{m+1}}}$', 
                            ha='left', va='center')
        
        # Draw edges between layers
        for n in range(n_layers - 1):
            layer_size_a = layer_sizes[n]
            layer_size_b = layer_sizes[n + 1]
            
            x_a = left + (n * h_spacing)
            x_b = left + ((n + 1) * h_spacing)
            
            # Calculate y positions for both layers
            layer_a_height = layer_size_a * v_spacing
            layer_b_height = layer_size_b * v_spacing
            layer_a_bottom = (top + bottom - layer_a_height) / 2
            layer_b_bottom = (top + bottom - layer_b_height) / 2
            
            # Draw connections
            for i in range(layer_size_a):
                for j in range(layer_size_b):
                    y_a = layer_a_bottom + (i * v_spacing)
                    y_b = layer_b_bottom + (j * v_spacing)
                    line = plt.Line2D([x_a, x_b], [y_a, y_b], 
                                    color='gray', alpha=0.2, zorder=1)
                    ax.add_artist(line)

    def save(self, filepath: str) -> None:
        """
        save weights and biases unique to this network's architecture

        """
        network = {}
        for i, layer in enumerate(self.trained_network):
            weights = layer.weights
            biases = layer.biases

            # add layer to dictionary with corresponding weights/biases
            network[str(i)] = {'weights': weights.tolist(),
                               'biases': biases.tolist()}

        with open(filepath, 'w') as outfile:
            json.dump(network, outfile, sort_keys=True, indent=4)

    def load(self, filepath: str) -> None:
        """
        load in and attach weights and biases to an architecture to skip training time

        """
        with open(filepath, 'r') as outfile:
            network = json.load(outfile)
        
        # loop through network dict, where each key is a layer
        layer_list = []
        for (_, value), layer in zip(network.items(), self.layers()):
            layer.weights = value['weights']
            layer.biases = value['biases']
            layer_list.append(layer)

        self.trained_network = layer_list

    def visualize_network(self, savename: Optional[str] = False) -> None:
        """
        Visualize the neural network architecture.
        """
        # Calculate aspect ratio based on network shape
        n_layers = len(self.layers())
        max_neurons = max(self.nodes_list)
        aspect_ratio = n_layers / max_neurons
        
        # Set figure size maintaining aspect ratio
        base_size = 8
        fig = plt.figure(figsize=(base_size, base_size))
        ax = fig.gca()
        ax.axis('off')
        
        self.draw_network(ax, 0.1, 0.9, 0.1, 0.9, self.nodes_list)
        
        if savename:
            if isinstance(savename, str):
                plt.savefig(f'{savename}.png', bbox_inches='tight')
            else:
                raise ValueError('savename must be a string or False')
        
        plt.tight_layout()
        plt.show()

    def visualize_validation(self) -> None:
        """
        Visualizes training metrics over iterations.
        """
        if self.iterations == 0:
            raise Exception("Network hasn't been trained yet.")
        
        x = range(self.iterations)
        fig, ax = plt.subplots(3, sharex=True, figsize=(10, 12))
        
        # Plot data loss
        ax[0].plot(x, self.data_loss_list, label='Training Loss')
        ax[0].set_title('Loss')
        ax[0].legend()
        
        # Plot accuracy
        ax[1].plot(x, self.accuracy_list, label='Training Accuracy')
        ax[1].set_title('Accuracy')
        ax[1].legend()
        
        # Plot learning rate
        ax[2].plot(x, self.learning_rate_list)
        ax[2].set_title('Learning Rate')
        
        plt.tight_layout()
        plt.show()

    def visualize_predictions(self, X: np.ndarray, y: np.ndarray, title: str = "Model Predictions") -> None:
        """
        Visualize model predictions for 2D input data.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, 2)
            2D input features to visualize
        y : ndarray, shape (n_samples,)
            True labels for coloring the points
        title : str, default="Model Predictions"
            Plot title
        
        Notes
        -----
        - Creates a contour plot of decision boundaries
        - Overlays scatter plot of actual data points
        - Only works for 2D input data
        """
        if X.shape[1] != 2:
            raise ValueError("This visualization only works for 2D input data")
        
        # Create a mesh grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        # Get predictions for all mesh points
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        
        # Plot data points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        plt.colorbar(scatter)
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.show()

@dataclass
class TrainingConfig:
    learning_rate: float = 1.0
    decay: float = 0.0
    momentum: Optional[float] = None
    batch_mode: str = 'batch mode'
    batch_size: Optional[int] = None
    patience: int = 5
    l2_lambda: float = 0.0

@dataclass
class MinimalConfig:
    learning_rate: float = 0.1  # smaller than default
    decay: float = 0.0
    momentum: Optional[float] = None
    batch_mode: str = 'batch mode'  # simplest processing mode
    batch_size: Optional[int] = None
    patience: int = None  # disable early stopping
    l2_lambda: float = 0.0  # no regularization

