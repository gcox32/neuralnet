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
        self.output = np.dot(self.inputs, self.weights) + self.biases # weights and biases
        
    def activate(self, inputs):
        self.output = self.activation_function(self.output) # activation 

    def activation_function(self, inputs): # still need leaky ReLU, parametric ReLU, swish
        if self.activation == 'relu':
            return np.maximum(0, inputs)
        elif self.activation == 'sigmoid' or self.activation == 'logistic':
            return 1/(1+np.exp(-inputs))
        elif self.activation == 'softmax':
            expo = np.exp(inputs - inputs.max())
            expo_sum = np.sum(np.exp(inputs))
            return expo/expo_sum
        elif self.activation == 'tanh' or self.activation == 'hyperbolic':
            tanh = (np.exp(inputs) - np.exp(-inputs))/(np.exp(inputs) + np.exp(-inputs))
            return tanh
        elif self.activation == None:
            return inputs
        else:
            Exception(f'{self.activation} is not a currently accepted activation function.')
        self.inputs = np.array(inputs)

    def backward(self):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, self.dvalues)
        self.dbiases = np.sum(self.dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dvalues = np.dot(self.dvalues, self.weights)
    
    def backwards_activation(self, dvalues):
        # Since we need to modify the original variable, let's make a copy of the values first
        self.dvalues = dvalues.copy()
        print(self.dvalues.shape, self.inputs.shape)
        if self.activation == 'relu':
            # Zero gradient where input values were negative 
            self.dvalues[self.inputs <= 0] = 0
        elif self.activation =='sigmoid' or self.activation == 'logistic':
            sig = 1 / (1 + np.exp(-self.inputs))
            self.dvalues = sig * (1 - sig)
        elif self.activation == 'softmax':
            pass
        elif self.activation == 'tanh' or self.activation == 'hyperbolic':
            tanh = (np.exp(self.inputs) - np.exp(-self.inputs))/(np.exp(self.inputs) + np.exp(-self.inputs))
            self.dvalues = 1 - tanh**2
        elif self.activation == None:
            pass 

    def params(self):
        return [self.n_inputs, self.n_neurons, self.activation]

class NeuralNetwork(object):

    loss_list = ['mse', 'categorical crossentropy', 'binary crossentropy']
    optimizer_list = ['adam', 'sgd', None]

    def __init__(self, architecture, loss = 'categorical crossentropy', optimizer = 'Adam'):
        self.architecture = architecture
        self.loss = loss.lower()
        try:
            self.optimizer = optimizer.lower()
        except:
            self.optimizer = optimizer
        self.check_params()

    def check_params(self):
        if self.loss not in self.loss_list:
            raise Exception(f'{self.loss} is not a recognized loss function.')
        if self.optimizer not in self.optimizer_list:
            raise Exception(f'{self.optimizer} is not a recognized optimizer.')

    def layers(self):
        layer_list = []
        for val in self.architecture.values():
            params = []
            for i in val.values():
                params.append(i)
            try:   
                layer_list.append(DenseLayer(params[0], params[1], params[2]))
            except:
                layer_list.append(DenseLayer(params[0], params[1], activation=None))
        layer_list[-1].outputlayer = True
        return layer_list

    def loss_function(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7) # clip predictions to avoid zeros
        samples = len(y_pred)        
        if self.loss == 'categorical crossentropy':
            y_pred = y_pred[range(samples), y_true]
            negative_log_likelihoods = -np.log(y_pred)
            data_loss = np.mean(negative_log_likelihoods)
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

    def feedforward(self, layer_list, inputs, y_true, iteration):

        for layer in layer_list:
            layer.weights += 0.05 * np.random.randn(layer.params()[0], layer.params()[1])
            layer.biases += 0.05 * np.random.randn(1, layer.params()[1])

            for idx, layer in enumerate(layer_list):
                layer.forward(inputs)
                inputs = layer.output

                layer.activate(inputs)
                inputs = layer.output
                
                # data has passed completely through the network
                if layer.outputlayer == True:
                    data_loss = self.loss_function(inputs, y_true)
                    accuracy = self.accuracy_function(inputs, y_true)
                    # if the loss function, based on the current weights and biases produces a loss lower than the previous lowest
                    # print at what iteration, what the log loss is, and what the accuracy is
                    if data_loss < self.lowest_loss:
                        print('New set of weights found, iteration:', iteration, 'loss:', data_loss, 'acc:', accuracy)
                        self.lowest_loss = data_loss

                        # adjust the weights list and biases list with the current best weights and biases if such values produced
                        # the best log loss for our input data 
                        for ix, lay in enumerate(layer_list):
                            self.weights_list[ix] = lay.weights
                            self.biases_list[ix] = lay.biases
                
                    return layer.output

    def back_prop(self, dvalues, layer_list, y_true):
        self.backward_loss(dvalues = dvalues, y_true = y_true)
        # the derivative values first produced will be from reversing the loss function, which
        # is part of the network, not just an individual layer
        dvalues = self.dvalues

        for i, layer in enumerate(layer_list[::-1]):
            # reverse the activation function for the layer using its derivative
            layer.backwards_activation(dvalues=dvalues)
            
            # take the derivative of the weighting
            layer.backward()

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

    def optimize(self, layer, learning_rate = 1.0):
        if self.optimizer == 'adam':
            pass
        elif self.optimizer == 'sgd':
            print('weights shape:',layer.weights.shape,'\ndweights shape:',layer.dweights.shape)
            layer.weights += -learning_rate * layer.dweights
            layer.biases += -learning_rate * layer.dbiases
        elif self.optimizer == None:
            pass
        else:
            pass

    def train(self, X, y_true, iterations = 1000):

        # first check for key shape Exception where final layer output does not match the number of possible classes
        output_neurons = self.layers()[-1].params()[1]
        class_count = len(Counter(y_true).keys())
        if  output_neurons != class_count:
            raise Exception(f'Neuron count in output layer ({output_neurons}) does not equal the number of classes ({class_count}).')

        # establish helper variables: starting point for log loss, initial weights and biases
        self.lowest_loss = 1000
        self.weights_list = []
        self.biases_list = []
        layer_list = self.layers() # create list instance
        for layer in layer_list:
            self.weights_list.append(layer.weights)
            self.biases_list.append(layer.biases)
        
        # using a for loop, show the input data to architecture "iterations" number of times, slightly adjusting the weights
        # and biases with each pass
        for iteration in range(iterations):
            print('start loop')
            inputs = X

            # feed forward through layers
            dvalues = self.feedforward(layer_list = layer_list, inputs = inputs, y_true = y_true, iteration = iteration)

            # feed backward through layers
            self.back_prop(dvalues = dvalues, layer_list = layer_list, y_true = y_true)

            # for layer in layer_list:
            #     self.optimize(layer)

    def visualize(self):
        pass