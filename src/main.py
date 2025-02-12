import numpy as np
import matplotlib.pyplot as plt
from src import DenseLayer, NeuralNetwork
from testdata import create_spiral_data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import json

np.random.seed(15)

# input data is 300 samples of 2 parameters (coordinates) each, fit into 3 possible classes
X, y = create_spiral_data(1000, 3)

param1 = [i[0] for i in X]
param2 = [i[1] for i in X]

plot = False

# Minimal architecture
minimal_arc = {
    'input': {
        'n_inputs': 2,  # minimum for 2D data
        'n_neurons': 3,  # minimum for 3-class problem
        'activation': None  # linear activation
    },
    'output': {
        'n_inputs': 3,
        'n_neurons': 3,
        'activation': 'softmax'  # required for classification
    }
}

# display data
if plot:
    plt.scatter(param1, param2, c = y)
    plt.show()

# load in architecutre from .json file
with open('data/architecture.json', 'r') as outfile:
    arc = json.load(outfile)

# establish loss function and optimizer
loss = 'categorical crossentropy'
optimizer = 'SGD'

# split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create NN instance
net = NeuralNetwork(
    architecture=arc,
    # architecture=minimal_arc,
    loss=loss,
    optimizer=optimizer
)

# train network weights and biases
net.train(X_train, y_train, iterations = 1000, learning_rate = 1.0, decay = 0.0, momentum=None)

# measure accuracy
acc = accuracy_score(y_test, net.predict(X_test))
print('Test accuracy: {:.2f}'.format(acc))

# see network architecture
net.visualize_network()

# see validation accuracy
net.visualize_validation()

# # After training
net.visualize_predictions(X_test, y_test, "Test Set Predictions")

# save weights and biases to recreate network for future use
net.save('data/savetest.json')
