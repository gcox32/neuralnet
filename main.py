import numpy as np
import matplotlib.pyplot as plt
from src import create_data, DenseLayer, NeuralNetwork

np.random.seed(15)

# input data is 300 samples of 2 parameters (coordinates) each, fit into 3 possible classes
X, y = create_data(100, 3)

param1 = [i[0] for i in X]
param2 = [i[1] for i in X]

arc = {
    'A': {
        'n_inputs': 2,
        'n_neurons': 16,
        'activation': 'relu',
    },
    'B': {
        'n_inputs': 16,
        'n_neurons': 8,
        'activation': 'sigmoid',
    },
    'output': {
        'n_inputs': 8,
        'n_neurons': 3,
        'activation': 'tanh',
    },
}

loss = 'categorical crossentropy'
optimizer = 'SGD'

net = NeuralNetwork(architecture=arc, loss=loss, optimizer=optimizer)
net.train(X, y, iterations = 3)

# plt.scatter(param1, param2, c = y)
# plt.show()