import numpy as np
import matplotlib.pyplot as plt
from src import DenseLayer, NeuralNetwork
from testdata import create_spiral_data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


np.random.seed(15)

# input data is 300 samples of 2 parameters (coordinates) each, fit into 3 possible classes
X, y = create_spiral_data(1000, 3)

param1 = [i[0] for i in X]
param2 = [i[1] for i in X]

plot = False

# display data
if plot:
    plt.scatter(param1, param2, c = y)
    plt.show()

arc = {
    'A': {
        'n_inputs': 2,
        'n_neurons': 32,
        'activation': 'ReLu',
    },
    'B': {
        'n_inputs': 32,
        'n_neurons': 64,
        'activation': 'ReLu',
    },
    'C': {
        'n_inputs': 64,
        'n_neurons': 128,
        'activation': 'ReLu',
    },
    'output': {
        'n_inputs': 128,
        'n_neurons': 3,
        'activation': 'Softmax',
    },
}

loss = 'categorical crossentropy'
optimizer = 'SGD'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

net = NeuralNetwork(architecture=arc, loss=loss, optimizer=optimizer)

net.train(X_train, y_train, iterations = 10000, learning_rate = 1.0, decay = 0.0, momentum=None)

acc = accuracy_score(y_test, net.predict(X_test))

print('Model accuracy: {:.2f}'.format(acc))

net.visualize_network()