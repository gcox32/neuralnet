import numpy as np
import matplotlib.pyplot as plt
from src import create_data, DenseLayer, NeuralNetwork
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


np.random.seed(15)

# input data is 300 samples of 2 parameters (coordinates) each, fit into 3 possible classes
X, y = create_data(100, 3)

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
        'n_neurons': 16,
        'activation': 'ReLu',
    },
    'B': {
        'n_inputs': 16,
        'n_neurons': 12,
        'activation': 'ReLu',
    },
    'C': {
        'n_inputs': 12,
        'n_neurons': 16,
        'activation': 'ReLu',
    },
    'output': {
        'n_inputs': 16,
        'n_neurons': 3,
        'activation': 'Softmax',
    },
}

loss = 'categorical crossentropy'
optimizer = 'SGD'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

net = NeuralNetwork(architecture=arc, loss=loss, optimizer=optimizer)

net.train(X_train, y_train, iterations = 10000, learning_rate = 1.0, decay = 0.0, momentum=None)

print(accuracy_score(y_test, net.predict(X_test)))

net.visualize_network()