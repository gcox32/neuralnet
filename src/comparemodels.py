# import my own code
from src import DenseLayer, NeuralNetwork
from testdata import create_spiral_data

# import outside modules
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

## import tensorflow
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(15)

# create data
X, y = create_spiral_data(1000, 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# create architecture
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
# instantiate my model
mynet = NeuralNetwork(architecture=arc, loss='categorical crossentropy', optimizer='sgd')

# instantiate keras similar keras model
kerasnet = Sequential()
kerasnet.add(Dense(16, input_dim = 2, activation = 'relu'))
kerasnet.add(Dense(12, activation = 'relu'))
kerasnet.add(Dense(16, activation = 'relu'))
kerasnet.add(Dense(1, activation = 'softmax'))
## compile model
kerasnet.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# train my model
mynet.train(X_train, y_train, iterations = 1000)

# train keras model
kerasnet.fit(X_train, y_train, epochs = 1000)

# evaluate my model
accuracy = accuracy_score(y_test, mynet.predict(X_test))
print('My model accuracy: {:.2f}'.format(accuracy*100))

# evaluate the keras model
accuracy = accuracy_score(y_test, kerasnet.predict(X_test))
print('Keras model accuracy: {:.2f}'.format(accuracy*100))