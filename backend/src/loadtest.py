from src import NeuralNetwork
from testdata import create_spiral_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

# load in architecutre from .json file
with open('data/architecture.json', 'r') as outfile:
    arc = json.load(outfile)

# create class instance
net = NeuralNetwork(architecture=arc)

# visualize network as reminder
net.visualize_network()

# load in weights and biases from previously saved trained weights and biases
net.load('data/savetest.json')

# create test data
X, y = create_spiral_data(1000, 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# make predictions
pred = net.predict(X_test)

# measure accuracy of predictions
acc = accuracy_score(y_test, pred)
print(f'Accuracy of loaded model: {acc}')

