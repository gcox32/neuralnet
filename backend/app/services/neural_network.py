from src.src import NeuralNetwork
import numpy as np

# Moving minimal_arc from app.py
minimal_arc = {
    'input': {
        'n_inputs': 2,
        'n_neurons': 3,
        'activation': None
    },
    'output': {
        'n_inputs': 3,
        'n_neurons': 3,
        'activation': 'softmax'
    }
}

def get_network_metrics():
    model = NeuralNetwork(architecture=minimal_arc)
    X = np.random.rand(100, 2)
    y = np.random.randint(0, 3, 100)
    model.train(X, y, iterations=100)
    return model.get_training_metrics()

def get_network_structure():
    model = NeuralNetwork(architecture=minimal_arc)
    return model.get_network_structure()

def train_network(data):
    model = NeuralNetwork(architecture=data['architecture'])
    model.train(np.array(data['X']), np.array(data['y']))
    return {
        'accuracy': model.accuracy_list[-1],
        'loss': model.data_loss_list[-1]
    } 