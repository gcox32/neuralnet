<a href="http://www.letmedemo.com"><img src="https://github.com/gcox32/neuralnet/blob/master/assets/logo.png" title="grantcoxdatasci" alt="grantcoxdatasci" width="200"></a>

# Build a Neural Network from the Ground Up

> A Python implementation of a neural network built from scratch, with both a core library and web interface.

## Features

- Custom neural network implementation with:
  - Configurable layer architecture via JSON
  - Multiple activation functions (ReLU, Leaky ReLU, ELU, Sigmoid, Tanh, Softmax)
  - Various optimizers (SGD, Adam, AdaGuard)
  - Support for different loss functions (MSE, Categorical/Binary Cross-entropy)
  - L2 regularization
  - Early stopping
  - Network visualization tools

- Flask web interface for:
  - Model training
  - File uploads
  - Status monitoring
 

## Installation

1. Clone the repository:

```bash
git clone https://github.com/gcox32/neuralnet.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage (Optional)

### Core Library

```python
from src import DenseLayer, NeuralNetwork

# Load architecture from JSON file
with open('data/architecture.json', 'r') as file:
    architecture = json.load(file)

# Create neural network instance
net = NeuralNetwork(architecture=architecture, loss='categorical crossentropy', optimizer='sgd')

# Train the network
net.train(X_train, y_train, iterations=1000, learning_rate=0.01, decay=0.0, momentum=0.9)

# Make predictions
predictions = net.predict(X_test)

# Visualize the network
net.visualize_network()

# Visualize the validation accuracy
net.visualize_validation()  

# Visualize the predictions
net.visualize_predictions(X_test, y_test)

# Save the model
net.save('data/model.pkl')

# Load the model
net = NeuralNetwork.load('data/model.pkl')

```

### Web Interface

yea, this is not even close to being done.

```bash
python app.py
```

## Documentation

The network supports:

- **Activation Functions**: ReLU, Leaky ReLU, ELU, Sigmoid, Tanh, Softmax
- **Loss Functions**: MSE, Categorical Cross-entropy, Binary Cross-entropy
- **Optimizers**: SGD, Adam, AdaGuard
- **Training Modes**: Batch, Mini-batch, Stochastic
- **Regularization**: L2 regularization
- **Early Stopping**: Based on validation loss

## Tests

seems like a good idea.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Grant Cox - [@Grant07549375](https://twitter.com/Grant07549375)

Project Link: [https://github.com/gcox32/neuralnet](https://github.com/gcox32/neuralnet)

## FAQ

- **How do I do *specifically* so and so?**
    - No problem! Just do this.

---

## Support

Reach out to me at one of the following places!

- Website at <a href="https://www.letmedemo.com" target="_blank">`letmedemo.com`</a>
- Twitter at <a href="https://twitter.com/Grant07549375" target="_blank">`@Grant07549375`</a>

---