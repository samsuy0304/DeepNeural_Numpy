# DeepNeural_Numpy

This module contains a simple implementation of a neural network for multi-class classification tasks. It includes functionalities for forward and backward propagation, parameter initialization, training, prediction, and model saving/loading.

## Class: NeuralNetwork

### Methods:

- **\_\_init\_\_**: Initializes the neural network with given layer dimensions.
- **initialize_parameters**: Initializes the parameters of the neural network.
- **sigmoid**: Computes the sigmoid activation function.
- **relu**: Computes the ReLU activation function.
- **softmax**: Computes the softmax activation function.
- **forward_propagation**: Performs forward propagation through the network.
- **compute_cost**: Computes the cost function.
- **backward_propagation**: Performs backward propagation to compute gradients.
- **update_parameters**: Updates parameters using gradients and learning rate.
- **train_neural_network**: Trains the neural network on the provided data.
- **predict**: Performs predictions on new data.
- **save_parameters**: Saves the trained parameters of the model.
- **load_parameters**: Loads pre-trained parameters from a file.

### Usage Example:

```python
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load data
f = np.load('data_file.npz')
Y_tr = f['train_labels']
X_train = f['train_data']
Y_t = f['test_labels']
X_test = f['test_data']

# Convert labels to integers using LabelEncoder
label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_tr)
Y_test_encoded = label_encoder.fit_transform(Y_t)

Y_train = np.array(Y_train_encoded, dtype=int)
Y_test = np.array(Y_test_encoded, dtype=int)

# Flatten and normalize input data
X_train_flatten = X_train.T / 255.0
X_test_flatten = X_test.T / 255.0

# Convert labels to one-hot encoding
Y_train_onehot = np.eye(58)[Y_train.reshape(-1)].T
Y_test_onehot = np.eye(58)[Y_test.reshape(-1)].T

# Define the neural network architecture
layer_dims = [X_train_flatten.shape[0], 512, 256, 128, 64, 58]

# Initialize and train the neural network
neural_network = NeuralNetwork(layer_dims)
neural_network.train_neural_network(X_train_flatten, Y_train_onehot, learning_rate=0.001, num_iterations=1000)

# Make predictions on the test set
predictions = neural_network.predict(X_test_flatten)

# Calculate accuracy
accuracy = np.mean(predictions == Y_test)
print(f'Test accuracy: {accuracy}')

