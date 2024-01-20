import numpy as np
from sklearn.preprocessing import LabelEncoder

class NeuralNetwork:
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.parameters = self.initialize_parameters()

    def initialize_parameters(layer_dims):
        ''' This layer basically takes dimensions from the dimension list and generates random paramenters'''
        parameters = {} #Dictionary to serve as a layer
        L = len(layer_dims)-1
        # For each layer, except the last
        for l in range(1, L+1):
            #Generate weights

            # Dimensions: (NN in forward layer, NN in backward layer)
            parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            
            #Generate bias (NN in forward layer, no. of points)
            parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))

        return parameters

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z))
        return exp_Z / exp_Z.sum(axis=0, keepdims=True)

    def forward_propagation(X,parameters): #Take input of images and random/fine tuned parameters
        #Store linear combination and activation matrice
        print("Forward Propogation")
        caches = [(0,X)]

        #A(0), shape (back layer,points)
        A = X 
        print("Dimensions of A(0):",A.shape)

        #Find the number of layers, number start from 0
        L = len(parameters)//2
        for i in range(1,L):
            
            print(f"Dimensions of W({l}):",parameters[f'W{l}'].shape)
            print(f"Dimensions of b({l}):",parameters[f'b{l}'].shape)

            # Z(l) = W(l)A(l-1)+b(l)
            # (number of nodes in lth layer,m) = (nn in lth,pixel)
            Z = np.dot(parameters[f'W{l}'], A) + parameters[f'b{l}']
            print(f"Dimensions of Z({l}):",Z.shape)
            
            A = relu(Z) # Then apply the function to create the activated output.
            print(f"Dimensions of A({l}):",Z.shape)
            
            #store Z(l), A(l)
            caches.append((Z,A))

        print(f"Dimensions of W(L) or W({L}):",parameters[f'W{L}'].shape)
        print(f"Dimensions of b(L) or b({L}):",parameters[f'b{L}'].shape)
        
        Z = np.dot(parameters[f'W{L}'], A) + parameters[f'b{L}']

        print(f"Dimensions of Z(L) or Z({L}):",Z.shape)
        
        AL = softmax(Z)

        print(f"Dimensions of AL or A({L}):",AL.shape)
        
        caches.append((Z, AL))
        
        return AL, caches

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -np.sum(np.multiply(Y, np.log(AL))) / m
        return cost

    def backward_propagation(self, AL, Y, caches):
        #gradient
        grads = {}
        print("Back Propogation")
        
        #total number of layers
        L = len(caches) 

        #number of points
        m = AL.shape[1] 

        #Ensuring dimensions
        print("Shape of Y before", Y.shape)
        Y = Y.reshape(AL.shape) 
        print("Shape of Y after", Y.shape)

        # (Labels,data points) = (Labels,data points)-(Labels,data points)
        dZ = AL - Y # Error in the prediction

        #dW(L) = (Labels,data points) (A(4).T)
        # (58,64) = (58,4058)(4058,64)
        dW = np.dot(dZ, caches[L - 1][1].T) / m#derivative of loss function
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(self.parameters[f'W{L}'].T, dZ)
        grads[f'dW{L}'] = dW
        grads[f'db{L}'] = db

        for l in reversed(range(1, L)):
            dZ = np.multiply(dA_prev, np.int64(caches[l][0] > 0))
            dW = np.dot(dZ, caches[l - 1][1].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dA_prev = np.dot(self.parameters[f'W{l}'].T, dZ)
            grads[f'dW{l}'] = dW
            grads[f'db{l}'] = db

        return grads

    def update_parameters(self, grads, learning_rate):
        L = len(self.parameters) // 2
        for l in range(1, L + 1):
            self.parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']
        return self.parameters

    def train_neural_network(self, X_train, Y_train, learning_rate, num_iterations):
        np.random.seed(1)
        for i in range(num_iterations):
            AL, caches = self.forward_propagation(X_train)
            cost = self.compute_cost(AL, Y_train)
            grads = self.backward_propagation(AL, Y_train, caches)
            self.parameters = self.update_parameters(grads, learning_rate)

        

    def predict(self, X):
        AL, _ = self.forward_propagation(X)
        predictions = np.argmax(AL, axis=0)
        return predictions


    def save_parameters(self, filename='trained_model.npz'):
        np.savez(filename, **self.parameters)

    def load_parameters(self, filename='trained_model.npz'):
        loaded_params = np.load(filename)

from Preprocess import Preprocess

f = np.load('data_file.npz')
Y_tr = f['train_labels']
X_train = f['train_data']
Y_t = f['test_labels']
X_test = f['test_data']

# Convert the labels to integers using LabelEncoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_tr)
Y_test_encoded = label_encoder.fit_transform(Y_t)


#
Y_train = np.array(Y_train_encoded, dtype=int)
Y_test = np.array(Y_test_encoded, dtype=int)

#This is because we start with (m,pixels) then we go to (pixels,m)
# Assuming you have X_train, Y_train, X_test, Y_test from your dataset
X_train_flatten = X_train.T / 255.0
X_test_flatten = X_test.T / 255.0

# Convert labels to one-hot encoding
# Y_train_onehot = one_hot(Y_train)
# Y_test_onehot = one_hot(Y_test)

Y_train_onehot = np.eye(58)[Y_train.reshape(-1)].T
Y_test_onehot = np.eye(58)[Y_test.reshape(-1)].T

# print(Y_train_onehot.shape)

# Define the neural network architecture (you can adjust these dimensions)
layer_dims = [X_train_flatten.shape[0], 512, 256,128,64, 58]

# Train the neural network
parameters = train_neural_network(X_train_flatten, Y_train_onehot, layer_dims, learning_rate=0.001, num_iterations=)



# Make predictions on the test set
predictions = predict(X_test_flatten, parameters)

# Calculate accuracy
accuracy = np.mean(predictions == Y_test)
print(f'Test accuracy: {accuracy}')
