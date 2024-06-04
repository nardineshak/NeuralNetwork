import numpy as np
import matplotlib.pyplot as plt

# purpose of this function is to initialize the weight & bias 
# that will be used in calculating the vector values using sigmoid function
def init_params(layer_dims):
    np.random.seed(3)
    params = {}
    L = len(layer_dims)

    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return params

# Sigmoid function, or an activation function used to shape the 
# output of a neuron. Produces a value between 0 and 1.
# Z (linear hypothesis) - Z = W*X + b , 
# W - weight matrix, b- bias vector, X- Input 
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

# Forward propagation, take the values from a prev layer and give it 
# as input to the next layer. 
# function params: training data and params
def forward_prop(X, params):
    A = X  # input the first year i.e. training data 
    caches = []
    L = len(params) // 2

    for l in range(1, L + 1):
        A_prev = A
    
        # linear hypothesis 
        Z = np.dot(params['W' + str(l)], A_prev) + params['b' + str(l)]

        # storing the linear cache 
        linear_cache = (A_prev, params['W' + str(l)], params['b' + str(l)])

        # apply sigmoid on linear hypothesis 
        A, activation_cache = sigmoid(Z)
        # storing both linear and activation cache
        cache = (linear_cache, activation_cache)
        caches.append(cache)
    
    return A, caches

# As cost function decreases, the performance becomes better.
# Algorithms such as Gradient Descent are used to update these 
# values in such a way that the cost function is minimized.
def cost_function(A, Y):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    cost = np.squeeze(cost)  # Ensure the cost is a scalar value
    return cost

# Gradient Descent needs updating terms called gradients which are calculated
# using backpropagation. Gradient values are calculated for each neuron in the
# network and it represents the change in the final output with respect to the 
# change in the parameters of that particular neuron
def one_layer_backward(dA, cache):
    linear_cache, activation_cache = cache

    Z = activation_cache
    dZ = dA * sigmoid(Z)[0] * (1 - sigmoid(Z)[0])  # derivative of sigmoid

    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def backprop(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = one_layer_backward(dAL, current_cache)
    
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = one_layer_backward(grads["dA" + str(l + 2)], current_cache)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters['W' + str(l + 1)] -= learning_rate * grads['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] -= learning_rate * grads['db' + str(l + 1)]
    return parameters

def train(X, Y, layer_dims, epochs, lr):
    params = init_params(layer_dims)
    cost_history = []

    for i in range(epochs):
        Y_hat, caches = forward_prop(X, params)
        cost = cost_function(Y_hat, Y)
        cost_history.append(cost)
        grads = backprop(Y_hat, Y, caches)
        params = update_parameters(params, grads, lr)
        if i % 100 == 0:
            print(f"Cost after {i} epochs: {cost}")
        
    return params, cost_history

# Example of running the neural network

# Sample data (for illustration purposes, replace with your actual data)
X = np.random.randn(3, 10)  # 3 features, 10 examples
Y = np.random.randint(0, 2, (1, 10))  # Binary target

# Define layer dimensions (3-layer neural network)
layer_dims = [3, 4, 2, 1]  # 3 input features, 4 units in 1st hidden layer, 2 units in 2nd hidden layer, 1 output unit

# Training parameters
epochs = 1000
learning_rate = 0.01

# Train the neural network
params, cost_history = train(X, Y, layer_dims, epochs, learning_rate)

# Plot the cost function
plt.plot(cost_history)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost function over epochs')
plt.show()
