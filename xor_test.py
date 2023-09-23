import numpy as np
from dense import Dense
from activation import Tanh
from losses import mse, mse_prime

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

epochs = 10000
learning_rate = 0.1
# training
for e in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        # forward propagation
        output = x
        for layer in network:
            output = layer.forward(output)

        # error calculation
        error += mse(y, output)

        # backward propagation
        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

        error /= len(X)
        print('%d/%d, error=%f' % (e + 1, epochs, error))
