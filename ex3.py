# Daniel Haber
# 322230020
import sys
import numpy as np
from scipy.special import softmax
from scipy.stats import zscore


def sigmoid(Z):
    return 1/(1+np.exp(-Z))


def one_hat_y(y):
    one_hat = np.zeros(10)
    one_hat[y] = 1
    return one_hat


def calc_loss(y_location, h2):
    return -1*np.log(h2[y_location])


def fprop(x, y, parameters):
    W1, b1, W2, b2 = [parameters[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1, x) + b1
    z1 = zscore(z1)
    h1 = sigmoid(z1)
    z2 = np.dot(W2, h1) + b2
    h2 = softmax(z2)
    forward_parameters = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2}
    for key in parameters:
        forward_parameters[key] = parameters[key]
    return forward_parameters


def bprop(forward_parameters):
    x, y, z1, h1, z2, h2 = [forward_parameters[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2')]
    y = y.reshape(10, 1)
    dz2 = (h2 - y)           # dL/dz2
    dW2 = np.dot(dz2, h1.T)  # dL/dz2 * dz2/dw2
    db2 = dz2                # dL/dz2 * dz2/db2
    dz1 = np.dot(forward_parameters['W2'].T, (h2 - y)) * sigmoid(z1) * (1-sigmoid(z1))  # dL/dz2 * dz2/dh1 * dh1/dz1
    dW1 = np.dot(dz1, x.T)   # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1                # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}


def update_parameters(parameters, bprop_parameters):
    W1, b1, W2, b2 = [parameters[key] for key in ('W1', 'b1', 'W2', 'b2')]
    dW1, db1, dW2, db2 = [bprop_parameters[key] for key in ('W1', 'b1', 'W2', 'b2')]
    eta = 0.01
    W1 = W1 - (eta * dW1)
    parameters['W1'] = W1
    W2 = W2 - (eta * dW2)
    parameters['W2'] = W2
    b1 = b1 - (eta * db1)
    parameters['b1'] = b1
    b2 = b2 - (eta * db2)
    parameters['b2'] = b2
    return parameters


def train_neural_network(train_x, train_y):
    # initialize parameters
    epochs = 36
    W1 = np.random.uniform(-1, 1, (128, 784))
    b1 = np.ones(128)
    b1 = np.reshape(b1, (128, 1))
    W2 = np.random.uniform(-1, 1, (10, 128))
    b2 = np.ones(10)
    b2 = np.reshape(b2, (10, 1))
    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    best_parameters = parameters
    min_loss = 20
    # shuffle data
    randomize = np.arange(len(train_x))
    np.random.shuffle(randomize)
    train_x, train_y = train_x[randomize], train_y[randomize]
    numtrain = int(0.8 * len(train_x))
    # make 80% validation set (to minimize the loss)
    validation_x, validation_y = train_x[numtrain:], train_y[numtrain:]
    train_x, train_y = train_x[:numtrain], train_y[:numtrain]
    for e in range(epochs):
        randomize = np.arange(len(train_x))
        np.random.shuffle(randomize)
        train_x, train_y = train_x[randomize], train_y[randomize]
        for x, y in zip(train_x, train_y):
            x = np.reshape(x, (784, 1))
            fprop_param = fprop(x, one_hat_y(y), parameters)
            brop_param = bprop(fprop_param)
            parameters = update_parameters(parameters, brop_param)
        # calculate the average loss on validation set and save if it improved
        loss = 0
        for x, y in zip(validation_x, validation_y):
            x = np.reshape(x, (784, 1))
            loss += calc_loss(y, fprop(x, one_hat_y(y), parameters)['h2'])
        avg_loss = loss / len(validation_x)
        if avg_loss < min_loss:
            min_loss = avg_loss
            best_parameters = parameters
    return best_parameters


def main():
    train_x = np.loadtxt(sys.argv[1])
    train_y = np.loadtxt(sys.argv[2], dtype=int)
    test_x = np.loadtxt(sys.argv[3])
    train_x /= 255.0
    test_x /= 255.0
    neural_network = train_neural_network(train_x, train_y)
    output_file = open("test_y", "w+")
    for x in test_x:
        x = np.reshape(x, (784, 1))
        fprop_param = fprop(x, 0, neural_network)
        y_hat = np.argmax(fprop_param['h2'])
        output_file.write(f"{y_hat}\n")
    output_file.close()



if __name__ == "__main__":
    main()
