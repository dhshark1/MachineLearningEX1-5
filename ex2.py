# Daniel Haber
# 322230020

import numpy as np
import sys


def make_matrix(data):
    # split all of the data into a list, breaking it up by \n symbols
    matrix = data.read().splitlines()
    for i in range(len(matrix)):
        # split by commas
        matrix[i] = matrix[i].split(",")
    matrix = np.array(matrix).astype(np.float64)
    return matrix


def normalize_matrix_zscore(matrix):
    # transpose the matrix so we can calculate the st dev and mean of each column easily
    transposed_matrix = np.transpose(matrix)
    for i in range(len(matrix[0])):
        # calculate mean and st dev of each column
        mean = np.mean(transposed_matrix[i])
        st_dev = transposed_matrix[i].std(0)
        # iterate through each column and insert normalized z-score value
        for j in range(len(matrix)):
            matrix[j][i] = (matrix[j][i] - mean) / st_dev
    return matrix


def perceptron(x_train, y_train):
    # initialize a matrix with dimensions 3x6 and filled with zeros
    w = np.zeros((3, 6))
    eta = 0.1
    # 500 epochs
    for e in range(500):
        # initialize an array with numbers 0 to the number of examples, evenly spaced
        # and then choose one random number from that array
        rand = np.arange(len(x_train))
        np.random.shuffle(rand)
        if e % 100 == 0 and e != 0:
            eta /= 2
        for x, y in zip(x_train[rand], y_train[rand]):
            # perform dot product between the input vector and the weights matrix
            # and then take the index where the maximum value appears
            y_hat = np.argmax(np.dot(w, x))
            # update the weights if our prediction is incorrect, eta is 0.1
            if y != y_hat:
                # strengthen the column that is the true class
                w[y, :] = w[y, :] + (eta * x)
                # weaken the column that we misclassified
                w[y_hat, :] = w[y_hat, :] - (eta * x)
    return w


def svm(x_train, y_train):
    # initialize a matrix with dimensions 3x6 and filled with zeros
    w = np.zeros((3, 6))
    eta = 0.1
    # 500 epochs
    for e in range(500):
        # initialize an array with numbers 0 to the number of examples, evenly spaced
        # and then choose one random number from that array
        rand = np.arange(len(x_train))
        np.random.shuffle(rand)
        if e % 100 == 0 and e != 0:
            eta /= 2
        for x, y in zip(x_train[rand], y_train[rand]):
            # perform dot product between the input vector and the weights matrix
            # and then take the index where the maximum value appears
            y_hat = np.argmax(np.dot(w, x))
            # update the weights if our prediction is incorrect, eta is 0.1 and lambda is 0.01
            if y != y_hat:
                # strengthen the column that is the true class
                w[y, :] = w[y, :] * (1 - eta * 0.01) + np.multiply(eta, x)
                # weaken the column that we misclassified
                w[y_hat, :] = w[y_hat, :] * (1 - eta * 0.01) - np.multiply(eta, x)
                remaining_index = 3 - y - y_hat
                w[remaining_index, :] = w[remaining_index, :] * (1 - eta * 0.01)
    return w


def passiveAggressive(x_train, y_train):
    # initialize a matrix with dimensions 3x6 and filled with zeros
    w = np.zeros((3, 6))
    # 500 epochs
    for e in range(500):
        # initialize an array with numbers 0 to the number of examples, evenly spaced
        # and then choose one random number from that array
        rand = np.arange(len(x_train))
        np.random.shuffle(rand)
        for x, y in zip(x_train[rand], y_train[rand]):
            # perform dot product between the input vector and the weights matrix
            # and then take the index where the maximum value appears
            y_hat = np.argmax(np.dot(w, x))
            # update the weights if our prediction is incorrect
            if y != y_hat:
                loss = max(0, 1 - (np.dot(w[y, :], x)) + (np.dot(w[y_hat, :], x)))
                tau = (loss / (2 * (np.linalg.norm(x) ** 2)))
                w[y, :] = w[y, :] + (tau * x)
                w[y_hat, :] = w[y_hat, :] - (tau * x)
    return w


def knn(x_train, y_train, x, k):
    x_copy = x_train.copy()
    y_copy = y_train.copy()
    # initialize an array of neighbours with size k
    neighbours = [99999.0] * k
    # initialize an array of clusters with size k
    clusters = [0] * k
    neighbours = np.array(neighbours)
    clusters = np.array(clusters)
    maxValue = neighbours.max()
    maxIndex = neighbours.argmax()
    # iterate through every row
    for i in range(len(x_copy)):
        # calculate the distance from the input row to test row
        distance = np.linalg.norm(x_copy[i] - x)
        # if we found a new minimal distance in the neighbours array
        # substitute the new example from the example that's furthest away
        if distance < maxValue:
            # save new distance
            neighbours[maxIndex] = distance
            # the flower we substituted in neighbours now belongs to a different cluster
            # so we update clusters too to be the y of that same row in x
            clusters[maxIndex] = y_copy[i]
            # find the new maximum values
            maxValue = neighbours.max()
            maxIndex = neighbours.argmax()
    # find unique values in clusters along with their counts
    vals, counts = np.unique(clusters, return_counts=True)
    # return the cluster with the most appearances
    index = np.argmax(counts)
    return vals[index]


def main():
    x_train_file = open(sys.argv[1], mode="r")
    y_train_file = open(sys.argv[2], mode="r")
    test_file = open(sys.argv[3], mode="r")
    x_train = make_matrix(x_train_file)
    y_train = np.array(y_train_file.read().splitlines()).astype(int)
    test_x = make_matrix(test_file)
    #normalize values for better learning
    x_train2 = normalize_matrix_zscore(x_train.copy())
    test_x2 = normalize_matrix_zscore(test_x.copy())
    # add bias
    b = np.full(len(x_train2), 1).reshape(len(x_train2), 1)
    x_train2 = np.append(x_train2, b, axis=1)
    b = np.full(len(test_x2), 1).reshape(len(test_x2), 1)
    test_x2 = np.append(test_x2, b, axis=1)
    # learn perceptron
    w_perceptron = perceptron(x_train2, y_train)
    # learn pa
    w_passiveAggressive = passiveAggressive(x_train2, y_train)
    # learn svm
    w_svm = svm(x_train2, y_train)

    output_file = open(sys.argv[4], mode="w")
    for x1, x2 in zip(test_x, test_x2):
        knn_yhat = knn(x_train, y_train, x1, 10)
        perceptron_yhat = np.argmax(np.dot(w_perceptron, x2))
        pa_yhat = np.argmax(np.dot(w_passiveAggressive, x2))
        svm_yhat = np.argmax(np.dot(w_svm, x2))
        output_file.write(f"knn: {knn_yhat}, perceptron: {perceptron_yhat}, svm: {svm_yhat}, pa: {pa_yhat}\n")


if __name__ == "__main__":
    main()
