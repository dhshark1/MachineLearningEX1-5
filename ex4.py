#Daniel Haber, Ron Dolgoy
#322230020, 311319099

import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt

MODEL_AB_SECOND_LAYER = 50
MODEL_AB_FIRST_LAYER = 100
MODEL_E_FIRST_LAYER = 128
MODEL_E_SECOND_LAYER = 64
MODEL_E_THIRD_LAYER = 10
MODEL_E_FOURTH_LAYER = 10
MODEL_E_FIFTH_LAYER = 10


def draw_graphs(acc_t, acc_v, loss_t, loss_v):
    num_epochs = np.arange(1, 26)
#     plt.subplot(FIRST_PLOT)
    plt.plot(num_epochs, acc_t, label='train')
    plt.plot(num_epochs, acc_v, label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Per Epoch')
    plt.legend()
    plt.show()
#     plt.subplot(SECOND_PLOT)
    plt.plot(num_epochs, loss_t, label='train')
    plt.plot(num_epochs, loss_v, label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss Per Epoch')
    plt.legend()
    plt.show()


class model_AB(nn.Module):
    def __init__(self, image_size):
        super(model_AB, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, MODEL_AB_FIRST_LAYER)
        self.fc1 = nn.Linear(MODEL_AB_FIRST_LAYER, MODEL_AB_SECOND_LAYER)
        self.fc2 = nn.Linear(MODEL_AB_SECOND_LAYER, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class model_C(nn.Module):
    def __init__(self, image_size):
        super(model_C, self).__init__()
        self.image_size = image_size
        self.dropout = nn.Dropout(0.2)
        self.fc0 = nn.Linear(image_size, MODEL_AB_FIRST_LAYER)
        self.fc1 = nn.Linear(MODEL_AB_FIRST_LAYER, MODEL_AB_SECOND_LAYER)
        self.fc2 = nn.Linear(MODEL_AB_SECOND_LAYER, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.dropout(F.relu(self.fc0(x)))
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class model_D(nn.Module):
    def __init__(self, image_size):
        super(model_D, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, MODEL_AB_FIRST_LAYER)
        self.bn0 = nn.BatchNorm1d(MODEL_AB_FIRST_LAYER)
        self.fc1 = nn.Linear(MODEL_AB_FIRST_LAYER, MODEL_AB_SECOND_LAYER)
        self.bn1 = nn.BatchNorm1d(MODEL_AB_SECOND_LAYER)
        self.fc2 = nn.Linear(MODEL_AB_SECOND_LAYER, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class model_E(nn.Module):
    def __init__(self, image_size):
        super(model_E, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, MODEL_E_FIRST_LAYER)
        self.fc1 = nn.Linear(MODEL_E_FIRST_LAYER, MODEL_E_SECOND_LAYER)
        self.fc2 = nn.Linear(MODEL_E_SECOND_LAYER, MODEL_E_THIRD_LAYER)
        self.fc3 = nn.Linear(MODEL_E_THIRD_LAYER, MODEL_E_FOURTH_LAYER)
        self.fc4 = nn.Linear(MODEL_E_FOURTH_LAYER, MODEL_E_FIFTH_LAYER)
        self.fc5 = nn.Linear(MODEL_E_FIFTH_LAYER, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)


class model_F(nn.Module):
    def __init__(self, image_size):
        super(model_F, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, MODEL_E_FIRST_LAYER)
        self.fc1 = nn.Linear(MODEL_E_FIRST_LAYER, MODEL_E_SECOND_LAYER)
        self.fc2 = nn.Linear(MODEL_E_SECOND_LAYER, MODEL_E_THIRD_LAYER)
        self.fc3 = nn.Linear(MODEL_E_THIRD_LAYER, MODEL_E_FOURTH_LAYER)
        self.fc4 = nn.Linear(MODEL_E_FOURTH_LAYER, MODEL_E_FIFTH_LAYER)
        self.fc5 = nn.Linear(MODEL_E_FIFTH_LAYER, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)


class model_G(nn.Module):
    def __init__(self, image_size):
        super(model_G, self).__init__()
        self.image_size = image_size
        self.dropout = nn.Dropout(0.2)
        self.fc0 = nn.Linear(image_size, 392)
        self.bn0 = nn.BatchNorm1d(392)
        self.fc1 = nn.Linear(392, 196)
        self.bn1 = nn.BatchNorm1d(196)
        self.fc2 = nn.Linear(196, 98)
        self.bn2 = nn.BatchNorm1d(98)
        self.fc3 = nn.Linear(98, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.dropout(F.relu(self.bn0(self.fc0(x))))
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def train(model, optimizer, data_set):
    model.train()
    x = data_set[0]
    y = data_set[1]
    batch_size = 64
    complete_iter = int(len(x) / batch_size)
    end = batch_size
    for i in range(complete_iter):
        data = x[end - batch_size: end]
        labels = y[end - batch_size: end]
        end += 64
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
    data = x[end-batch_size:]
    labels = y[end-batch_size:]
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, labels)
    loss.backward()
    optimizer.step()


def run_model(model, optimizer, train_set, validation_set):
    loss_v = np.zeros(25)
    loss_t = np.zeros(25)
    acc_v = np.zeros(25)
    acc_t = np.zeros(25)
    for epoch in range(25):
        train(model, optimizer, train_set)
        acc_t[epoch], loss_t[epoch] = test(train_set, model)
        acc_v[epoch], loss_v[epoch] = test(validation_set, model)
    draw_graphs(acc_t, acc_v, loss_t, loss_v)


def test(data_set, model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        output = model(data_set[0])
        test_loss += F.nll_loss(output, data_set[1], reduction='sum').item()
        prediction = output.max(1, keepdim=True)[1]
        correct += prediction.eq(data_set[1].view_as(prediction)).cpu().sum()
        accuracy = 100. * correct / len(data_set[0])
        test_loss /= len(data_set[0])
        print('\nTest set: Average loss: {:.4f}, Accuracy: {} / {} ({:.0f}%)\n'.format(test_loss, correct,
                                                                               len(data_set[0]), accuracy))
    return accuracy, test_loss


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


def main():
    train_x = np.loadtxt(sys.argv[1])
    train_y = np.loadtxt(sys.argv[2], dtype=int)
    test_x = np.loadtxt(sys.argv[3])
    output_log = sys.argv[4]
    train_x = np.array(train_x, dtype=float)
    train_y = np.array(train_y, dtype=int)
    test_x = np.array(test_x, dtype=float)
    train_x = normalize_matrix_zscore(train_x)
    test_x = normalize_matrix_zscore(test_x)
    train_x = torch.FloatTensor(train_x)
    train_y = torch.LongTensor(train_y)
    test_x = torch.FloatTensor(test_x)
    randomize = np.arange(len(train_x))
    np.random.shuffle(randomize)
    train_x, train_y = train_x[randomize], train_y[randomize]
    numtrain = int(0.8 * len(train_x))
    # make 80% validation set (to minimize the loss over)
    validation_x, validation_y = train_x[numtrain:], train_y[numtrain:]
    # train_x, train_y = train_x[:numtrain], train_y[:numtrain]
    train_data = [train_x, train_y]
    validation_data = [validation_x, validation_y]
    model = model_G(image_size=784)
    # sgd_optimizer = optim.SGD(model.parameters(), lr=0.1)
    # run_model(model, sgd_optimizer, train_data, validation_data)
    adam_optimizer = optim.Adam(model.parameters(), lr=0.0005)
    run_model(model, adam_optimizer, train_data, validation_data)
    model.eval()
    output = model(test_x)
    prediction = output.max(1, keepdim=True)[1]
    f = open(output_log, "w+")
    for y in prediction:
        f.write(f"{int(y)}\n")
    f.close()


if __name__ == "__main__":
    main()