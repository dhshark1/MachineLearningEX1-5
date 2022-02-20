#Daniel Haber 322230020
#Ron Dolgoy 311319099

from gcommand_dataset import GCommandLoader
import torch
import ntpath
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class conv_net(nn.Module):
    def __init__(self):
        super(conv_net, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 20, 3, 1, 2), nn.BatchNorm2d(20), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3))
        self.layer2 = nn.Sequential(nn.Conv2d(20, 50, 3, 1, 2), nn.BatchNorm2d(50), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3))
        self.layer3 = nn.Sequential(nn.Conv2d(50, 100, 3, 1, 2), nn.BatchNorm2d(100), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3))
        self.layer4 = nn.Sequential(nn.Conv2d(100, 500, 3, 1, 2), nn.BatchNorm2d(500), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2000, 120)
        self.bn1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn2 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 30)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


def train(model, optimizer, train_loader):
    model.train()
    running_loss = 0.0
    correct = 0.0
    for data,labels in tqdm(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        running_loss += loss.item() * data.size(0)
        output = output.max(1, keepdim=True)[1]
        correct += output.eq(labels.view_as(output)).cpu().sum()
        loss.backward()
        optimizer.step()
    accuracy = 100. * correct / len(train_loader.dataset)
    running_loss /= len(train_loader.dataset)
    print('Train: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        running_loss, correct, len(train_loader.dataset), accuracy))
    return running_loss, accuracy


def validation(model, validation_loader):
    model.eval()
    correct = 0
    average_loss = 0
    with torch.no_grad():
        for data, target in tqdm(validation_loader):
            output = model(data)
            average_loss += F.nll_loss(output, target, reduction='sum').item()
            predict = output.max(1, keepdim=True)[1]
            correct += predict.eq(target.view_as(predict)).cpu().sum()
        size = len(validation_loader.dataset)
        average_loss /= size
        accuracy = 100. * correct / size
        print('Validation: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            average_loss, correct, size, accuracy))
        return average_loss, accuracy


def main():
    validation_set = GCommandLoader('./valid')
    test_set = GCommandLoader('./test')
    data_set = GCommandLoader('./train')
    classes = list(data_set.class_to_idx)
    train_loader = DataLoader(data_set, batch_size=128, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=128, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)
    test_files_name = list(map(lambda x: ntpath.basename(x[0]), test_loader.dataset.spects))
    model = conv_net()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_losses = []
    train_accuracy = []
    validation_losses = []
    validation_accuracy = []

    for epoch in range(28):
        loss, accuracy = train(model, optimizer, train_loader)
        train_losses.append(loss)
        train_accuracy.append(accuracy)
        loss, accuracy = validation(model, validation_loader)
        validation_losses.append(loss)
        validation_accuracy.append(accuracy)
    print(train_losses)
    print(train_accuracy)
    print(validation_losses)
    print(validation_accuracy)

    model.eval()
    to_print = [None] * 6836
    with torch.no_grad():
        for k, (data, label) in enumerate(tqdm(test_loader)):
            output = model(data)
            predicted = torch.max(output.data, 1)[1]
            for i in range(len(predicted)):
                wav_name = str(test_files_name[k + i])
                label = str(classes[int(predicted[i])])
                index = int(wav_name[:-4])-6836
                to_print[index] = [wav_name, label]
    f = open("test_y", "w+")
    for item in to_print:
        if item is None:
            continue
        f.write(f"{item[0]},{item[1]}\n")
    f.close()


if __name__ == '__main__':
    main()