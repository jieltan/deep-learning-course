import torch
import numpy as np
from math import sqrt
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import time
import pickle
import bcolz



def loader(dictionary, pad=15):

    print("Loading Train data")
    X_train = []
    y_train = []


    f = open('../../Homework2/code/data/train.txt')
    for l in f:
        y_train.append(int(l[0]))
        line = l[2:].split()
        temp = []
        count = 0
        for item in line:

            if item in dictionary:
                temp.append(dictionary[item])
                count += 1
            if count == pad:
                break

        while count < pad:
            for item in line:

                if item in dictionary:
                    temp.append(dictionary[item])
                    count += 1
                if count == pad:
                    break

        X_train.append(temp)


    y_train = np.asarray(y_train).reshape(-1,1)



    print("Loading Test data")
    X_test = []
    y_test = []

    f = open('../../Homework2/code/data/test.txt')
    for l in f:
        y_test.append(int(l[0]))
        line = l[2:].split()
        temp = []
        count = 0

        for item in line:

            if item in dictionary:
                temp.append(dictionary[item])
                count += 1
            if count == pad:
                break

        while count < pad:
            for item in line:
                if item in dictionary:
                    temp.append(dictionary[item])
                    count += 1
                if count == pad:
                    break

        X_test.append(temp)


    y_test = np.asarray(y_test).reshape(-1,1)



    print("Loading Unlabelled data")
    X_unlabelled = []


    f = open('../../Homework2/code/data/unlabelled.txt')
    for l in f:
        line = l[2:].split()
        temp = []
        count = 0
        for item in line:

            if item in dictionary:
                temp.append(dictionary[item])
                count += 1
            if count == pad:
                break

        while count < pad:
            for item in line:

                if item in dictionary:
                    temp.append(dictionary[item])
                    count += 1
                if count == pad:
                    break

        X_unlabelled.append(temp)


    return X_train, y_train, X_test, y_test, X_unlabelled

def create_emb_layer(weights_matrix, non_trainable=False):
    #print(weights_matrix.shape)
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    #emb_layer.load_state_dict({'weight': weights_matrix})
    #if non_trainable:
    #    emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


class cnn(nn.Module):
    def __init__(self, weights_matrix, pool='avg',kernel=5,input_dim=15,num_filter=128):
        super().__init__()

        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)

        self.embed = nn.Embedding(num_embeddings,embedding_dim)
        self.conv = nn.Conv1d(embedding_dim, num_filter, kernel)
        if pool == 'avg':
            self.pool = nn.AvgPool1d(input_dim-kernel+1)
        else:
            self.pool = nn.MaxPool1d(input_dim-kernel+1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(num_filter, 1)
        self.sig = nn.Sigmoid()


    def init_weights(self):
        C_in = self.fc.weight.size(1)
        nn.init.normal_(self.fc.weight, 0.0, 1 / sqrt(C_in))
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        emb = self.embed(x.long())
        #print(emb.shape)
        h1 = self.conv(emb.permute(0,2,1))
        h2 = self.pool(h1).squeeze()
        #print(h2.shape)
        h3 = self.fc(self.relu(h2))
        h4 = self.sig(h3)
        return h4


def train(trainloader, net, criterion, optimizer, device):
    for epoch in range(8):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        for i, (representations, labels) in enumerate(trainloader):
            representations = representations.to(device).float()
            labels = labels.to(device).float()

            optimizer.zero_grad()
            output = net(representations)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, end - start))
                start = time.time()
                running_loss = 0.0
    print('Finished Training')




def test(testloader, net, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            representations, labels = data
            representations = representations.to(device).float()
            labels = labels.to(device).float()
            outputs = net(representations)
            outputs[outputs < 0.5] = 0
            outputs[outputs >= 0.5] = 1
            total += labels.size(0)
            correct += (outputs == labels).sum().item()
    print('Accuracy: %d %%' % (
        100 * correct / total))



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dictionary = {}
    index = 0
    with open('data/train.txt', 'r') as f:
        for l in f:
            line = l[2:].split()
            for item in line:
                if item not in dictionary:
                    dictionary[item] = index
                    index += 1


    X_train, y_train, X_test, y_test, X_unlabelled = loader(dictionary)

    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)
    X_unlabelled = torch.tensor(X_unlabelled)

    trainset = data_utils.TensorDataset(X_train, y_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=True)

    testset = data_utils.TensorDataset(X_test, y_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False)

    unlabelledset = data_utils.TensorDataset(X_unlabelled)
    unlabelledloader = data_utils.DataLoader(unlabelledset, batch_size=100, shuffle=False)




    matrix_len = len(dictionary)
    weights_matrix = np.zeros((matrix_len, 50))

    for i, word in enumerate(dictionary):
        weights_matrix[i] = np.random.normal(scale=0.6, size=(50,))



    net = cnn(weights_matrix,pool='max',kernel=5).to(device)
    net.init_weights()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.8)

    train(trainloader, net, criterion, optimizer, device)
    test(testloader, net, device)

    #f = open('predictions_q1.txt', 'w')

    #for data in unlabelledloader:
    #    info, = data
    #    output = net(info.to(device).float())
    #    output[output < 0.5] = int(0)
    #    output[output >= 0.5] = int(1)
    #    for item in output:
    #        f.write(str(int(item.item())))
    #        f.write("\n")
    #f.close()

if __name__ == "__main__":
    main()

