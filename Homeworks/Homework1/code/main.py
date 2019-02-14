import pickle
from logistic import *
from svm import *
from softmax import *
from solver import *
import numpy as np


def main():
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    mnist_xtest = []
    mnist_xtrain = []
    mnist_ytest = []
    mnist_ytrain = []
    logdata = {
        'X_train': data[0][0:500],
        'y_train': data[1][0:500],
        'X_val'  : data[0][500:750],
        'y_val'  : data[1][500:750]
        }
    test_feat = data[0][750:1000]
    test_label = data[1][750:1000]
    with open('mnist_train.csv', 'r') as f:
        for line in f:
            proc = [int(c) for c in line.strip().split(',')]
            mnist_ytrain.append(proc[0])
            mnist_xtrain.append(proc[1:])
    with open('mnist_test.csv', 'r') as f:
        for line in f:
            proc = [int(c) for c in line.strip().split(',')]
            mnist_ytest.append(proc[0])
            mnist_xtest.append(proc[1:])

    logdata = {
        'X_train': data[0][0:500],
        'y_train': data[1][0:500],
        'X_val'  : data[0][500:750],
        'y_val'  : data[1][500:750]
        }
    softmaxdata = {
        'X_train': np.array(mnist_xtrain[0:4000]),
        'y_train': np.array(mnist_ytrain[0:4000]),
        'X_val'  : np.array(mnist_xtrain[4000:5000]),
        'y_val'  : np.array(mnist_ytrain[4000:5000])
        }

   #print('logistic')
    #logistic = LogisticClassifier(input_dim=20)
    #logsolver = Solver(logistic, logdata,
    #                    update_rule='sgd',
    #                    optim_config={
    #                        'learning_rate':70,
    #                        },
    #                    lr_decay=0.95,
    #                    num_epochs=50, batch_size=100,
    #                    print_every=100)
    #logsolver.train()

    #print('hidden logistic')
    #hidden_logistic = LogisticClassifier(input_dim=20, hidden_dim=5)
    #hidden_logsolver = Solver(hidden_logistic, logdata,
    #                    update_rule='sgd',
    #                    optim_config={
    #                        'learning_rate':1.86,
    #                        },
    #                    lr_decay=0.95,
    #                    num_epochs=50, batch_size=100,
    #                    print_every=100)
    #hidden_logsolver.train()

    print('svm')
    svm = SVM(input_dim=20)
    svmsolver = Solver(svm, logdata,
                        update_rule='sgd',
                        optim_config={
                            'learning_rate':6.25,
                            },
                        lr_decay=0.95,
                        num_epochs=50, batch_size=100,
                        print_every=100)
    svmsolver.train()

    #print('hidden svm')
    #hidden_svm = SVM(input_dim=20, hidden_dim=5, reg=0.1)
    #hidden_svmsolver = Solver(hidden_svm, logdata,
    #                    update_rule='sgd',
    #                    optim_config={
    #                        'learning_rate': 3,
    #                        },
    #                    lr_decay=0.95,
    #                    num_epochs=50, batch_size=100,
    #                    print_every=100)
    #hidden_svmsolver.train()
    #print(hidden_svmsolver.check_accuracy(logdata['X_train'], logdata['y_train'], num_samples=500))

	#print('softmax')
    #softmax = SoftmaxClassifier(input_dim=784)
    #softmaxsolver = Solver(softmax, softmaxdata,
    #                    update_rule='sgd',
    #                    optim_config={
    #                        'learning_rate':50,
    #                        },
    #                    lr_decay=0.95,
    #                    num_epochs=50, batch_size=100,
    #                    print_every=100)
    #softmaxsolver.train()

    #print('hidden svm')
    #hidden_softmax = SVM(input_dim=20, hidden_dim=5, reg=0.1)
    #hidden_softmaxsolver = Solver(hidden_softmax, logdata,
    #                    update_rule='sgd',
    #                    optim_config={
    #                        'learning_rate': 3,
    #                        },
    #                    lr_decay=0.95,
    #                    num_epochs=50, batch_size=100,
    #                    print_every=100)
    #hidden_softmaxsolver.train()
    #print(hidden_svmsolver.check_accuracy(logdata['X_train'], logdata['y_train'], num_samples=500))

if __name__ == '__main__':
    main()
