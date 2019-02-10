import pickle
from logistic import *
from svm import *
from solver import *
import numpy as np


def main():
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    logdata = {
        'X_train': data[0][0:500],
        'y_train': data[1][0:500],
        'X_val'  : data[0][500:750],
        'y_val'  : data[1][500:750]
        }
    logistic = LogisticClassifier(input_dim=20)
    logsolver = Solver(logistic, logdata,
                        update_rule='sgd',
                        optim_config={
                            'learning_rate':1,
                            },
                        lr_decay=0.95,
                        num_epochs=50, batch_size=100,
                        print_every=100)
    logsolver.train()

if __name__ == '__main__':
    main()
