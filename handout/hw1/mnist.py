"""Problem 3 - Training on MNIST"""
import numpy as np

# TODO: Import any mytorch packages you need (XELoss, SGD, etc)
from mytorch.nn.loss import CrossEntropyLoss
from mytorch.nn.sequential import Sequential
from mytorch.nn.batchnorm import BatchNorm1d
from mytorch.nn.linear import Linear
from mytorch.nn.activations import ReLU
from mytorch.optim.sgd import SGD
from mytorch.tensor import Tensor

# NOTE: Batch size pre-set to 100. Shouldn't need to change.
BATCH_SIZE = 100

def mnist(train_x, train_y, val_x, val_y):
    """Problem 3.1: Initialize objects and start training
    You won't need to call this function yourself.
    (Data is provided by autograder)
    
    Args:
        train_x (np.array): training data (55000, 784) 
        train_y (np.array): training labels (55000,) 
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        val_accuracies (list(float)): List of accuracies per validation round
                                      (num_epochs,)
    """
    # TODO: Initialize an MLP, optimizer, and criterion
    model = Sequential(Linear(784, 20), BatchNorm1d(20), ReLU(), Linear(20, 10))
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.1)

    # TODO: Call training routine (make sure to write it below)
    val_accuracies = train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=3)
    return val_accuracies


def train(model, optimizer, criterion, train_x, train_y, val_x, val_y, batch_size=100, num_epochs=3):
    """Problem 3.2: Training routine that runs for `num_epochs` epochs.
    Returns:
        val_accuracies (list): (num_epochs,)
    """
    val_accuracies = []
    
    # TODO: Implement me! (Pseudocode on writeup)
    model.train()
    for iter in range(num_epochs):
        print('Iteration: {}'.format(iter))

        train_combined = np.hstack((train_x, np.expand_dims(train_y, 1)))
        np.random.shuffle(train_combined)
        train_x_shuffle = train_combined[:, :train_x.shape[1]]
        train_y_shuffle = train_combined[:, train_x.shape[1]:]

        # print(train_x_shuffle.shape, train_y_shuffle.shape)
        
        
        num_batches = train_x_shuffle.shape[0]/batch_size + 1
        batches = np.vstack((np.array_split(train_x_shuffle, num_batches), np.array_split(train_y_shuffle, num_batches))).T

        for i, (batch_data, batch_labels) in enumerate(batches):
            batch_labels = batch_labels.astype('int32')

            optimizer.zero_grad()
            out = model.forward(Tensor(batch_data, requires_grad=True))
            loss = criterion(out, Tensor(batch_labels))
            loss.backward()
            optimizer.step()
            if i%100 == 0:
                accuracy = validate(model, val_x, val_y)
                val_accuracies.append(accuracy)
                model.train()
        print(val_accuracies[-1])
        print('--------------------------------') 

    return val_accuracies

def validate(model, val_x, val_y, batch_size=100):
    """Problem 3.3: Validation routine, tests on val data, scores accuracy
    Relevant Args:
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        float: Accuracy = correct / total
    """
    #TODO: implement validation based on pseudocode
    model.eval()
    val_combined = np.hstack((val_x, np.expand_dims(val_y, 1)))
    np.random.shuffle(val_combined)
    val_x_shuffle = val_combined[:, :val_x.shape[1]]
    val_y_shuffle = val_combined[:, val_x.shape[1]:]

    num_batches = val_x_shuffle.shape[0]/batch_size + 1
    batches = np.vstack((np.array_split(val_x_shuffle, num_batches), np.array_split(val_y_shuffle, num_batches))).T

    num_correct = 0
    for (batch_data, batch_labels) in batches:
        out = model.forward(Tensor(batch_data, requires_grad=True))
        batch_preds = np.argmax(out.data)
        num_correct += np.sum(batch_preds==batch_labels)

    accuracy = num_correct / len(val_y_shuffle)

    return accuracy
