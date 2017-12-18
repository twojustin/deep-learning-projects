import sys
import argparse
import json
import numpy as np
from my_answers import NeuralNetwork, iterations, learning_rate, hidden_nodes, output_nodes
from prep_data import train_features, train_targets, val_features, val_targets

def MSE(y, Y):
    return np.mean((y-Y)**2)



####################
### Set the hyperparameters in you myanswers.py file ###
####################

FLAGS = None


def train():
    iterations = FLAGS.iterations or 3000
    learning_rate = FLAGS.learning_rate or 1
    hidden_nodes = FLAGS.hidden_nodes or 12
    output_nodes = 1

    N_i = train_features.shape[1]
    network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

    losses = {'train':[], 'validation':[]}
    print('iterations', FLAGS)
    for ii in range(iterations):
        iteration(network, losses, ii)


def iteration(network, losses, ii):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    X, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']

    network.train(X, y)

    # Printing out the training progress
    train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
    if (ii+1) % 10 == 0 and ii > (FLAGS.iterations/2):
        print(json.dumps({"epoch": ii, "validation loss": val_loss,
                          "learning_rate": FLAGS.learning_rate,
                          "hidden_nodes": FLAGS.hidden_nodes}))
    # sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \
    #                  + "% ... Training loss: " + str(train_loss)[:5] \
    #                  + " ... Validation loss: " + str(val_loss)[:5])
    # sys.stdout.flush()

    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=3000,
                        help='Number of iterations to run trainer')
    parser.add_argument('--learning_rate', type=float, default=1,
                        help='Initial learning rate')
    parser.add_argument('--hidden_nodes', type=int, default=12)
    FLAGS, unparsed = parser.parse_known_args()

    train()
