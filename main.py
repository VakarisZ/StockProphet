import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from tensorflow.python.data import Dataset

from api import API

def showFrame(data):
    print(data.describe())
    data.plot()
    plt.show()

def showArray(label,data):
    fig1 = plt.figure(label)
    ax = fig1.gca()
    plt.ylabel(label)
    ax.plot(data)

def plotStockData(data):
    showArray('average', data[:, 0])
    showArray('marketNumberOfTrades', data[:, 1])
    showArray('marketVolume', data[:, 2])
    plt.show()


def main():
    #data = API.getMonthsData('atvi')
    #API.printToFileJson('atvi_new.json', data)
    #API.printToFileVerbose('atvi_verbose.txt', data)
    all_data = pandas.read_json(API.getJsonFromFile('atvi_new.json'), 'records')
    all_data = all_data.drop(['changeOverTime', 'date', 'high', 'low', 'marketChangeOverTime', 'marketClose', 'average',
                   'marketNotional', 'marketOpen', 'notional', 'numberOfTrades', 'open', 'volume', 'close', 'minute', 'label',
                              'marketHigh', 'marketLow'], 1)
    all_data = all_data.values
    #Data count
    n = all_data.shape[0]
    #Colum count
    p = all_data.shape[1]
    #Spliting into training and testing
    train_n = int(np.floor(n * 0.8))
    training_data = all_data[0:train_n]
    #training_data2 = all_data[np.arange(0, train_n), :]
    testing_data = all_data[train_n:]
    #Scaling data to interval [0:1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    training_data = scaler.fit_transform(training_data)
    testing_data = scaler.transform(testing_data)

    # Build X and y
    X_train = training_data[:, 1:]
    y_train = training_data[:, 0]
    X_test = testing_data[:, 1:]
    y_test = testing_data[:, 0]

    # Initializers
    sigma = 1
    weight_initializer = tensorflow.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tensorflow.zeros_initializer()

    # Network
    # Model architecture parameters
    n_stocks = 500
    n_neurons_1 = 1024
    n_neurons_2 = 512
    n_neurons_3 = 256
    n_neurons_4 = 128
    n_target = 1
    # Placeholder
    X = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, n_stocks])
    Y = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None])

    #hidden weights
    W_hidden_1 = tensorflow.Variable(weight_initializer([n_stocks, n_neurons_1]))
    bias_hidden_1 = tensorflow.Variable(bias_initializer([n_neurons_1]))
    W_hidden_2 = tensorflow.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tensorflow.Variable(bias_initializer([n_neurons_2]))
    W_hidden_3 = tensorflow.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
    bias_hidden_3 = tensorflow.Variable(bias_initializer([n_neurons_3]))
    W_hidden_4 = tensorflow.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
    bias_hidden_4 = tensorflow.Variable(bias_initializer([n_neurons_4]))

    # Output weights
    W_out = tensorflow.Variable(weight_initializer([n_neurons_4, 1]))
    bias_out = tensorflow.Variable(bias_initializer([1]))

    # Hidden layer
    hidden_1 = tensorflow.nn.relu(tensorflow.add(tensorflow.matmul(X, W_hidden_1), bias_hidden_1))
    hidden_2 = tensorflow.nn.relu(tensorflow.add(tensorflow.matmul(hidden_1, W_hidden_2), bias_hidden_2))
    hidden_3 = tensorflow.nn.relu(tensorflow.add(tensorflow.matmul(hidden_2, W_hidden_3), bias_hidden_3))
    hidden_4 = tensorflow.nn.relu(tensorflow.add(tensorflow.matmul(hidden_3, W_hidden_4), bias_hidden_4))

    # Output layer (must be transposed)
    out = tensorflow.transpose(tensorflow.add(tensorflow.matmul(hidden_4, W_out), bias_out))

    # Cost function
    mse = tensorflow.reduce_mean(tensorflow.squared_difference(out, Y))

    # Optimizer
    opt = tensorflow.train.AdamOptimizer().minimize(mse)

    # Initializers
    sigma = 1
    weight_initializer = tensorflow.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tensorflow.zeros_initializer()

    # Make Session
    net = tensorflow.Session()
    # Run initializer
    net.run(tensorflow.global_variables_initializer())

    # Setup interactive plot
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(y_test)
    line2, = ax1.plot(y_test * 0.5)
    plt.show()

    # Number of epochs and batch size
    epochs = 10
    batch_size = 256

    for e in range(epochs):

        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]

        # Minibatch training
        for i in range(0, len(y_train) // batch_size):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]
            # Run optimizer with batch
            net.run(opt, feed_dict={X: batch_x, Y: batch_y})

            # Show progress
            if np.mod(i, 5) == 0:
                # Prediction
                pred = net.run(out, feed_dict={X: X_test})
                line2.set_ydata(pred)
                plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
                file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.jpg'
                plt.savefig(file_name)
                plt.pause(0.01)
    # Print final MSE after Training
    mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
    print(mse_final)

if __name__ == "__main__":
    main()