import tensorflow as tf
from api import API
from matplotlib import pyplot as plt
import numpy as np
import random

class Data(object):
    def __init__(self, size):
        self.data = API.getPreparedData(90,5)
        # Mixing data
        all_batches0 = [[self.data[0][x], self.data[1][x]] for x in range(0, len(self.data[0]))]
        random.seed(80)
        random.shuffle(all_batches0)
        # Splitting data into batches
        all_batches = [[], []]
        all_batches[0] = [x[0] for x in all_batches0]
        all_batches[1] = [x[1] for x in all_batches0]
        all_batches2 = [[all_batches[0][x:x+size], all_batches[1][x:x+size]] for x in range(0, len(self.data[0]), size)]
        # Splitting into training and testing
        # Batches count
        n2 = len(all_batches2)
        # column count in one line of input
        n = len(all_batches[0])
        train_n2 = int(np.floor(n2 * 0.8))
        self.training_data = all_batches2[0:train_n2]
        train_n = int(np.floor(n * 0.8))
        self.testing_data = [[], []]
        self.testing_data[0] = all_batches[0][train_n:]
        self.testing_data[1] = all_batches[1][train_n:]
        self.batch_no = -1

    def getBatch(self):
        if self.batch_no == len(self.training_data) - 1:
            self.reset()
        self.batch_no += 1
        #input = self.training_data[self.batch_no]
        #output = [x[1] for x in self.training_data[self.batch_no]]
        return [self.training_data[self.batch_no][0], self.training_data[self.batch_no][1]]

    def reset(self):
        random.seed(4)
        random.shuffle(self.training_data)
        self.batch_no = -1


# Structure of
def neural_net(x, weights, biases):
    # Hidden fully connected layer with x neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with x neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Hidden fully connected layer with x neurons
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.add(tf.matmul(tf.tanh(layer_3), weights['out']), biases['out'])

    return out_layer


def tryNeuralNet(data, learning_rate, num_steps, seed):

    # Network Parameters
    n_hidden_1 = 90 # 1st layer number of neurons
    n_hidden_2 = 60 # 2nd layer number of neurons
    n_hidden_3 = 5 # 3rd layer number of neurons
    num_input = 180 # data input (data of 60 minutes)
    num_classes = 5 # number of output minutes

    # tf Graph input
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1], seed)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], seed+1)),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], seed+3)),
        'out': tf.Variable(tf.random_normal([n_hidden_3, num_classes], seed+2))
    }
    biases = {
        'b1': tf.Variable(tf.random_uniform([n_hidden_1], 0, 70, tf.float32, seed+3)),
        'b2': tf.Variable(tf.random_uniform([n_hidden_2], 0, 70, tf.float32, seed+4)),
        'b3': tf.Variable(tf.random_uniform([n_hidden_3], 0, 70, tf.float32, seed+6)),
        'out': tf.Variable(tf.random_uniform([num_classes], 0, 70, tf.float32, seed+5))
    }

    # Construct model
    logits = neural_net(X, weights, biases)
    # prediction = tf.nn.softmax(logits)

    # Mean squared error
    cost = tf.reduce_sum(tf.pow(logits - Y, 2))

    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        for step in range(1, num_steps+1):
            [batch_x, batch_y] = data.getBatch()
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        print("Optimization Finished!")

        # Calculate accuracy for our stocks
        acc, pred, loss = sess.run([accuracy, logits, cost], feed_dict={X: data.testing_data[0],
                                          Y: data.testing_data[1]})
        return pred, loss, acc,

data = Data(100)
loss = []
learns = []
preds = []
acc = []

for learn in range(1, 10):
    res = tryNeuralNet(data, learn/100_00, 1000, 123)
    loss.append(res[1])
    preds.append(res[0])
    acc.append(res[2])
    learns.append(learn/100_00)
    #drawing
    #fig1 = plt.figure(1)
    #plt.subplot(211)
    #plt.plot(range(5), data.testing_data[1][1], range(5), res[0][1])
    #plt.subplot(212)
    #plt.plot(range(5), data.testing_data[1][30], range(5), res[0][30])
    #plt.show()
    #plt.gcf().clear()

plt.plot(learns, loss)
plt.show()
a = 3
#tryNeuralNet(data, 0.2, 20)