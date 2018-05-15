import tensorflow as tf
from api import API
from matplotlib import pyplot as plt
import numpy as np
import pandas
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

class Data(object):
    def __init__(self):
        self.data = API.getPreparedData(30,5)
        # Data count
        n = len(self.data[0])
        # column count in one line of input
        p = len(self.data[0][0])
        # Spliting into training and testing
        train_n = int(np.floor(n * 0.8))
        self.training_data = [self.data[0][0:train_n], self.data[1][0:train_n]]
        self.testing_data = [self.data[0][train_n:], self.data[1][train_n:]]
        self.batch_no = -1

    def getBatch(self, size):
        self.batch_no += 1
        if len(self.training_data[0]) > (size*(self.batch_no+1)):
            return self.training_data[0][self.batch_no*size:(self.batch_no+1)*size][:], \
                   self.training_data[1][self.batch_no*size:(self.batch_no+1)*size][:]
        else:
            self.batch_no = -1
            return self.getBatch(size)

    def reset(self):
        self.batch_no = -1


# Structure of
def neural_net(x, weights, biases):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.softmax(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.softmax(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(tf.tanh(layer_2), weights['out']) + biases['out']
    return out_layer

def tryNeuralNet(data, learning_rate, num_steps):
    # Parameters
    batch_size = int(np.floor(len(data.training_data[0])/num_steps))
    display_step = 100

    # Network Parameters
    n_hidden_1 = 100 # 1st layer number of neurons
    n_hidden_2 = 100 # 2nd layer number of neurons
    num_input = 90 # MNIST data input (img shape: 28*28)
    num_classes = 5 # MNIST total classes (0-9 digits)

    # tf Graph input
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    # Construct model
    logits = neural_net(X, weights, biases)
    #prediction = tf.nn.softmax(logits)
    prediction = logits

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        for step in range(1, num_steps+1):
            batch_x, batch_y = data.getBatch(batch_size)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

        print("Optimization Finished!")

        # Calculate accuracy for MNIST test images
        acc, pred, loss = sess.run([accuracy, prediction, loss_op], feed_dict={X: data.testing_data[0],
                                          Y: data.testing_data[1]})
        print("Testing outputs:", sess.run(prediction, feed_dict={X: data.testing_data[0],
                                          Y: data.testing_data[1]}))
        return pred, loss

data = Data()
results = []
learns = []
for learn in range(1, 9):
    results.append(tryNeuralNet(data, learn/10, 100)[1])
    learns.append(learn/10)
plt.plot(learns, results)
plt.show()