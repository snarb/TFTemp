import tensorflow as tf
import numpy as np


def get_target_result(x):
    return np.log(x)


def multilayer_perceptron(x, weights, biases):
    """Create model."""
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Parameters
learning_rate = 0.01
training_epochs = 10**6
batch_size = 500
display_step = 500

# Network Parameters
n_hidden_1 = 50  # 1st layer number of features
n_hidden_2 = 10  # 2nd layer number of features
n_input = 1


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden_2, 1], stddev=0.1))
}

biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden_1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden_2])),
    'out': tf.Variable(tf.constant(0.1, shape=[1]))
}

x_data = tf.placeholder(tf.float32, [None, 1])
y_data = tf.placeholder(tf.float32, [None, 1])

# Construct model
pred = multilayer_perceptron(x_data, weights, biases)

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(pred - y_data))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# train = optimizer.minimize(loss)
train = tf.train.AdamOptimizer(1e-4).minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

for step in range(training_epochs):
    x_in = np.random.rand(batch_size, 1).astype(np.float32)
    y_in = get_target_result(x_in)
    sess.run(train, feed_dict={x_data: x_in, y_data: y_in})
    if(step % display_step == 0):
        curX = np.random.rand(1, 1).astype(np.float32)
        curY = get_target_result(curX)

        curPrediction = sess.run(pred, feed_dict={x_data: curX})
        curLoss = sess.run(loss, feed_dict={x_data: curX, y_data: curY})
        print(("For x = {0} and target y = {1} prediction was y = {2} and "
               "squared loss was = {3}").format(curX, curY,
                                                curPrediction, curLoss))