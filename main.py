import tensorflow as tf
import numpy as np
from random import randint


log_dir = '/home/pkonovalov/TBlogs/' + str(randint(0, 100000))
# Parameters
learning_rate = 0.01
training_epochs = 1000000
batch_size = 50
display_step = 700

# Network Parameters
n_hidden_1 = 50 # 1st layer number of features
n_hidden_2 = 10 # 2nd layer number of features
n_input = 1

max_value = 999999
bits_count = len(bin(max_value)) - 2



def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def GetTargetResult(x):
    curY = np.log(x)
    return curY

  # We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations

# Create model
def multilayer_perceptron(x):
    hidden1 = nn_layer(x, bits_count, n_hidden_1, 'layer1')
    hidden2 = nn_layer(hidden1, n_hidden_1, n_hidden_2, 'layer2')
    out_layer = nn_layer(hidden2, n_hidden_2, bits_count, 'outLayer', act = tf.identity)
    return out_layer

def bitfield(n):
    return [int(digit) for digit in bin(n)[2:]]

# Launch the graph.
sess = tf.InteractiveSession()

x_data = tf.placeholder(tf.float32, [None, 1])
y_data = tf.placeholder(tf.float32, [None, 1])

# Construct model
pred = multilayer_perceptron(x_data)


# Minimize the mean squared errors.
# loss = tf.reduce_mean(tf.abs(pred - y_data))
loss = tf.reduce_mean(tf.square(pred - y_data))

# meanLoss = tf.reduce_mean(loss)
tf.summary.scalar('loss', loss)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)


# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
tf.global_variables_initializer().run()


for step in range(training_epochs):
    # x_in = np.random.rand(batch_size, 1).astype(np.float32)

    x_in = np.random.uniform(low=0.0, high=1.0, size=(batch_size, 1)).astype(np.float32)
    y_in = GetTargetResult(x_in)
    feedDict = {x_data: x_in, y_data: y_in}
    _, lossValue = sess.run([train_step, loss], feed_dict = feedDict)


    if step % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feedDict,
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
        train_writer.add_summary(summary, step)
        print('Adding run metadata for', step)
    else:
        summary, _ = sess.run([merged, train_step], feed_dict=feedDict)
        train_writer.add_summary(summary, step)




    if(step % display_step == 0):
        curX = np.random.uniform(low=0.0, high=1.0, size=(1, 1)).astype(np.float32)
        curY =  GetTargetResult(curX)

        curPrediction = sess.run(pred, feed_dict={x_data: curX})
        curLoss = sess.run(loss, feed_dict={x_data: curX, y_data: curY})
        print("For x = {0} and target y = {1} prediction was y = {2} and squared loss was = {3}".format(curX, curY,curPrediction, lossValue))

