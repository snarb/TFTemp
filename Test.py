import tensorflow as tf
import numpy as np

log_dir = '/home/pkonovalov/TBlogs'
# Parameters
learning_rate = 0.005
training_epochs = 1000000
batch_size = 500
display_step = 700

# Network Parameters
n_hidden_1 = 50 # 1st layer number of features
n_hidden_2 = 10 # 2nd layer number of features
n_input = 1

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
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations

# Create model
def multilayer_perceptron(x):
    hidden1 = nn_layer(x, 1, n_hidden_1, 'layer1')
    hidden2 = nn_layer(hidden1, n_hidden_1, n_hidden_2, 'layer2')
    out_layer = nn_layer(hidden2, n_hidden_2, 1, 'outLayer', act = tf.identity)
    return out_layer


x_data = tf.placeholder(tf.float32, [None, 1])
y_data = tf.placeholder(tf.float32, [None, 1])

# Construct model
pred = multilayer_perceptron(x_data)

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
    y_in = GetTargetResult(x_in)
    sess.run(train, feed_dict={x_data: x_in, y_data: y_in})
    if(step % display_step == 0):
        curX = np.random.rand(1, 1).astype(np.float32)
        curY = GetTargetResult(curX)

        curPrediction = sess.run(pred, feed_dict={x_data: curX})
        curLoss = sess.run(loss, feed_dict={x_data: curX, y_data: curY})
        print(("For x = {0} and target y = {1} prediction was y = {2} and "
               "squared loss was = {3}").format(curX, curY,
                                                curPrediction, curLoss))