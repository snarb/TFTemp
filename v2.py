import tensorflow as tf
import itertools
import numpy as np
from random import randint
import random
from sklearn.model_selection import train_test_split
import math
from tensorflow.python.ops import random_ops
import sys


log_dir = 'E:/TBlogs/' + str(randint(0, 100000))
# Parameters
# batch - 50: 690 000
# Rate: 1e-4 - Win! Avg speed: 1.46. Step: 33900
# Rate: 5e-3 - Win! Avg speed: 2.35. Step: 21000
# 1e-2: Win! Avg speed: 3.45. Step: 14100 / 15300
# 1e-1 - fail
learning_rate = 1e-2
training_epochs = 1000000
batch_size = 30
display_step = 300

max_value = 999999
bits_count = len(bin(max_value)) - 2

# Network Parameters
n_hidden_1 = 100 # 1st layer number of features
n_hidden_2 = 100 # 2nd layer number of features
n_input = bits_count
n_output = bits_count
seed = 10000
# Speed: 4.17 Avg speed: 2.35
# Win! Avg speed: 2.35. Step: 21000

def binary_cross_entropy(output, target, epsilon=1e-8, name='bce_loss'):
    """Computes binary cross entropy given `output`.

    For brevity, let `x = output`, `z = target`.  The binary cross entropy loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Parameters
    ----------
    output : tensor of type `float32` or `float64`.
    target : tensor of the same type and shape as `output`.
    epsilon : float
        A small value to avoid output is zero.
    name : string
        An optional name to attach to this layer.

    References
    -----------
    - `DRAW <https://github.com/ericjang/draw/blob/master/draw.py#L73>`_
    """
#     from tensorflow.python.framework import ops
#     with ops.op_scope([output, target], name, "bce_loss") as name:
#         output = ops.convert_to_tensor(output, name="preds")
#         target = ops.convert_to_tensor(targets, name="target")
    with tf.name_scope(name):
        return tf.reduce_mean(-(target * tf.log(output + epsilon) +
                              (1. - target) * tf.log(1. - output + epsilon)))

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
  # curY = np.log(x)
  curY = x
  return curY
#
# def weight_variable(shape):
#       # initial = tf.truncated_normal(shape, stddev=0.1)
#
#       # Estimating fan_in and fan_out is not possible to do perfectly, but we try.
#       # This is the right thing for matrix multiply and convolutions.
#       fan_in = float(shape[-2])
#       fan_out = float(shape[-1])
#       for dim in shape[:-2]:
#           fan_in *= float(dim)
#           fan_out *= float(dim)
#
#       trunc_stddev = math.sqrt(1.3 * 2.0 / fan_in)
#       return random_ops.truncated_normal(shape, 0.0, trunc_stddev)

      # return tf.Variable(initial)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, seed=seed)
    return tf.Variable(initial)


def bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  # initial = tf.constant(0.0, shape=shape)
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
  hidden1 = nn_layer(x, n_input, n_hidden_1, 'layer1')
  hidden2 = nn_layer(hidden1, n_hidden_1, n_hidden_2, 'layer2')
  out_layer = nn_layer(hidden2, n_hidden_2, n_output, 'outLayer', act = tf.nn.sigmoid)
  return out_layer

def bitfield(n):
  strRepr = bin(n)[2:]
  strRepr = strRepr.zfill(bits_count)

  res = []
  for i in range(bits_count):
    res.append(strRepr[i])

  return res
# return [int(digit) for digit in bin(n)[2:]]

def GenPermutations():
    res = []
    for i in range(max_value):
        res.append(i)

    return res

def ToTFbatch(curList):
    return np.array(curList).reshape(batch_size, bits_count)


allVarsX = GenPermutations()


# convertedXList = []
# convertedYList = []
#
# for curXval in allVarsX:
#     bitXlist = bitfield(curXval)
#     convertedXList.append(bitXlist)
#
#     curY = GetTargetResult(curXval)
#     bitYlist = bitfield(curY)
#     convertedYList.append(bitXlist)

  # y_in = GetTargetResult(x_in)

# for curYval in allVarsX:
#  isT = 0
#  bitYlist = bitfield(curYval)
#  convertedYList.append(bitYlist)

x_train, x_test = train_test_split(allVarsX, test_size = 0.2)

# convertedXList = random.shuffle(convertedXList)
# convertedYList = random.shuffle(convertedYList)
#
# trainCount = int(len(convertedXList) * 0.8)
# testCount = len(convertedXList) - trainCount
# x_train = convertedXList[:trainCount]
# x_test = convertedXList[trainCount:]
# y_train = convertedYList[:trainCount]
# y_test = convertedXList[trainCount:]

# trainCount = int(len(allVarsX) * 0.8)
# testCount = len(allVarsX) - trainCount
# x_train = allVarsX[:trainCount]
# x_test = allVarsX[trainCount:]

# Launch the graph.
sess = tf.InteractiveSession()

x_data = tf.placeholder(tf.float32, [None, n_input])
y_data = tf.placeholder(tf.float32, [None, n_output])

# Construct model
pred = multilayer_perceptron(x_data)

# Minimize the mean squared errors.
# loss = tf.reduce_mean(tf.abs(pred - y_data))
# loss = tf.reduce_mean(tf.square(pred - y_data))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y_data))

loss = binary_cross_entropy(pred, y_data)
# meanLoss = tf.reduce_mean(loss)
tf.summary.scalar('loss', loss)

# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_data, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
tf.global_variables_initializer().run()

prevTestAc = 0
numSteps = 0
acSum = 0
win = False

for step in range(training_epochs):
  # x_in = np.random.rand(batch_size, 1).astype(np.float32)

  # x_in = np.random.randint(low=0, high=max_value - 1, size=(batch_size))

  batch = random.sample(x_train, batch_size)

  convertedXList = []
  convertedYList = []
  for curXval in batch: #np.nditer(x_in)
    bitXlist = bitfield(curXval)
    bitYlist = bitfield(GetTargetResult(curXval))
    convertedYList.append(bitYlist)
    convertedXList.append(bitXlist)

  # y_in = GetTargetResult(x_in)

  # for curXval in x_train: #np.nditer(x_in)
   #   isT = 0
   #   bitYlist = bitfield(curXval)
   #   convertedYList.append(bitYlist)

  # curX = np.random.randint(low=0, high=max_value, size=(1))
  # curY = GetTargetResult(curX)
  # bitXlist = bitfield(curX)
  # bitXlist = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  # xAr = np.array(bitXlist).reshape(1, 20)
  # yAr = xAr

  # bitYlist = bitfield(curY)
  # yAr = np.array(bitYlist).reshape(1, 20)
  # curPrediction = sess.run(pred, feed_dict={x_data: xAr})
  # curLoss = sess.run(loss, feed_dict={x_data: xAr, y_data: yAr})

  feedDict = {x_data: ToTFbatch(convertedXList), y_data: ToTFbatch(convertedYList)}
  _, trainAccuracy = sess.run([train_step, accuracy], feed_dict = feedDict)

  if step % display_step == 0:
      convertedXList = []
      convertedYList = []

      test_batch = random.sample(x_train, batch_size)

      for curXval in test_batch:
          bitXlist = bitfield(curXval)
          bitYlist = bitfield(GetTargetResult(curXval))
          convertedYList.append(bitYlist)
          convertedXList.append(bitXlist)

      # Record execution stats
      feedDict = {x_data: ToTFbatch(convertedXList), y_data: ToTFbatch(convertedYList)}
      testAccuracy = sess.run(accuracy, feed_dict=feedDict)

      print('Train accuracy: {0:.1f}%. Test accuracy: {1:.1f}%.'.format(trainAccuracy * 100, testAccuracy * 100))

      speed = 50000 * (testAccuracy - prevTestAc) / display_step
      prevTestAc = testAccuracy
      acSum += speed
      numSteps += 1
      avg = acSum / numSteps
      print('Speed: {0:.2f} Avg speed: {1:.2f}'.format(speed, avg))
      if(testAccuracy >= 0.99  and not win):
        print('Win! Avg speed: {0:.2f}. Step: {1}. learning_rate: {2}'.format(avg, step * batch_size, learning_rate))
        win = True
        sys.exit()

      if(step > 20000):
           print("Lose")
      #   run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      #   run_metadata = tf.RunMetadata()
      #   summary, _ = sess.run([merged, train_step],
      #   feed_dict=feedDict,
      #   options=run_options,
      #   run_metadata=run_metadata)
      #
      #   train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
      #   train_writer.add_summary(summary, step)
      #   print('Adding run metadata for', step)
  else:
      pass
     # summary, _ = sess.run([merged, train_step], feed_dict=feedDict)
     # train_writer.add_summary(summary, step)

  # if(step % display_step == 0):
   #   curX = np.random.randint(low=0, high=max_value, size=(1))
   #   curY = GetTargetResult(curX)
   #   bitXlist = bitfield(curX)
   #   xAr = np.array(bitXlist).reshape(1, 20)
   #   bitYlist = bitfield(curY)
   #   yAr = np.array(bitYlist).reshape(1, 20)
  #
   #   curPrediction = sess.run(pred, feed_dict={x_data: xAr})
   #   curLoss = sess.run(loss, feed_dict={x_data: xAr, y_data: yAr})
   #   print("For x = {0} and target y = {1} prediction was y = {2} and squared loss was = {3}".format(curX, curY,curPrediction, lossValue))