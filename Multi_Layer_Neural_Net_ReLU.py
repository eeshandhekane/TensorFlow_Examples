# Import all the necessary modules and download the data of mnist
from __future__ import division, print_function, absolute_import
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# We now define the input and weight vectors
# Note that the shape of the tensor in tensorflow can be checked by X.get_shape(). But this returns the shape as TensorShape([Dimesion(d1), Dimension(d2), ...]), which is practically useless
# To get it as a list of integers, use X.get_shape().as_list()


# Define the number of hidden laeyr neurons
N1 = 200
N2 = 100
N3 = 50


# Now, we need to see something. If the input shape is [1, 28*28], then the input is an array of length 784. 
# If there are N1 neurons in the first hidden layer, then there are bound to be 784 weights per neuron (to account for 784 inputs) and these weights will be written as a column
# Thus, if there are N1 such neurons, there are going to be N1 such columns and that defines a matrix for us. We call it W1, which has 784 rows, per column of which represents the weights corresponding to per hidden neuron. Thus, after the multiplication, answer is [1, N1] shaped array. We add bias to it, which in this case, matches perfectly
# The next thing to see is that this is the calculation for all the next layers
# The only bug that remains to solve is how to handle batch_size (does the None take care of it or not)? We will see!


# Another invention with the -1 thing!!
# Let A = tf.Variable(tf.truncated_normal([100, 50, 10, 3], stddev = 0.1))
# If we do B = tf.reshape(A, [-1, 500]), we get B.get_shape().as_list() -> [3000, 500]. Of course, the way the values are reordered is a mess and should be avoided if not clear
# If we do multiple -1 passes, then the shape is returned as None
# A = tf.Variable(tf.truncated_normal([100, 50, 10, 3], stddev = 0.1)); B = tf.reshape(A, [-1, -1, 30]), we get B.get_shape().as_list() -> [None, None, 30] as the shape can notbe inferred exactly.
# Unnecessary jugglery should be avoided in this case!!


# We want to have the input as an array of 28*28 size
X = tf.placeholder(tf.float32, [None, 1, 28*28])
# First weight and bias layer
W1 = tf.Variable(tf.truncated_normal([28*28, N1], stddev = 0.1))
B1 = tf.Variable(tf.zeros([N1]))
# Second weight and bias layer
W2 = tf.Variable(tf.truncated_normal([N1, N2], stddev = 0.1))
B2 = tf.Variable(tf.zeros([N2]))
# Third weight and bias layer
W3 = tf.Variable(tf.truncated_normal([N2, N3], stddev = 0.1))
B3 = tf.Variable(tf.zeros([N3]))
# Last output layer:
W4 = tf.Variable(tf.truncated_normal([N3, 10], stddev = 0.1))
B4 = tf.Variable(tf.zeros([10]))


# We need to initialize all the variables as soon as they are defined
init = tf.initialize_all_variables()


# Define the forward pass--the predicted label one hot vector--the true one hot label and the loss--optimizer--step trio
X = tf.reshape(X, [-1, 28*28])
Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y_pred = tf.nn.softmax(tf.matmul(Y3, W4) + B4)
Y_true = tf.placeholder(tf.float32, [None, 10])
# BTW, * is element-wise product, and reduce_sum is the component-wise addition
cross_entropy_loss = -tf.reduce_sum(Y_true * tf.log(Y_pred))
corr_pred = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_true, 1))
corr_perc = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
# Define the optimization now
optim = tf.train.GradientDescentOptimizer(0.003)
training_step = optim.minimize(cross_entropy_loss)


# Now, we define the training iterations
itr = 1000
sess = tf.Session()
# DO NOT FORGET TO RUN THE INITIALIZER. It is the thing that actually feeds in the data into the variables
sess.run(init)
for i in range(itr):
	# Train on a training batch
	batch_X, batch_Y = mnist.train.next_batch(100)
	training_data = { X: batch_X, Y_true: batch_Y }
	sess.run(training_step, feed_dict = training_data)
	# Check the performance on training set
	corr_tr, acc_tr = sess.run([corr_pred, corr_perc], feed_dict = training_data)
	#print('[TRAINING]  Correct Predictions = ' + str(corr) + ' Percentage Accuracy = ' + str(acc), end = '\r')
	# Check the performance on the entire testing data
	batch_X_test = mnist.test.images
	batch_Y_test = mnist.test.labels
	testing_data = { X: batch_X_test, Y_true: batch_Y_test }
	corr_te, acc_te = sess.run([corr_pred, corr_perc], feed_dict = testing_data)
	print('[TRAINING]  Correct Predictions = ' + str(corr_tr) + ' Percentage Accuracy = ' + str(acc_tr), end = '\r')


# Again, the learning rate has an effect. At lr = 0.003, it fails with 12% accuracy. lr needs to be tested and then reported 
# The same is the case with 0.0003
# The same is true with 0.03. What is the problem! The activation function needs to have gradients that do not become 0 quickly.
# Notice that the gradients become 0 very quickly and the acc value stabilizes at 12%, which is very poor
# Upon using ReLU as the activation function, we quickly escalate to the percentage of 17% with 0.03
# With 0.3, performance worsens to 9.8%
# With 0.003, we get at 14%. Now, it seems that the NN can learn but the learning speed is very slow and we need more iterations!!
# Something is still wrong! Why are we stuck at 18%
