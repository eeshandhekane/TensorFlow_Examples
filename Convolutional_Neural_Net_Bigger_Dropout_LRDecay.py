# Import all the necessary modules and download the data of mnist
from __future__ import division, print_function, absolute_import
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# Define batch size and iterations
itr = 1000
batch_size = 100


# Define the number of convolutional layer kernels
K1 = 6
K2 = 12
K3 = 24
# Define the number of neurons in fully connected layer
N1 = 200


# Now, there has to be an understanding of why this is done. In the previous network, we obtained some performance. It is astoinishing in itself (at ~97%)
# But, we want to see if we can do better. Also, the graph of the output performance indicates that there is high mismatch in training and testing accuracies and it increases
# Thus, there is some overfitting. To avoid this, we must use dropout. But we should also add some freedom to the net before we start killing neurons in iterations.
# Hence, the bigger network size is justified.


# We define the dropout in the manner shown in the next paragraph. It inputs a probability entry keep_prob.
# What it does is that it either increases the input element by 1/keep_prob or makes it 0 to give the output element.
# This way, the expected sum remains the same but the neurons learn uncorrelated patterns. This avoids the overfitting.
pkeep = 0.75


# Define the placeholders and variables
X = tf.placeholder(tf.float32, [None, 28*28])
Y_true = tf.placeholder(tf.float32, [None, 10])
W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K1], stddev = 0.1)) # 6 x 6 kernels, with 1 grayscale input channel and 6 such kernels. Stride = 1
B1 = tf.Variable(tf.ones([K1])/10)
W2 = tf.Variable(tf.truncated_normal([5, 5, K1, K2], stddev = 0.1)) # 5 x 5 kernels, with K1 input channels and K2 output channels. Stride = 2
B2 = tf.Variable(tf.ones([K2])/10)
W3 = tf.Variable(tf.truncated_normal([5, 5, K2, K3], stddev = 0.1)) # 5 x 5 kernels, with K2 input channels and K3 output channels. Stride = 2
B3 = tf.Variable(tf.ones([K3])/10)
# Now we need to calculate the weight vector size for the fully connected layer
# Note that we will be using padding = 'SAME' passed to the conv2d layer and hence, we can use simpler calculations for the computation of shape of the output feature map
# Note that Stride of 1 does not change the shape of the output feature map. Thus, output feature map is going to be 28 x 28 in shape
# Note that Stride of 2 halves the shape of the feature map with 'SAME' passed. Thus, output feature map is going to be 14 x 14 in shape
# Note that Stride of 2 again halves the shape of the feature map with 'SAME' passed and hence, the output feature map is going to be 7 x 7 in shape
# Notice that there are K3 such feature maps stacked on top of one another in the output of the third convolutional layers because of the number of kernels being K3
# Thus, we need to accept input array (after flattening) of size 7*7*K3 as input and there are 200 neurons in the fully connected layer. Thus, the output array size is 200
W4 = tf.Variable(tf.truncated_normal([7*7*K3, N1], stddev = 0.1))
B4 = tf.Variable(tf.ones([N1])/10)
W5 = tf.Variable(tf.truncated_normal([N1, 10], stddev = 0.1))
B5 = tf.Variable(tf.ones([10])/10)


# Here, we also incorporate decaying learning rate that decreases the learning rate as training progresses
# The idea here is simple, if we are already in a trough of the loss (function of weights), we need to reach the bottom of this valley.
# If step size is large for the valley, we might keep oscillating and this reflects in the performance, where accuracy keeps on oscillating at the end near the optimal value
# Thus, decreasing the learning rate as the training progresses that we smoothly reach the bottom of the valley of the error function
# The arguments to tf.train.exponential_decay() are initial_rate = 0.003, global step = 1000, decay_step = 100, decay_rate = 0.99, staircase = True
learning_rate = tf.train.exponential_decay(0.003, 1000, 100, 0.99, True)


# Initializer for all the variables
init = tf.initialize_all_variables()


# Define the forward pass
# First reshape the input as batch x 28 x 28 x 1 tensor
X1 = tf.reshape(X, [-1, 28, 28, 1])
# Define the forward pass through the convolutional layers
Y1 = tf.nn.relu(tf.nn.conv2d(X1, W1, strides = [1, 1, 1, 1], padding = 'SAME') + B1)
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides = [1, 2, 2, 1], padding = 'SAME') + B2)
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides = [1, 2, 2, 1], padding = 'SAME') + B3)
# DO NOT FORGET TO FLATTEN THIS OUTPUT INTO AN ARRAY. The batch size remains the same and that the shape now becomes array of 7*7*K3 with appropriate batch size
Y4 = tf.reshape(Y3, [-1, 7*7*K3])
# Define the pass through fully connected layers, with dropout
Y5 = tf.nn.dropout(Y4, pkeep)
Y6 = tf.nn.relu(tf.matmul(Y5, W4) + B4)
Y7 = tf.nn.dropout(Y6, pkeep)
Y_pred = tf.nn.softmax(tf.matmul(Y7, W5) + B5)


# Define the loss, optimizer and training step
cross_entropy_loss = -tf.reduce_sum(Y_true * tf.log(Y_pred))
corr_pred = tf.equal(tf.argmax(Y_true, 1), tf.argmax(Y_pred, 1))
corr_perc = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
optim = tf.train.GradientDescentOptimizer(learning_rate)
training_step = optim.minimize(cross_entropy_loss)


# Initialize the variables using run on the initializer
sess = tf.Session()
sess.run(init)


# Training loops
for i in range(itr):
	print('[INFO] Iteration : ' + str(i))
	# Get the training data
	batch_X, batch_Y = mnist.train.next_batch(batch_size)
	# Run the training step
	training_data = { X: batch_X, Y_true: batch_Y }
	sess.run(training_step, feed_dict = training_data)
	# Calculate accuracy on training data
	corr_tr, acc_tr = sess.run([corr_pred, corr_perc], feed_dict = training_data)
	# Calculate accuracy on testing data
	batch_X_test = mnist.test.images
	batch_Y_test = mnist.test.labels
	testing_data = { X: batch_X_test, Y_true: batch_Y_test }
	corr_te, acc_te = sess.run([corr_pred, corr_perc], feed_dict = testing_data)
	print('[TRAINING] Correct Predictions = ' + str(corr_tr) + ' Accuracy = ' + str(acc_tr))
	print('[TESTING] Correct Predictions = ' + str(corr_te) + ' Accuracy = ' + str(acc_te))
