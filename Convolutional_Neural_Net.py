# Import all the necessary modules and download the data of mnist
from __future__ import division, print_function, absolute_import
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# An RGB image has 3 channels of images size W x H, thus the tensor shape is W x H x 3.
# We consider a neuron that has as many weights as is the "kernel size" x "kernel size" x "previous channel size = 3". Thus, now a neuron corresponds to a one kernel.
# The output is a feature map that has width "W - kernel size + 1" x "H - kernel size + 1" x "number of previous channel neurons = previous channel count".
# In this manner, the number of channels, and feature map size propagate in layers.
# Thus, in the video's example, W1 = W1[ 4 x 4 x 3 ] for one neuron in the first hidden layer.
# Now, in the second neuron of the first hidden layer, we will have W2 = W2[ 4 x 4 x 3 ]
# Thus, W, the net weight tensor can be written as a concatenation of Wi's along the last dimension: W = W[ 4 x 4 x 3 x 2 ]. Here, 4 x 4 is the kernel size, 3 is the input channel number, 2 is the output channel number.
# Unless there is a padding, the feature size decreases. But this decrease has to be drastic and that the NN has to output only 10 numbers.
# The subsampling is carried out using the pooling layers.
# Another way to do subsampling is to vary the stride.


# Now, the CNN that we aim to build is as follows--
# Input shape: I[ Batch Size x 28 x 28 x 1 ]
# First layer weights: W1[ 5 x 5 x 1 x 4 ], i.e., the kernel is 5 x 5 wide, there is 1 input channel (because of the black-white/grayscale image) and 4 kernels and stride 1
# Second layer weights: W2[ 4 x 4 x 4 x 8 ], i.e., 4 x 4 kernel size, with 4 input channels, 8 kernels (output channels) with stride 2
# Third layer weights: W3[ 4 x 4 x 8 x 12 ], i.e., 4 x 4 kernel size, with 8 input channels, 12 kernels (output channels) with stride 2
# Then, we add a fully connected layer for classifier task followed by a softmax layer of size 10
# All the layers appropriately have added biases
# Now, the input image (primary feature map) has size 28 x 28
# The output of the first layer gives feature map of size 28 x 28 (There has to be a padding for this to happen, right?) because of the stride of 1
# Then, the output of the second layer gives feature map of size 14 x 14 because of the stride of 2
# Then, the output of the third layer gives feature map of size 7 x 7 because of the stride of 2
# Now, note that the last convolutional layer has feature map size 7 x 7 with 12 such feature maps (12 channels outputted).
# This needs to be flattened into a 7*7*12 sized array and passed against a 200 neuron fully connected layer which gives 200 outputs
# The end softmax layer inputs this 200 sized vector and outputs a 10 sized array, giving 10 softmax outputs.
# Thus, the fully connected layer has W4 = W4[ 7*7*12, 200]
# And, finally the softmax layer has W5 = W5[ 200, 10]


# Define iterations and batch size
itr = 1000
batch_size = 100


# Define the kernels per convolutional layer
K1 = 4
K2 = 8
K3 = 12 
# Define the neurons per fully connected layer
N1 = 200


# There needs to be a discussion on the size of the bias vector
# Note that the first hidden layer has total output size of [ 5 x 5 x 1 x 4 ], i.e., there are 4 output channels, i.e., there are 4 hidden layers
# Thus, each layer will get its own bias. This bias will be appropriately broadcasted to match the feature map shape requirements


# Define the weights and biases of the convolutional layers
# EXTREMELY CRUCIAL: THE VALUE OF A TF feed_dict entry can not be a tensor in itself. Thus, we need to perform all the reshaping inside the loop only
X = tf.placeholder(tf.float32, [None, 28*28])
Y_true = tf.placeholder(tf.float32, [None, 10])
W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K1], stddev = 0.1))
B1 = tf.Variable(tf.ones([K1])/10)
W2 = tf.Variable(tf.truncated_normal([4, 4, K1, K2], stddev = 0.1))
B2 = tf.Variable(tf.ones([K2])/10)
W3 = tf.Variable(tf.truncated_normal([4, 4, K2, K3], stddev = 0.1))
B3 = tf.Variable(tf.ones([K3])/10)
# Define the weights and biases of the fully connected layer
W4 = tf.Variable(tf.truncated_normal([7*7*K3, N1], stddev = 0.1))
B4 = tf.Variable(tf.ones([N1])/ 10)
W5 = tf.Variable(tf.truncated_normal([N1, 10], stddev = 0.1))
B5 = tf.Variable(tf.ones([10])/10)


# NEVER EVER FORGET TO INITIALIZE THE VARIABLES
init = tf.initialize_all_variables()


# Some minor (actually, major) details are in order. The input tensor of [ batch, height, width, input_channel ] with weight vector of [ kernel_size, kernel_size, input_channel, output_channel ], the conv2d layer flattens the weight vector as a 2D matrix-- W[ kernel_size*kernel_size*input_channel, out_channel] and extracts image patch of I[ batch, out_height, out_width, kernel_size*kernel_size*input_channel ] and performs I*W to get output of the shape [ batch, out_height, out_width, out_channel ]. The exact mathematical expression is not very relevant.
# The other details is stride. WE MUST HAVE [1, stride_horizontal, stride_vertical, 1] as the stride parameter.


# Define the forward pass thorugh the convolutional layers
X1 = tf.reshape(X, [-1, 28, 28, 1])
Y1 = tf.nn.relu(tf.nn.conv2d(X1, W1, strides = [1, 1, 1, 1], padding = 'SAME') + B1)
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides = [1, 2, 2, 1], padding = 'SAME') + B2)
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides = [1, 2, 2, 1], padding = 'SAME') + B3)
# Define the pass through the flattening layers
Y4 = tf.reshape(Y3, [-1, 7*7*K3])
Y5 = tf.nn.relu(tf.matmul(Y4, W4) + B4)
Y_pred = tf.nn.softmax(tf.matmul(Y5, W5) + B5)


# Define the loss and optimizer
cross_entropy_loss = -tf.reduce_sum(Y_true * tf.log(Y_pred))
corr_pred = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_true, 1))
corr_perc = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
optim = tf.train.GradientDescentOptimizer(0.003)
training_step = optim.minimize(cross_entropy_loss)


# Initialize using run
sess = tf.Session()
sess.run(init)


# Training loops
for i in range(itr):
	# Get the training batch
	batch_X, batch_Y = mnist.train.next_batch(batch_size)
	# Run the training for one iteration
	training_data = { X: batch_X, Y_true: batch_Y }
	sess.run(training_step, feed_dict = training_data)
	# Train on the training data batch
	corr_tr, acc_tr = sess.run([corr_pred, corr_perc], feed_dict = training_data)
	print('[TRAINING]  Correct Predictions = ' + str(corr_tr) + ' Percentage Accuracy = ' + str(acc_tr))
	# Test on the entire test data
	batch_X = mnist.test.images 
	batch_Y = mnist.test.labels
	testing_data = { X: batch_X, Y_true: batch_Y }
	corr_te, acc_te = sess.run([corr_pred, corr_perc], feed_dict = testing_data)
	print('[TESTING]  Correct Predictions = ' + str(corr_te) + ' Percentage Accuracy = ' + str(acc_te))
