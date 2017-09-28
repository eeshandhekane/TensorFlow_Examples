from __future__ import division, print_function, absolute_import
# We have to import tensorflow, named as tf for convenience
import tensorflow as tf
# We have to import the data by downloading it appropriately and name it mnist. This is done through the following two standard steps
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#from tensorflow.examples.tutorials import mnist
# Define variables and placeholders
# Variables are something that the neural network learns, tf computes it for us
# For training data and other type of data that is made available on the go, during training, we use placeholders
# In placeholder, we first declare the datatype and then the tensor size
# None = I do not know yet
# 28, 28, 1 = 28 x 28 image with 1 channel
X = tf.placeholder(tf.float32, [None, 784])
# Weight and biases are variables that will be learnt
# The sizes of these tensors will be fixed and are mentioned. We initialize them using tf.zeros(), by specifying all the weights with appropriate dimensions to be 0.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
init = tf.initialize_all_variables()
# The process of training is essentially figuring out the values of the variables in the process of training
# tf.reshape() changes the tensor shape appropriately. -1 as the only argument flattens out the tensor. -1 alongwith other dimensions treats -1 as unknown and calculates that shape by itself
Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b)
# Y_true is fed on the go and hence, is unknown and represented as a placeholder. Also, None defines the "I do not know" dimension.
Y_true = tf.placeholder(tf.float32, [None, 10])
# * gives the dot-wise/element-wise product. tf.reduce() computes the sum of elements across mentioned dimensions. If nothing is mentioned, a single scalar is returned. If dimension is mentioned, ddition is carried out along that (as in, in 2 x 2 case, axis = 0 mention adds elements of all the rows to compute the sum and hence, a tensor of the column size is retained)
cross_entropy_loss = -tf.reduce_sum(Y_true * tf.log(Y))
# Compute the number of equalities
corr_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_true, 1))
# Analogously, tf.reduce_mean() must be computing the mean. tf.cast typecasts a tensor into another tensor
correct_percent = tf.reduce_mean(tf.cast(corr_pred, tf.float32)) 
# The aboce two steps are required for printing the accuracy of the net, nothing less nothing more!!
# Now, we want to compute the gradient of the loss function and backpropagate it
# tf.train.GradientDescentOptimizer() is one of the several available optimizers that can be used. It inputs learning rate and gradient lock, which is avoided by default False.
optim = tf.train.GradientDescentOptimizer(0.0003)
# Make the optimizer minimize the cross entropy loss
training_step = optim.minimize(cross_entropy_loss)
# Steps till now DO NOT produce any numerical values. It just develops a graph in the memory, called the Data-Flow-Graph (DFG).
# The actual computation mechanism needs to feed the placeholders with values and then evaluate some answer
# The minimization step is essentially a FORMULA DERIVATION and thus, needs the computation graph. But the values are fed on the go
itr = 1000
# Iterate over 1000 iterations
# TensorFlow uses the concept of sessions to evaluate the answers. Thus, even if we add two variable tensors and want to see the result, we have to run a session.
# This is cumbersome for smaller applications but beuatiful (as well as SEXY) for training neural networks, where we can simply state the name of the variable to be seen and then just run the session to train for the variables and see the value of the specified variables.
# To strat with, we need to run a session where we can initilize all the variables with ACTUAL numerical values (and not figuratively mention that we want to initialize them)
# This is achieved by running a session on init
sess = tf.Session()
# init is a computation EDGE in the DFG of the above definitions
sess.run(init)
# Now, we are ready to run iterations of training
for i in range(itr):
	# Load X and Y batches from the MNIST dataset
	batch_X, batch_Y = mnist.train.next_batch(100)
	# Feed this batch as the input using DICTIONARY to the appropriate keys to the placeholders. In our case, these are X, Y
	training_batch = { X: batch_X, Y_true: batch_Y }
	# Now, we run a session to train the network using a batch. What should this session be trained on?? It should be the training_step!
	# Note that tf looks at the variable to run the session on and then starts to go back the DFG to see all its dependencies and then figure out their values. 
	# This way, it rolls back to variables from training_step -> cross_entropy_loss -> Y_true, Y. At this stage, it sees that Y_true placeholder is required and feels the value of this placeholder using the input dictionary input in the session run command.
	# Now, the Y is tracked back as-- Y -> X, W, b. Note that X is a placeholde which is filled with the dictionary content and W, b are Variables, where the graph is halted, because W, b need to be learnt.
	# Now, placeholder field, and W, b with initialized values (previous run session), everything is computed till cross_entropy_loss as defined (along the computation of variables mentioned above that appear on the DFG leading to cross_entropy_loss). After training_step needs to be evaluated, which changes the values of all the involved variables so that the loss is minimized. In this way, indirectly and withoutany need of mentioning the gradients, we can train the variables
	sess.run(training_step, feed_dict = training_batch)
	# Now the weights are updated, we need to see the performance on training images batch, which is defined in terms of corr_pred and correct_percent.
	# To evaluate these, correct_percentage -> corr_pred -> Y_true, Y is the trace-back. Y_true is loaded from the placeholder.
	# Y -> X, W, b, all of which can be computed either from training_batch and Variable values respectively
	# Thus, the forward computation starts at X, W, b, Y_true and computes the desired entities. Note that no values get changed in these as no NODE (computation demanded by the edge entities) changes the value of any variable
	# Note that we can savor the values of the accuracy and percentage of success by assigning them to some entities
	corr, acc = sess.run([corr_pred, correct_percent], feed_dict = training_batch)
	print('[TRAINING]  Correct Predictions = ' + str(corr) + ' Percentage Accuracy = ' + str(acc))
	# Now, if we may want to test the performance of the intermediate trained model on test batch, so we can extract out a batch and calculate the same error measures by passing as the values of the placeholders the test batch that we have picked
	# We first load a batch for testing. We load the entire MNIST test data in the batch (makes sense!!)
	testing_batch = { X: mnist.test.images, Y_true: mnist.test.labels}
	#corr, acc = sess.run([corr_pred, correct_percent], feed_dict = testing_batch)
	#print('[TESTING]  Correct Predictions = ' + str(corr) + ' Percentage Accuracy = ' + str(acc))
# It must be noted that the learning rate is extremely crucial in the convergence. With lr = 0.03, we get accuracy only of 12% or so. It builds up at 92% with 0.003
