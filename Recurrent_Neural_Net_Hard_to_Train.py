# Recurrent Neural Nets input a vector and there are hidden states that output the tanh of the computation
# Just like sigmoid is a differentiable smooth function from 0 to 1, tanh is a differentiable and smooth function from -1 to 1
# The difference is that the current output of the RNN is fed back to itself as its input in the next instant
# Thus, the way to visualize is this- There is a vector input of size n and hidden layer of size m, then the effective input is some concatenation of the n and m sized arrays
# The net output is the softmax over the hidden state. Refer to the slides and presentation for extra clarity
# The adjustible parameter is the number of hidden state neurons, denoted by N. The input size and the output size are fixed
# Mathematically, X_eff[t] = X[t] | H[t - 1], where ' | ' represents concatenation pipe for generating the effective input
# Then, H[t] = tanh(X_eff[t].W_hidden[t] + B_hidden[t]), which is the computation for the next hidden state
# Then, the output is generated as-- Y[t]= softmax(H[t].W + B), which generates the output
# What is the input vector X? RNNs are usually used for time sequence modeling and hence, X is a feature representation of the input feature.
# As an instance, X can be one-hot encoding of words of a language or letters of an alphabet if we are dealing with NLP tasks using RNN
# What is the output vector Y? It is usually the feature representation of the desired output.
# For instance, it can be the character that we wish to obtain from the previous sequence of character inputs.
# Note that the output, while training, can go wrong because of the wrongness of the weights as well as the wrongness of the state.
# Thus, WEIGHTS AND BIASES ARE SHARED ACROSS ALL THE UNFOLDED INSTANCES OF THE RNN
# We can go DEEP with RNNs as well!! We need to stack multiple hidden states or layers on top of each other, inside each instance, before the output is computed.
# There is a problem-- Empirically, it is observed that one-hot encoding is problematic (without embedding)
# Another problem-- RNNs can not model Long-Term Independencies over long unfolding lengths as the context from previous state is only a fixed length before.
# Thus, any task of reasonable practical utility demands the neural net to be deep in the sense of number of unfoldings it does. But deep nature has inherent problem of itself
# Problem with deep nature of RNN (or, in general, any NN)-- Vanishing/Exploding gradients cause divergence in nets (or, no convergence).
# The solution was invented to be LSTMs. These have gates and the gates can be used to remember or forget the context subject to learnt weights. This helps in modelling long-term dependencies.
# The gated nature can be shown to explain why gradients will not explode in the case of an LSTM.
# Mathematically (OH NO!!), we first concatenate the input with the context: X_eff[t] = X_t | H[t - 1]
# Then, there are three neural-network-like computations at three different gates: (with assumption that non-linear activation is sigmoid)
# Forget Gate:
# f[t] = sigma(X_eff[t].W_f[t] + b_f[t]), where f is the forget gate output, W_f is the weight of forget gate, b_f is the bias of forget gate and sigma is non-linear activation
# Update Gate:
# u[t] = sigma(X_eff[t].W_u[t] + b_u[t]), where u is the update gate output, W_u is the weight of update gate, b_u is the bias of update gate and sigma is non-linear activation
# Result Gate:
# r[t] = sigma(X_eff[t].W_r[t] + b_r[t]), where r is the result gate output, W_r is the weight of result gate, b_r is the bias of result gate and sigma is non-linear activation
# Input:
# X'[t] = tanh(X_eff[t].W_c[t] + b_c[t]), here W_c and b_c are sizing parameters. Note that inside the cell, all the vectors computed lead to arrays of size n
# Whereas, the actual input is of size p and the effective input that we get after concatenation is of size p + n. This p + n size input needs to be mapped to n sized one.
# Now, the LSTM has a hidden state vector H and a memory state vector, denoted for some reason by C.
# Naturally, C should be updated as follows-- current memory should be previous memory scaled by forget gate answer added to the current input scaled by update gate answer. So,
# C[t] = f[t]*C[t - 1] + u[t]*X'[t], where * denotes element-wise product and not a matrix multiplication (which is being denoted by ' . ')
# And (not-so-)naturally, the hidden state should be updated (for some reason) as follows--
# H[t] = r[t]*tanh(C[t])
# A possibly plausible reasoning for this that r tells what proportion of the memory needs to be exposed to the outside as the answer and simultaneously remembered in state
# The output of the LSTM cell at time instant t is then given by the simple NN like equation--
# Y[t] = softmax(H[t]*W[t] + b[t]), where the W and b are weights and biases of the output gate.
# The obvious question here is that WHY WOULD ONE CHOOSE THESE EQUATIONS?? The choice of equations at all the gates and the internal computations seems to be arbitrary.
# The solution is GRU: Gated Recurrent Unit. It is a "cheaper" solution to LSTMs, as LSTMs have 3 necessar gates but GRU has only 2. (Okay, what is the big deal then?)
# The equations go as follows--
# X_eff[t] = X[t] | H[t - 1], has size p + n
# z[t] = sigma(X_eff[t].W_z[t] + b_z[t]), has size n
# r[t] = sigma(X_eff[t].W_r[t] + b_r[t]), has size n
# X'[t] = X[t] | r[t]*H[t - 1], has size p + n
# X''[t] = tanh(X'[t].W_c[t] + b_c[t]), has size n
# H[t] = (1 - z[t])*H[t - 1] + z[t]*X''[t], has size n
# Y[t] = softmax(H[t].W[t] + b[t]), has size m


# Dependencies
import tensorflow as tf
import numpy as np
import os, sys, re


# Extract MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot = True)


# Parameters
BATCH_SIZE = 128
CELL_SIZE = 28
STACK_SIZE = 2
UNFOLD_SIZE = 28 # This is the SEQ_LEN param in google talks
NUM_CLASS = 10
ITR = 10000


# Define placeholders and variables
X = tf.placeholder(tf.float32, [BATCH_SIZE, UNFOLD_SIZE * CELL_SIZE])
H_in = tf.placeholder(tf.float32, [BATCH_SIZE, CELL_SIZE * STACK_SIZE])
# Define classifier weights
W = tf.Variable(tf.truncated_normal([CELL_SIZE * STACK_SIZE, NUM_CLASS], stddev = 0.1))
B = tf.Variable(tf.ones([NUM_CLASS])/10)
Y_true = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASS])


# Define the RNN/LSTM/GRU cell
gru_cell = tf.contrib.rnn.GRUCell(CELL_SIZE)
multi_gru_cell = tf.contrib.rnn.MultiRNNCell([gru_cell]*STACK_SIZE, state_is_tuple = False)
X_ = tf.reshape(X, [BATCH_SIZE, UNFOLD_SIZE, CELL_SIZE])
H_r, H = tf.nn.dynamic_rnn(multi_gru_cell, X_, initial_state = H_in)
#print '[DEBUG] : ', H.shape
# H_r has the list of all the hidden states: [BATCH_SIZE, UNFOLD_SIZE, CELL_SIZE] for each entry in batch, for each time instant, there is an output of cell size
# We device the classification task as using ALL THE INTERMEDIATE STATES in H_r to collectively predict the class of the image
Y_logits = tf.add(tf.matmul(H, W), B) # The output of this fully connected layer are defined as the logits
Y_pred = tf.nn.softmax(Y_logits)
class_pred = tf.argmax(Y_pred, 1)
class_true = tf.argmax(Y_true, 1)
corr_pred = tf.equal(class_pred, class_true)
corr_perc = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
softmax_cross_entropy_with_logits = tf.nn.softmax_cross_entropy_with_logits(logits = Y_logits, labels = Y_true)
optim = tf.train.AdamOptimizer(1e-3)
training_step = optim.minimize(softmax_cross_entropy_with_logits)


# Define a session
sess = tf.Session()


# Initialize all variables
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess.run(init)


# # Define the testing batch
# test_batch_X = mnist.test.next_batch(BATCH_SIZE)
# test_batch_Y = mnist.test.next_batch(BATCH_SIZE)
# print '[DEBUG] ', test_batch_X.shape
# print '[DEBUG] ', test_batch_Y.shape
# THERE IS SOME GLITCH WITH mnist.test.next_batch


# Training loop
for itr in range(ITR):
	# Get the training data
	batch_X, batch_Y = mnist.train.next_batch(BATCH_SIZE)
	# There is a glitch with the mnist.test.next_batch thing. Is there a problem with mnist.train.next_batch??
	# print '[DEBUG] ', batch_X.shape
	# print '[DEBUG] ', batch_Y.shape
	# # NO, THERE ISN'T!!! This may be causing problems in Tanaya Mam's Code??
	# Define the input state
	H_init = np.zeros([BATCH_SIZE, CELL_SIZE*STACK_SIZE])
	# Define training batch
	feed_dict = { X : batch_X , Y_true : batch_Y , H_in : H_init }
	# Train
	sess.run([training_step], feed_dict = feed_dict)
	# Test the performance ocassionally
	if itr%100 == 0 :
		# Define testing batch as the training batch itself
		c_pred, c_true, corr_acc = sess.run([class_true, class_pred, corr_perc], feed_dict = feed_dict) # Never match the names in return values and actual global variables
		# Display
		print '[TESTING] Iteration : ', itr
		print '[TESTING] 	Predicted classes : ', c_pred
		print '[TESTING] 	True classes : ', c_true
		print '[TESTING] 	Accuracy : ', corr_acc


# Indeed the training is now slower in pace!! ( :D )
# Yet, the performance is indeed increasing and crosses the 90% accuracy mark by 800 th iteration on batch size of 128
