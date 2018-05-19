# Dependencies
import tensorflow as tf
import numpy as np
import cv2
import os, sys, re
#import matplotlib.pyplot as plt


# These model a data distributioin over large number of images
# Z is a low dimensional input, which is passed through a generative model to get a generated high dimensional output
# Distribution should be viewed as the realm of possibilities
# We have generated distributions vs true distributions-- We define a loss function that makes the two distributions indistinguishable


# Auto Encoder (AE) is an identity model!! (Boring)
# AE initializes the weights randomly -> Full forward pass via bottleneck layer -> Construct loss via MSE (Mean Squared Errors) -> Calculate grads and backprop -> Iterate
# There are some problems with AE's--
# 1. They overfit unless there is large amount of data
# 2. Vanishing gradients problem
# NOVEL SOLUTION-- Variational Auto Encoders!! (Exciting)
# Add a variational component inside the data in order to regularize/make robust/improve training
# Input x-> what is probability q_\phi(z | x) ?? -> what is the probability p_\theta(\hat{x} | z) -> \hat{x}
# Thus, we get the \hat{x} as estimated reconstruction of x
# NNs are considered universal function approximators!! (Never Forget)


# Extract MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot = True)


# Parameters
NUM_PIX = 28*28
LATENT_DIM = 20 # Bottleneck latent dimensions (hit and trial)
HIDDEN_DIM = 500 # Hidden layer neuron numbers
ITR = 100000
RECORD_ITR = 1000


# # with tf.device('/gpu:0'):
# # Define placeholders and variables
# X = tf.placeholder(tf.float32, shape = (None, NUM_PIX))
# # Define weights, we call the function each time we need a NEW weight
# def weight_variable(shape, name):
# 	initializer_this = tf.truncated_normal(shape, stddev = 0.1)
# 	return tf.Variable(initializer_this, name)
# # Define biases, we call the function each time we need a NEW bias
# def bias_variable(shape, name):
# 	initializer_this = tf.truncated_normal(shape, stddev = 0.1)
# 	return tf.Variable(initializer_this, name)
# # Define fully connected layers, we call the function with inputs to perform FC layer
# def fully_connected_layer_function(X, W, B):
# 	return tf.add(tf.matmul(X, W), B)


# Define placeholders and variables
X = tf.placeholder(tf.float32, shape = (None, NUM_PIX))
W_enc = tf.Variable(tf.truncated_normal([NUM_PIX, HIDDEN_DIM], stddev = 0.1))
B_enc = tf.Variable(tf.truncated_normal([HIDDEN_DIM], stddev = 0.1))
W_mu = tf.Variable(tf.truncated_normal([HIDDEN_DIM, LATENT_DIM], stddev = 0.1))
B_mu = tf.Variable(tf.truncated_normal([LATENT_DIM], stddev = 0.1))
W_logstd = tf.Variable(tf.truncated_normal([HIDDEN_DIM, LATENT_DIM], stddev = 0.1))
B_logstd = tf.Variable(tf.truncated_normal([LATENT_DIM], stddev = 0.1))
W_dec = tf.Variable(tf.truncated_normal([LATENT_DIM, HIDDEN_DIM], stddev = 0.1))
B_dec = tf.Variable(tf.truncated_normal([HIDDEN_DIM], stddev = 0.1))
W_rec = tf.Variable(tf.truncated_normal([HIDDEN_DIM, NUM_PIX], stddev = 0.1))
B_rec = tf.Variable(tf.truncated_normal([NUM_PIX], stddev = 0.1))


# Define forward pass
H_enc = tf.nn.tanh(tf.add(tf.matmul(X, W_enc), B_enc))
mu = tf.add(tf.matmul(H_enc, W_mu), B_mu)
logstd = tf.add(tf.matmul(H_enc, W_logstd), B_logstd)
# print '[INFO] Encoder output : ', H_enc.shape # (?, 500)
# print '[INFO] Mu output : ', mu.shape # (?, 20)
# print '[INFO] Logstd output : ', logstd.shape # (?, 20)
noise = tf.random_normal([1, LATENT_DIM])
# print '[INFO] Noise output : ', noise.shape
Z = mu + tf.multiply(noise, tf.exp(.5*logstd))
# print '[INFO] Latent Vector : ', Z.shape
H_dec = ttf.nn.tanh(tf.add(tf.matmul(Z, W_dec), B_dec))
H_rec = tf.nn.softmax(tf.add(tf.matmul(H_dec, W_rec), B_rec))
# print '[INFO] Decoder output : ', H_dec.shape
# print '[INFO] Reconstruction output : ', H_rec.shape
#sys.exit()


# # Define ENCODER
# # Define layer1
# W_enc = weight_variable([NUM_PIX, HIDDEN_DIM], 'W_enc')
# B_enc = bias_variable([HIDDEN_DIM], 'B_enc')
# # tanh : maps to [-1 to 1]. When to use?? Simply put, get data around 0 (??)
# H_enc = tf.nn.tanh(fully_connected_layer_function(X, W_enc, B_enc))
# print H_enc.shape
# # Define BOTTLENECK
# W_mu = weight_variable([HIDDEN_DIM, LATENT_DIM], 'W_mu')
# B_mu = bias_variable([LATENT_DIM], 'B_mu')
# mu = fully_connected_layer_function(H_enc, W_mu, B_mu) # NO TANH
# print mu.shape
# W_logstd = weight_variable([HIDDEN_DIM, LATENT_DIM], 'W_logstd')
# B_logstd = bias_variable([LATENT_DIM], 'B_logstd')
# logstd = fully_connected_layer_function(H_enc, W_logstd, B_logstd) # NO TANH
# print logstd.shape
# # Define noise to be added : Shift and scale it!!
# noise = tf.random_normal([1, LATENT_DIM])
# print noise.shape
# Z = mu + tf.multiply(noise, tf.exp(0.5*logstd))
# print Z.shape
# # While backprop, we do not care about noise!! We update mu and logstd
# # Define DECODER
# W_dec = weight_variable([LATENT_DIM, HIDDEN_DIM], 'W_dec')
# B_dec = bias_variable([HIDDEN_DIM], 'B_dec')
# H_dec = tf.nn.tanh(fully_connected_layer_function(Z, W_dec, B_dec))
# # Define RECONSTRUCTOR
# W_rec = weight_variable([HIDDEN_DIM, NUM_PIX], 'W_rec')
# B_rec = bias_variable([NUM_PIX], 'B_rec')
# H_rec = tf.nn.softmax(fully_connected_layer_function(H_dec, W_rec, B_rec)) # output values in 0 and 1


# Define LOSS. This is the trickiest part of the whole code!!
# L = E_{Z ~ q(Z | X)}\log{p(X | Z)} - KL(q(Z | X) || p(Z))
# This is the variational lower bound L
# E represents how well X is generated given Z
# \log{p(X | Z)} = \sum_{i = 1}^N X(i)\log{Z(i)} + (1 - X(i))\log{(1 - Z(i))}
# KL(q(Z | X) || p(Z)) = -1/2*\sum_{j = 1}^J(1 + 2\log\sigma(j) - \mu(j)^2 - \sigma(j)^2)
log_likelihood_term = tf.reduce_sum(X*tf.log(H_rec + 1e-8) + (1 - X)*tf.log(1 - H_rec + 1e-8), axis = 1)
# print '[INFO] Log-likelihood output : ', log_likelihood_term.shape # [INFO] Log-likelihood output :  (?,)
KL_divergence_term = -0.5*tf.reduce_sum(1 + 2*logstd - tf.pow(mu, 2) - tf.exp(2*logstd), axis = 1)
# print '[INFO] KL-Divergence output : ', KL_divergence_term.shape # [INFO] KL-Divergence output :  (?,)
variational_lower_bound = tf.reduce_mean(log_likelihood_term - KL_divergence_term)
training_step = tf.train.AdadeltaOptimizer().minimize(-variational_lower_bound) # Note the -1 sign!!!!
#sys.exit()


# Initialize!!
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()


# Define for keeping track
variational_lower_bound_array = []
log_likelihood_term_array = []
KL_divergence_term_array = []
iteration_array = [i*RECORD_ITR for i in range(ITR/RECORD_ITR)] # Little trick!!


# Prepare for testing
num_pairs = 5
image_indices = np.random.randint(0, 200, num_pairs)


# Training loop
for itr in range(ITR):
	# np.round makes the mnist data binary
	# Beginner's guide for np.round
	# >>> np.round(-0.9)
	# -1.0
	# >>> np.round(-0.1)
	# -0.0
	# >>> np.round(0.8)
	# 1.0
	# >>> np.round(0.4)
	# 0.0
	# >>> np.round(0.5)
	# 0.0
	# >>> np.round(1.5)
	# 2.0
	# >>> np.round(-0.5)
	# -0.0
	# >>> np.round(-1.5)
	# -2.0
	# Pick an image randomly
	X_batch = np.round(mnist.train.next_batch(200)[0])
	# Update weight
	sess.run(training_step, feed_dict = { X : X_batch })
	# Occasionally, get data
	if itr%500 == 0 :
		# Get output values
		vlb, llt, kldt = sess.run([variational_lower_bound, log_likelihood_term, KL_divergence_term], feed_dict = { X : X_batch })
		# Append to arrays
		variational_lower_bound_array.append(vlb)
		log_likelihood_term_array.append(llt)
		KL_divergence_term_array.append(kldt)
		# Display
		# print '[INFO] VLB : ', vlb.shape
		# print '[INFO] LLT : ', llt.shape
		# print '[INFO] KLDT : ', kldt.shape
		print '[TRAINING] VLB : ', vlb
		#print '[TRAINING] LLT : ', llt
		#print '[TRAINING] KLDT : ', kldt
	# Test occasionally
	# test_info = []
	# if (itr) % 10000 == 0 :
	# 	test_info.append([[], []])
	# 	for pair in range(num_pairs) :
	# 		x = np.reshape(mnist.test.images[image_indices[pair]], (1, NUM_PIX))
	# 		x_image = np.reshape(x, (28,28))
	# 		test_info[-1][0].append(x_image) # Store train info
	# 		x_rec = sess.run([H_rec], feed_dict = { X : x })
	# 		x_rec = np.reshape(x_rec, (28, 28))
	# 		test_info[-1][1].append(x_rec) # Store recon info
	# 		print '[TESTING] Iteration : ', itr, ' DONE!!'
	# 		#while 1 :
	# 		#	plt.imshow(x_image)
	# 		#	plt.imshow(x_rec)
	# 		#	if cv2.waitKey(1) and 0xFF == ord('q'): # At key q, ...
	# 		#		break
	# 		#cv2.destroyAllWindows()
# Save the np array
# test_info = np.array(test_info)
# np.save('VAE_data.npy', test_info)


# Eeshan--iIwYcTb
