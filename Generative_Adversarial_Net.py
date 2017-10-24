# Dependencies
import tensorflow as tf
import numpy as np
import cv2


# Extract MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot = True)


# Parameters : D for Discriminator, G for Generator, K for Kernel
D_K1 = 32
D_K2 = 64
D_K3 = 1024
D_K4 = 1 # Predict 0/1 : fake/real
BATCH_SIZE = 128
Z_DIM = 200
G_K1 = 3136
ITR = 5000


# We will code Deep Convolutional Generative Adversarial Network
# Generator generates images and discriminator differentiates between real (dataset) images and fake (generated) images


# DCGAN is better in comparison with Vanilla GANs in two aspects--
# 1. They use batch normalization :
# It does not increase accuracy, but helps in convergence
# It is important because different features might have different value ranges and it is important to make them equivalent in some sense
# Inputs are a batch X = { x_1, x_2, ..., x_m }
# \mu = \frac{1}{m}\sum_{i = 1}^mx_i
# \sigma^2 = \frac{1}{m}\sum_{i = 1}^m(x_i - \mu)^2
# \hat{x_i} = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
# y_i = \gamma\hat{x_i} + \beta, where \gamma and \beta are trainable weights
# 2. They use ReLU in place of sigmoid
# ReLU outperforms sigmoid layers


# GANs are very hard to train
# Sometimes, Generator fools Discriminator by finding one value that beats the later. This needs to be mitigated.


# The training of DCGANs is a mini-max game
# Generator gets as input Z, which is random noise
# Generator upsamples that noise in order to generate a sample, which tries to mimic the dataset samples
# Discriminator takes as input images and then downsample to perform a task that makes it differentiate between fake and real


# Define placeholders and variables
X = tf.placeholder(tf.float32, [None, 28*28]) # Placeholder for input images
X_im = tf.reshape(X, [-1, 28, 28, 1]) # Reshape the array as 1 channel RGB image of shape 28 x 28
#X_im1 = tf.reshape(X, [-1, 28, 28, 1]) # Reshape the array as 1 channel RGB image of shape 28 x 28
# Define discriminator variables
D_W1 = tf.Variable(tf.truncated_normal([5, 5, 1, D_K1], stddev = 0.1), name = 'D_W1')
D_B1 = tf.Variable(tf.ones([D_K1])/10, name = 'D_B1')
D_W2 = tf.Variable(tf.truncated_normal([5, 5, D_K1, D_K2], stddev = 0.1), name = 'D_W2')
D_B2 = tf.Variable(tf.ones([D_K2])/10, name = 'D_B2')
D_W3 = tf.Variable(tf.truncated_normal([7*7*D_K2, D_K3], stddev = 0.1), name = 'D_W3')
D_B3 = tf.Variable(tf.ones([D_K3])/10, name = 'D_B3')
D_W4 = tf.Variable(tf.truncated_normal([D_K3, D_K4], stddev = 0.1), name = 'D_W4')
D_B4 = tf.Variable(tf.ones([D_K4])/10, name = 'D_B4')
# Define generator variables
G_W1 = tf.Variable(tf.truncated_normal([Z_DIM, G_K1], stddev = 0.1), name = 'G_W1')
G_B1 = tf.Variable(tf.ones([G_K1])/10, name = 'G_B1')
G_W2 = tf.Variable(tf.truncated_normal([3, 3, 1, Z_DIM/2], stddev = 0.1), name = 'G_W2')
G_B2 = tf.Variable(tf.ones([Z_DIM/2])/10, name = 'G_B2')
G_W3 = tf.Variable(tf.truncated_normal([3, 3, Z_DIM/2, Z_DIM/4], stddev = 0.1), name = 'G_W3')
G_B3 = tf.Variable(tf.ones([Z_DIM/4])/10, name = 'G_B3')
G_W4 = tf.Variable(tf.truncated_normal([1, 1, Z_DIM/4, 1], stddev = 0.1), name = 'G_W4')
G_B4 = tf.Variable(tf.ones([1])/10, name = 'G_B4')


# Define discriminator function
def DiscriminatorForward(x):
	D_Y1 = tf.nn.relu(tf.add(tf.nn.conv2d(x, D_W1, strides = [1, 1, 1, 1], padding = 'SAME'), D_B1))
	D_Y2 = tf.nn.avg_pool(D_Y1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
	D_Y3 = tf.nn.relu(tf.add(tf.nn.conv2d(D_Y2, D_W2, strides = [1, 1, 1, 1], padding = 'SAME'), D_B2))
	D_Y4 = tf.nn.avg_pool(D_Y3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
	D_Y5 = tf.reshape(D_Y4, [-1, 7*7*D_K2])
	D_Y6 = tf.nn.relu(tf.add(tf.matmul(D_Y5, D_W3), D_B3))
	D_Y7 = tf.add(tf.matmul(D_Y6, D_W4), D_B4)
	return D_Y7


# # Trial
# Y = DiscriminatorForward(X_im)
# Y1 = DiscriminatorForward(X_im1)
# print '[INFO] Trainable variables with X_im, X_im1 : ', tf.trainable_variables()
# print '[INFO] Trainable variables with X_im ONLY   : ', tf.trainable_variables()
# This shows that we can reuse the weights in several forward passes if the weights are defined properly outside the loop


# Define generator function
def GeneratorForward(BATCH_SIZE_, Z_DIM_, reuse = None):
	Z = tf.truncated_normal([BATCH_SIZE_, Z_DIM_], mean = 0, stddev = 1, name = 'z')
	G_Y1 = tf.add(tf.matmul(Z, G_W1), G_B1)
	G_Y2 = tf.reshape(G_Y1, [-1, 56, 56, 1])
	# Batch-normalize
	G_Y3 = tf.layers.batch_normalization(G_Y2, epsilon = 1e-5, trainable = True, reuse = reuse, name = 'BN1')
	#G_Y3 = tf.layers.batch_normalization(G_Y2, epsilon = 1e-5, reuse = None, trainable = True, name = 'BN1')
	G_Y4 = tf.nn.relu(G_Y3)
	G_Y5 = tf.add(tf.nn.conv2d(G_Y4, G_W2, strides = [1, 2, 2, 1], padding = 'SAME'), G_B2)
	G_Y6 = tf.layers.batch_normalization(G_Y5, epsilon = 1e-5, trainable = True, reuse = reuse, name = 'BN2')
	G_Y7 = tf.nn.relu(G_Y6)
	# G_Y71 = tf.image.resize_images(G_Y7, [56, 56]) # [I DON'T THINK THAT THIS STEP IS REQUIRED, SIRAJ!!]
	# I guess it is indeed needed, because the image shape has gone down by factor of 2 due to striding in convolutions ( :D )
	G_Y71 = tf.image.resize_images(G_Y7, [56, 56])
	G_Y8 = tf.add(tf.nn.conv2d(G_Y71, G_W3, strides = [1, 2, 2, 1], padding = 'SAME'), G_B3)
	G_Y9 = tf.layers.batch_normalization(G_Y8, epsilon = 1e-5, trainable = True, reuse = reuse, name = 'BN3')
	G_Y10 = tf.nn.relu(G_Y9)
	G_Y101 = tf.image.resize_images(G_Y10, [56, 56])
	G_Y11 = tf.add(tf.nn.conv2d(G_Y101, G_W4, strides = [1, 2, 2, 1], padding = 'SAME'), G_B4)
	G_Y12 = tf.sigmoid(G_Y11)
	# Return this final generated image
	return G_Y12 # Note that in this step, the final image size comes from [56, 56] to [28, 28] due to striding of 2. The return shape is BATCH_SIZE x 28 x 28 x 1


# As explained below, there is no need to define the initializer now!!
#init = tf.global_variables_initializer()
#sess = tf.Session()
#sess.run(init)


# We have a huge trouble of reuse of variables that are implicit in definition of a layer
# We want the functional layer to be trainable, reusable and the later demands that scope OR name be given
# Batch norm layer demands input and we give optional epsilon. We also provide name
# 	G_Y3 = tf.layers.batch_normalization(G_Y2, epsilon = 1e-5, name = 'BN1')
# With these only, one instance of Z1 is allowed!!
# Now, we also add trainable = True
# 	G_Y3 = tf.layers.batch_normalization(G_Y2, epsilon = 1e-5, name = 'BN1', trainable = True)
# With these, one instance of Z1 is allowed!!
# Buw, both Z1 and Z2 are not allowed
# For instance, the following code gives error--
# 	Z1 = GeneratorForward(BATCH_SIZE, Z_DIM)
# 	Z2 = GeneratorForward(BATCH_SIZE, Z_DIM)
# ERROR--
# ValueError: Variable BN1/beta already exists, disallowed. Did you mean to set reuse=True in VarScope? Originally defined at:
#	Z3 = GeneratorForward(BATCH_SIZE, Z_DIM)
#
#  File "GAN.py", line 92, in GeneratorForward
#    G_Y3 = tf.layers.batch_normalization(G_Y2, epsilon = 1e-5, name = 'BN1', trainable = True)
#  File "GAN.py", line 110, in <module>
#    Z1 = GeneratorForward(BATCH_SIZE, Z_DIM)
# This shows that reuse = True is a must in such cases. 
# So, we use reuse = True
# But now, if we remove that name = 'BN1', and instantiate Z1 and Z2, we get following--
# Code--
#	Z1 = GeneratorForward(BATCH_SIZE, Z_DIM)
#	Z2 = GeneratorForward(BATCH_SIZE, Z_DIM)
# ERROR--
# 	ValueError: Variable batch_normalization/beta does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=None in VarScope?
# Now we consider only the following line--
# 	Z1 = GeneratorForward(BATCH_SIZE, Z_DIM)
# We get the following error--
# ERROR--
#	ValueError: Variable batch_normalization/beta does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=None in VarScope?
# This shows that without name, we must initialize the variables.
# So, we allow for the component of code to be uncommented
# 	init = tf.global_variables_initializer()
# 	sess = tf.Session()
# 	sess.run(init)
# Now, run the code
# 	Z1 = GeneratorForward(BATCH_SIZE, Z_DIM)
# We still get the same error--
# ValueError: Variable batch_normalization/beta does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=None in VarScope?
# Now, we try to assign name to the functional layer. We do this in order to set scope/define name so that the functional is actually reused
# Thus, we use--
# G_Y3 = tf.layers.batch_normalization(G_Y2, epsilon = 1e-5, trainable = True, reuse = None, name = 'BN1')
# Now, code--
# 	Z1 = GeneratorForward(BATCH_SIZE, Z_DIM) (WORKS GOOD!!)
# Now, code--
# 	Z1 = GeneratorForward(BATCH_SIZE, Z_DIM)
# 	Z2 = GeneratorForward(BATCH_SIZE, Z_DIM)
# We get ERROR--
	# ValueError: Variable BN1/beta already exists, disallowed. Did you mean to set reuse=True in VarScope? Originally defined at:

	#   File "GAN.py", line 92, in GeneratorForward
	#     G_Y3 = tf.layers.batch_normalization(G_Y2, epsilon = 1e-5, trainable = True, reuse = None, name = 'BN1')
	#   File "GAN.py", line 149, in <module>
	#     Z1 = GeneratorForward(BATCH_SIZE, Z_DIM)
# Thus, we set reuse = True and pass the reuse parameter to the function.
# 	Z1 = GeneratorForward(BATCH_SIZE, Z_DIM)
# 	Z2 = GeneratorForward(BATCH_SIZE, Z_DIM, True)
# This gives no error! Hence, we need to create a variable with name and then reuse it appropriately.
# Now, we comment out the initializer, as it does not seem to be needed. Once both Z1 and Z2 are initiated, we again check trainable variables
#	Z1 = GeneratorForward(BATCH_SIZE, Z_DIM)
#	Z2 = GeneratorForward(BATCH_SIZE, Z_DIM, True)
#	print '[INFO] Trainable variables with X_im, X_im1 : ', tf.trainable_variables()
# 	print '[INFO] Trainable variables with X_im ONLY   : ', tf.trainable_variables()
# Carefully note the output!! It is wonderful!!
# Corresponding to the above two print functions, we have following [INFO] messages--
# 	[INFO] Trainable variables with X_im, X_im1 :  [<tf.Variable 'Variable:0' shape=(5, 5, 1, 32) dtype=float32_ref>, <tf.Variable 'Variable_1:0' shape=(32,) dtype=float32_ref>, <tf.Variable 'Variable_2:0' shape=(5, 5, 32, 64) dtype=float32_ref>, <tf.Variable 'Variable_3:0' shape=(64,) dtype=float32_ref>, <tf.Variable 'Variable_4:0' shape=(3136, 1024) dtype=float32_ref>, <tf.Variable 'Variable_5:0' shape=(1024,) dtype=float32_ref>, <tf.Variable 'Variable_6:0' shape=(1024, 1) dtype=float32_ref>, <tf.Variable 'Variable_7:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'Variable_8:0' shape=(200, 3136) dtype=float32_ref>, <tf.Variable 'Variable_9:0' shape=(3136,) dtype=float32_ref>, <tf.Variable 'BN1/beta:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'BN1/gamma:0' shape=(1,) dtype=float32_ref>]
# 	[INFO] Trainable variables with X_im ONLY   :  [<tf.Variable 'Variable:0' shape=(5, 5, 1, 32) dtype=float32_ref>, <tf.Variable 'Variable_1:0' shape=(32,) dtype=float32_ref>, <tf.Variable 'Variable_2:0' shape=(5, 5, 32, 64) dtype=float32_ref>, <tf.Variable 'Variable_3:0' shape=(64,) dtype=float32_ref>, <tf.Variable 'Variable_4:0' shape=(3136, 1024) dtype=float32_ref>, <tf.Variable 'Variable_5:0' shape=(1024,) dtype=float32_ref>, <tf.Variable 'Variable_6:0' shape=(1024, 1) dtype=float32_ref>, <tf.Variable 'Variable_7:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'Variable_8:0' shape=(200, 3136) dtype=float32_ref>, <tf.Variable 'Variable_9:0' shape=(3136,) dtype=float32_ref>, <tf.Variable 'BN1/beta:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'BN1/gamma:0' shape=(1,) dtype=float32_ref>]
# And corresponding to the previous print sentences, we have the following two--
# 	[INFO] Trainable variables with X_im, X_im1 :  [<tf.Variable 'Variable:0' shape=(5, 5, 1, 32) dtype=float32_ref>, <tf.Variable 'Variable_1:0' shape=(32,) dtype=float32_ref>, <tf.Variable 'Variable_2:0' shape=(5, 5, 32, 64) dtype=float32_ref>, <tf.Variable 'Variable_3:0' shape=(64,) dtype=float32_ref>, <tf.Variable 'Variable_4:0' shape=(3136, 1024) dtype=float32_ref>, <tf.Variable 'Variable_5:0' shape=(1024,) dtype=float32_ref>, <tf.Variable 'Variable_6:0' shape=(1024, 1) dtype=float32_ref>, <tf.Variable 'Variable_7:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'Variable_8:0' shape=(200, 3136) dtype=float32_ref>, <tf.Variable 'Variable_9:0' shape=(3136,) dtype=float32_ref>]
# 	[INFO] Trainable variables with X_im ONLY   :  [<tf.Variable 'Variable:0' shape=(5, 5, 1, 32) dtype=float32_ref>, <tf.Variable 'Variable_1:0' shape=(32,) dtype=float32_ref>, <tf.Variable 'Variable_2:0' shape=(5, 5, 32, 64) dtype=float32_ref>, <tf.Variable 'Variable_3:0' shape=(64,) dtype=float32_ref>, <tf.Variable 'Variable_4:0' shape=(3136, 1024) dtype=float32_ref>, <tf.Variable 'Variable_5:0' shape=(1024,) dtype=float32_ref>, <tf.Variable 'Variable_6:0' shape=(1024, 1) dtype=float32_ref>, <tf.Variable 'Variable_7:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'Variable_8:0' shape=(200, 3136) dtype=float32_ref>, <tf.Variable 'Variable_9:0' shape=(3136,) dtype=float32_ref>]
# There are only two more trainable variables in the game now!! Thus, we have succesfully managed to achieve the reusability without scopes, by just assigning names and using reuse appropriately!!!!
# We further bolster our observation by creating many more forward paths of generator
# Code--
# Z1 = GeneratorForward(BATCH_SIZE, Z_DIM)
# Z2 = GeneratorForward(BATCH_SIZE, Z_DIM, True)
# Z3 = GeneratorForward(BATCH_SIZE, Z_DIM, True)
# Z4 = GeneratorForward(BATCH_SIZE, Z_DIM, True)
# print '[INFO] Trainable variables with X_im, X_im1 : ', tf.trainable_variables()
# print '[INFO] Trainable variables with X_im ONLY   : ', tf.trainable_variables()
# # OUTPUT (NO ERROR)--
# [INFO] Trainable variables with X_im, X_im1 :  [<tf.Variable 'Variable:0' shape=(5, 5, 1, 32) dtype=float32_ref>, <tf.Variable 'Variable_1:0' shape=(32,) dtype=float32_ref>, <tf.Variable 'Variable_2:0' shape=(5, 5, 32, 64) dtype=float32_ref>, <tf.Variable 'Variable_3:0' shape=(64,) dtype=float32_ref>, <tf.Variable 'Variable_4:0' shape=(3136, 1024) dtype=float32_ref>, <tf.Variable 'Variable_5:0' shape=(1024,) dtype=float32_ref>, <tf.Variable 'Variable_6:0' shape=(1024, 1) dtype=float32_ref>, <tf.Variable 'Variable_7:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'Variable_8:0' shape=(200, 3136) dtype=float32_ref>, <tf.Variable 'Variable_9:0' shape=(3136,) dtype=float32_ref>, <tf.Variable 'BN1/beta:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'BN1/gamma:0' shape=(1,) dtype=float32_ref>]
# [INFO] Trainable variables with X_im ONLY   :  [<tf.Variable 'Variable:0' shape=(5, 5, 1, 32) dtype=float32_ref>, <tf.Variable 'Variable_1:0' shape=(32,) dtype=float32_ref>, <tf.Variable 'Variable_2:0' shape=(5, 5, 32, 64) dtype=float32_ref>, <tf.Variable 'Variable_3:0' shape=(64,) dtype=float32_ref>, <tf.Variable 'Variable_4:0' shape=(3136, 1024) dtype=float32_ref>, <tf.Variable 'Variable_5:0' shape=(1024,) dtype=float32_ref>, <tf.Variable 'Variable_6:0' shape=(1024, 1) dtype=float32_ref>, <tf.Variable 'Variable_7:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'Variable_8:0' shape=(200, 3136) dtype=float32_ref>, <tf.Variable 'Variable_9:0' shape=(3136,) dtype=float32_ref>, <tf.Variable 'BN1/beta:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'BN1/gamma:0' shape=(1,) dtype=float32_ref>]
# There are only two trainable variables!! HENCE, DONE!!


# Discriminator loss--
# \nabla_D \frac{1}{m}\sum{i = 1}^m \log{D(x^i)} + \log{1 - D(G(z^i))}, where m is batch size
# The first term stands for optimizing the probability that the real data is ranked high, towards 1
# The minimax game is best represented in terms of \log of probs (think in terms of likelihoods)
# The second term stands for optimizing the probability that the fake data is ranked poorly/low
# Generator loss--
# \nabla_G \frac{1}{m}\sum_{i = 1}^m \log{D(G(z^i))}, where m is the batch size
# This is because G wants that its "production be real good" and hence, wants to increase D(G(z^i)) to 1, which in turn reflects in minimizing log(1 - D(G(z^i)))
# The game slowly converges by training alternatively


# Define session
sess = tf.Session()


# We confirm one thing again!
# Code--
# 	Z = GeneratorForward(BATCH_SIZE, Z_DIM,True)
# OUTPUT--
# 	ValueError: Variable BN1/beta does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=None in VarScope?
# Thus, we need to initialize the variables before we can write down the forward path
# Also, all the required variables are defined inside
# Hence, we define the initializer (well, not quite!!)


# # Good to have this check!!
# print '[INFO] Trainable variables : '
# for item in tf.trainable_variables():
# 	print '[INFO] 	', str(item)


# We define the forward pass
G_of_Z_trial = GeneratorForward(BATCH_SIZE, Z_DIM) # To initialize the net!!
# SANITY CHECK!!
# print '[INFO] Trainable variables : '
# for item in tf.trainable_variables():
# 	print '[INFO] 	', str(item)
# OUTPUT--
# As expected, three pairs of batch_normalization weights are initialized in addition
	# [INFO] 	<tf.Variable 'BN1/beta:0' shape=(1,) dtype=float32_ref>
	# [INFO] 	<tf.Variable 'BN1/gamma:0' shape=(1,) dtype=float32_ref>
	# [INFO] 	<tf.Variable 'BN2/beta:0' shape=(100,) dtype=float32_ref>
	# [INFO] 	<tf.Variable 'BN2/gamma:0' shape=(100,) dtype=float32_ref>
	# [INFO] 	<tf.Variable 'BN3/beta:0' shape=(50,) dtype=float32_ref>
	# [INFO] 	<tf.Variable 'BN3/gamma:0' shape=(50,) dtype=float32_ref>
# Define the actual forward pass entry with reusing the weights
# Note that in each forward pass, new random input vector will appear!!
G_of_Z = GeneratorForward(BATCH_SIZE, Z_DIM, True)
# Define the discriminator forward path on real data
D_of_X = DiscriminatorForward(X_im)
# Define the discriminator forward path on fake data : note that the G_of_Z inside is cleverly chosen!! It allows the weights to be reused automatically!!
D_of_G_of_Z = DiscriminatorForward(G_of_Z)
# SANITY CHECK AGAIN!! NO EXTRA VARIABLES IN TRAINABLE LIST SHOULD APPEAR!!
# Code--
# print '[INFO] Trainable variables : '
# for item in tf.trainable_variables():
# 	print '[INFO] 	', str(item)
# OUTPUT--
# As expected!! Nothing extra comes up!!
# ...
# ...
# [INFO] 	<tf.Variable 'G_B3:0' shape=(50,) dtype=float32_ref>
# [INFO] 	<tf.Variable 'G_W4:0' shape=(1, 1, 50, 1) dtype=float32_ref>
# [INFO] 	<tf.Variable 'G_B4:0' shape=(1,) dtype=float32_ref>
# [INFO] 	<tf.Variable 'BN1/beta:0' shape=(1,) dtype=float32_ref>
# [INFO] 	<tf.Variable 'BN1/gamma:0' shape=(1,) dtype=float32_ref>
# [INFO] 	<tf.Variable 'BN2/beta:0' shape=(100,) dtype=float32_ref>
# [INFO] 	<tf.Variable 'BN2/gamma:0' shape=(100,) dtype=float32_ref>
# [INFO] 	<tf.Variable 'BN3/beta:0' shape=(50,) dtype=float32_ref>
# [INFO] 	<tf.Variable 'BN3/gamma:0' shape=(50,) dtype=float32_ref>
# Also, the forward pass definition is correct!!


# Define the losses!! There are three terms!!
# Generator loss
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_of_G_of_Z, labels = tf.ones_like(D_of_G_of_Z)))
# Discriminator loss on real data
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_of_X, labels = tf.fill([BATCH_SIZE, 1], 0.9)))
# Discriminator loss on fake data
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_of_G_of_Z, labels = tf.zeros_like(D_of_G_of_Z)))
# Net discriminator loss 
D_loss = D_loss_real + D_loss_fake


# Get list of all trainable variables
train_vars = tf.trainable_variables()
# # Print names of all trainables
# print '[INFO] Names of trainable variables : '
# for item in train_vars :
# 	print '[INFO] 	', item.name
# Define important batches of trainables
D_vars = [var for var in train_vars if 'D_' in var.name]
G_vars = [var for var in train_vars if 'G_' in var.name]
B_vars = [var for var in train_vars if 'BN' in var.name]
# print '[INFO] Discriminator trainables : '
# for item in D_vars :
# 	print '[INFO] 	', item.name
# print '[INFO] Generator trainables : '
# for item in G_vars :
# 	print '[INFO] 	', item.name
# print '[INFO] Batch normalization trainables : '
# for item in B_vars :
# 	print '[INFO] 	', item.name
# # OUTPUT-- (SEEMS ALL GOOD!!)
# [INFO] Discriminator trainables : 
# [INFO] 	D_W1:0
# [INFO] 	D_B1:0
# [INFO] 	D_W2:0
# [INFO] 	D_B2:0
# [INFO] 	D_W3:0
# [INFO] 	D_B3:0
# [INFO] 	D_W4:0
# [INFO] 	D_B4:0
# [INFO] Generator trainables : 
# [INFO] 	G_W1:0
# [INFO] 	G_B1:0
# [INFO] 	G_W2:0
# [INFO] 	G_B2:0
# [INFO] 	G_W3:0
# [INFO] 	G_B3:0
# [INFO] 	G_W4:0
# [INFO] 	G_B4:0
# [INFO] Batch normalization trainables : 
# [INFO] 	BN1/beta:0
# [INFO] 	BN1/gamma:0
# [INFO] 	BN2/beta:0
# [INFO] 	BN2/gamma:0
# [INFO] 	BN3/beta:0
# [INFO] 	BN3/gamma:0
# Another check!!-- EVERYTHING IS GOOD TO GO!
# J_vars = D_vars + B_vars
# print '[INFO] Discriminator trainables after appending the batch normalization trainables : '
# for item in J_vars :
# 	print '[INFO] 	', item.name
training_step_D_fake = tf.train.AdamOptimizer(0.0001).minimize(D_loss_fake, var_list = D_vars + B_vars)
training_step_D_real = tf.train.AdamOptimizer(0.0001).minimize(D_loss_real, var_list = D_vars)
training_step_G = tf.train.AdamOptimizer(0.0001).minimize(G_loss, var_list = G_vars + B_vars)


# Define placeholders
D_real_count_placeholder = tf.placeholder(tf.float32)
D_fake_count_placeholder = tf.placeholder(tf.float32)
G_count_placeholder = tf.placeholder(tf.float32)


# Define performance indices
D_of_G_score = tf.reduce_mean(DiscriminatorForward(G_of_Z))
D_of_X_score = tf.reduce_mean(DiscriminatorForward(X_im))
example_generated_images = GeneratorForward(5, Z_DIM, True)


# There are several points of failure for GANs
# 1. D loss approaches to 0, which gives no gradients for generator training
# 2. D loss rises in unbounded fashion on generated images, which does not help D training and stalls the G training
# 3. D accuracy diverges, wherein D either predicts everything as real or as fake


# Define saver
saver = tf.train.Saver()


# Run initializer
init = tf.global_variables_initializer()
sess.run(init)


# Define parameters
gLoss = 0
dLossFake = 1
dLossReal = 1
d_real_count = 0
d_fake_count = 0
g_count = 0
# Training loop :
for i in range(ITR) :
	# Get training batch
	real_images_batch = mnist.train.next_batch(BATCH_SIZE)[0]
	# If the dLossFake is too high--
	if dLossFake > 0.6 :
		_, dLossReal, dLossFake, gLoss = sess.run([training_step_D_fake, D_loss_real, D_loss_real, G_loss], feed_dict = { X : real_images_batch })
		d_fake_count += 1
	# If the gLoss is too high--
	if gLoss > 0.5 :
		_, dLossReal, dLossFake, gLoss = sess.run([training_step_G, D_loss_real, D_loss_fake, G_loss], feed_dict = { X : real_images_batch })
		g_count += 1
	# If the dLossReal is too high--
	if dLossReal > 0.45 :
		_, dLossReal, dLossFake, gLoss = sess.run([training_step_D_real, D_loss_real, D_loss_fake, G_loss], feed_dict = { X : real_images_batch })
		d_real_count += 1
	# Occasionally, print stuff
	if i%100 == 0 :
		print '[TRAINING] Iteration : ', i, 'd_fake_count : ', d_fake_count, ' d_real_count : ', d_real_count, ' g_count : ', g_count
		print '[TRAINING] Discriminator on real images : ', sess.run([D_of_X_score], feed_dict = { X : real_images_batch })
		print '[TRAINING] Discriminator on real images : ', sess.run([D_of_G_score])
	# Rarely, even test!!


# Finally, print images learnt by the generator!
moment_of_truth = sess.run([G_of_Z])
moment_of_truth = np.array(moment_of_truth)
print moment_of_truth.shape
moment_of_truth_ = moment_of_truth[0]
print moment_of_truth_.shape
print moment_of_truth_[0].shape
im1 = moment_of_truth_[0].reshape([28, 28])
im2 = moment_of_truth_[1].reshape([28, 28])
im3 = moment_of_truth_[2].reshape([28, 28])
im4 = moment_of_truth_[3].reshape([28, 28])
im5 = moment_of_truth_[4].reshape([28, 28])
while 1:
	cv2.imshow('im1', im1)
	cv2.imshow('im2', im2)
	cv2.imshow('im3', im3)
	cv2.imshow('im4', im4)
	cv2.imshow('im5', im5)
	if cv2.waitKey(1) and 0xFF == ord('q'):
		break
cv2.destroyAllWindows()