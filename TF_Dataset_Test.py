# Dependencies
import os
import sys
import numpy as np
import tensorflow as tf
import time
import cv2


"""
tf.data.Dataset--

The dataset consists of elements of the same structure.
Each element has one or more tf.Tensor objects, which are called components.
Each component has tf.DType (type of elements of tensor) and tf.TensorShape (fully/partially defined shape of element)
tf.data.Dataset.output_types and tf.data.Dataset.output_shapes allow to inspect these properties, as inferred from the input data
Datasets can be nested and the constituents of the dataset can be named
"""


print('\n\n####################################################################################################')
print('DATASETS')
print('####################################################################################################')
# Simple datasets--
dataset_d_1 = tf.data.Dataset.from_tensor_slices(np.random.random([4, 10])) # Batch of 4 entries
print('[INFO] The dataset d_1 has elements with types : ' + str(dataset_d_1.output_types))
print('[INFO] The dataset d_1 has elements with shapes : ' + str(dataset_d_1.output_shapes))
# [INFO] The dataset d_1 has elements with types : <dtype: 'float64'>
# [INFO] The dataset d_1 has elements with shapes : (10,)
dataset_d_2 = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3, 4]))
print('[INFO] The dataset d_2 has elements with types : ' + str(dataset_d_2.output_types))
print('[INFO] The dataset d_2 has elements with shapes : ' + str(dataset_d_2.output_shapes))
# [INFO] The dataset d_2 has elements with types : <dtype: 'int64'>
# [INFO] The dataset d_2 has elements with shapes : ()
dataset_d_3 = tf.data.Dataset.from_tensor_slices(tf.truncated_normal([10, 15])) # 10 sized batch with 15 dim features!
print('[INFO] The dataset d_3 has elements with types : ' + str(dataset_d_3.output_types))
print('[INFO] The dataset d_3 has elements with shapes : ' + str(dataset_d_3.output_shapes))
# [INFO] The dataset d_3 has elements with types : <dtype: 'float32'>
# [INFO] The dataset d_3 has elements with shapes : (15,)

# Simulating real-life datasets : 100 entries with features of size 20 and 100 labels. Include the constituents as a tuple
dataset_d_4 = tf.data.Dataset.from_tensor_slices((tf.truncated_normal([100, 20], mean = 0.0, stddev = 0.1), tf.constant(np.array([int(x) for x in range(100)]).astype(np.float32))))
print('[INFO] The dataset d_4 has elements with types : ' + str(dataset_d_4.output_types))
print('[INFO] The dataset d_4 has elements with shapes : ' + str(dataset_d_4.output_shapes))
# [INFO] The dataset d_4 has elements with types : (tf.float32, tf.float32)
# [INFO] The dataset d_4 has elements with shapes : (TensorShape([Dimension(20)]), TensorShape([]))

# Nested datasets
dataset_d_5 = tf.data.Dataset.zip((dataset_d_1, dataset_d_3))
print('[INFO] The dataset d_5 has elements with types : ' + str(dataset_d_5.output_types))
print('[INFO] The dataset d_5 has elements with shapes : ' + str(dataset_d_5.output_shapes))

# Naming the constituents of the dataset. In this case, we need to pass the named constituents as key-valued pairs
dataset_d_6 = tf.data.Dataset.from_tensor_slices({ 'feats_1' : tf.truncated_normal([100, 20], mean = 0.0, stddev = 0.1) , 'labels_1' : tf.constant(np.array([int(x) for x in range(100)]).astype(np.float32)) })
print('[INFO] The dataset d_6 has elements with types : ' + str(dataset_d_6.output_types))
print('[INFO] The dataset d_6 has elements with shapes : ' + str(dataset_d_6.output_shapes))
# [INFO] The dataset d_6 has elements with types : {'feats_1': tf.float32, 'labels_1': tf.float32}
# [INFO] The dataset d_6 has elements with shapes : {'feats_1': TensorShape([Dimension(20)]), 'labels_1': TensorShape([])}
dataset_d_7 = tf.data.Dataset.from_tensor_slices({ 'feats_1' : tf.truncated_normal([1000, 20], mean = 0.0, stddev = 0.1) , 'labels_1' : tf.constant(np.array([int(x) for x in range(1000)]).astype(np.float32)) })
print('[INFO] The dataset d_7 has elements with types : ' + str(dataset_d_7.output_types))
print('[INFO] The dataset d_7 has elements with shapes : ' + str(dataset_d_7.output_shapes))
# [INFO] The dataset d_7 has elements with types : {'feats_1': tf.float32, 'labels_1': tf.float32}
# [INFO] The dataset d_7 has elements with shapes : {'feats_1': TensorShape([Dimension(20)]), 'labels_1': TensorShape([])}
dataset_d_8 = tf.data.Dataset.zip({ 'dataset_constituent_1' : dataset_d_6 , 'dataset_constituent_7' : dataset_d_7 })
print('[INFO] The dataset d_8 has elements with types : ' + str(dataset_d_8.output_types))
print('[INFO] The dataset d_8 has elements with shapes : ' + str(dataset_d_8.output_shapes))
# [INFO] The dataset d_8 has elements with types : {'dataset_constituent_1': {'feats_1': tf.float32, 'labels_1': tf.float32}, 'dataset_constituent_7': {'feats_1': tf.float32, 'labels_1': tf.float32}}
# [INFO] The dataset d_8 has elements with shapes : {'dataset_constituent_1': {'feats_1': TensorShape([Dimension(20)]), 'labels_1': TensorShape([])}, 'dataset_constituent_7': {'feats_1': TensorShape([Dimension(20)]), 'labels_1': TensorShape([])}}


"""
tf.data.Dataset.map--

Datasets of any shape can be transformed using the "map" transformations
These maps input functions that take as input a function-- Either a well-defined stand-alone function or a lambda function
"""


print('\n\n####################################################################################################')
print('DATASETS AND MAPS')
print('####################################################################################################')
# Define complex datasets
dataset_d_1 = tf.data.Dataset.from_tensor_slices({ 'feats_1' : tf.random_uniform([2, 10]) })
# Define maps
map_d_1 = lambda x: x['feats_1']*2 # The map multiplies input's feats_1 by 2
dataset_d_1.map(map_d_1) # As each element of d_1 is a single tf.Tensor, our map must have 1 input

# Define unnamed datasets
dataset_d_1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([2, 10]))
dataset_d_2 = tf.data.Dataset.from_tensor_slices( ( tf.truncated_normal([2, 15]) , tf.constant(np.random.randint(0, 2, [2, 3]).astype(np.float32)) ) ) 
dataset_d_3 = tf.data.Dataset.zip((dataset_d_1 , dataset_d_2))
dataset_d_1.map(lambda x : x*2) # Map a function that inputs each entry and multiplies it by 2
dataset_d_2.map(lambda x, y : (x*2, y*3)) # A lambda function can return multiple entries as a tuple!
# dataset_d_3.map(lambda x, (y, z) : (x*2, (y*3, z*4))) # THIS IS NOT SUPPORTED!
# dataset_d_3.flat_map(lambda x, y : (x*2, y)) # THIS FAILS TOO! Take home : Avoid complicated dataset structures
dataset_d_3.map(lambda x, y : (x*2, y)) # This is fine!!


"""
Iterators over the dataset--

Datsets can be accessed via iterators over the dataset
Iterators are of different types-- one-shot, initializable, reinitializable and feedable
"""

"""
One-shot iterators--

These iterators are the simplest type of iterators. 
They iterate over the entire dataset once and only once.
They can handle input pipelines that are based on queue, except for parametrization.
Thus, to range over all the dataset exactly once, we can use these iterators
"""


print('\n\n####################################################################################################')
print('ONE SHOT ITERATORS')
print('####################################################################################################')
# Create a dataset using range attribute of dataset
dataset_d_1 = tf.data.Dataset.range(4)
# Create a one-shot iterator
itr_one_shot_d_1 = dataset_d_1.make_one_shot_iterator()
# Create an op that gives the next element! Note that in the fashion of tf, we can not get the data right-away. We need to create an op, run in via a session to get the data
op_next_element = itr_one_shot_d_1.get_next()
# Create a session and run the iterator to get all elements
sess = tf.Session()
for i in range(4) :
	op_next_element_ = sess.run(op_next_element)
	print('[INFO] Iteration number : ' + str(i) + ' dataset entry returned by the iterator : ' + str(op_next_element_))
# [INFO] Iteration number : 0 dataset entry returned by the iterator : 0
# [INFO] Iteration number : 1 dataset entry returned by the iterator : 1
# [INFO] Iteration number : 2 dataset entry returned by the iterator : 2
# [INFO] Iteration number : 3 dataset entry returned by the iterator : 3
# We can also try and make it run for more number of times than the number of entries in the dataset! THE ITERATOR GIVES AN ERROR THEN!!
# dataset_d_2 = tf.data.Dataset.from_tensor_slices(np.array([int(x*x) for x in range(5)]))
# itr_one_shot_d_2 = dataset_d_2.make_one_shot_iterator()
# op_next_element_2 = itr_one_shot_d_2.get_next()
# sess = tf.Session()
# for i in range(6) : # There are only 5 entries in the dataset
# 	op_next_element_ = sess.run(op_next_element_2)
# 	print('[INFO] Iteration number : ' + str(i) + ' dataset entry returned by the iterator : ' + str(op_next_element_))
# # [INFO] Iteration number : 0 dataset entry returned by the iterator : 0
# # [INFO] Iteration number : 1 dataset entry returned by the iterator : 1
# # [INFO] Iteration number : 2 dataset entry returned by the iterator : 4
# # [INFO] Iteration number : 3 dataset entry returned by the iterator : 9
# # [INFO] Iteration number : 4 dataset entry returned by the iterator : 16
# # OutOfRangeError (see above for traceback): End of sequence
# # 	 [[Node: IteratorGetNext_1 = IteratorGetNext[output_shapes=[[]], output_types=[DT_INT64], _device="/job:localhost/replica:0/task:0/device:CPU:0"](OneShotIterator_1)]]


"""
Initializable iterators--

Many times, the size of dataset (in general, the structure of dataset) might depend upon the input fed to a placeholder.
Thus, there is a need to have iterators that can incorporate this into their definition.
Initializable iterators are good for this case, wherein, the dataset is created based on the values taken by a placeholder.
"""


print('\n\n####################################################################################################')
print('INITIALIZABLE ITERATORS')
print('####################################################################################################')
# Create a dynamically generated dataset and corresponding initializable iterator
pl_size = tf.placeholder(tf.int64, shape = [])
dataset_d_1 = tf.data.Dataset.range(pl_size)
# # Try to make one-shot iterator on this. THIS GIVES AN ERROR SINCE THE DATASET HAS PLACEHOLDER VALUES INVOLVED
# itr_init_d_1 = dataset_d_1.make_one_shot_iterator()
# ValueError: Cannot capture a placeholder (name:Placeholder, type:Placeholder) by value.
itr_init_d_1 = dataset_d_1.make_initializable_iterator()
op_next_element = itr_init_d_1.get_next()
sess = tf.Session()
# We first need to initialize the dataset with its appropriate value fed in dictioanry
op_dataset_initializer = itr_init_d_1.initializer # Not a method called, but an attribute!
sess.run(op_dataset_initializer, feed_dict = {pl_size : 5}) # Feed in the size and create the dataset
# Now the dataset is initialized, we can have the entries obtained from it!
for i in range(5) :
	op_next_element_ = sess.run(op_next_element)
	print('[INFO] Iteration number : ' + str(i) + ' dataset entry returned by the iterator : ' + str(op_next_element_))
# We can try an run the iterator for more number of iterations than the number of the entries in the dataset. WE GET AN ERROR for excessing the limits
# OutOfRangeError (see above for traceback): End of sequence
# 	 [[Node: IteratorGetNext_1 = IteratorGetNext[output_shapes=[[]], output_types=[DT_INT64], _device="/job:localhost/replica:0/task:0/device:CPU:0"](Iterator)]]
# However, we can re-create the dataset and get values from the iterator
sess.run(op_dataset_initializer, feed_dict = {pl_size : 10}) 
# Now the dataset is initialized, we can have the entries obtained from it!
for i in range(10) :
	op_next_element_ = sess.run(op_next_element)
	print('[INFO] Iteration number : ' + str(i) + ' dataset entry returned by the iterator : ' + str(op_next_element_))
# [INFO] Iteration number : 0 dataset entry returned by the iterator : 0
# [INFO] Iteration number : 1 dataset entry returned by the iterator : 1
# [INFO] Iteration number : 2 dataset entry returned by the iterator : 2
# [INFO] Iteration number : 3 dataset entry returned by the iterator : 3
# [INFO] Iteration number : 4 dataset entry returned by the iterator : 4
# [INFO] Iteration number : 5 dataset entry returned by the iterator : 5
# [INFO] Iteration number : 6 dataset entry returned by the iterator : 6
# [INFO] Iteration number : 7 dataset entry returned by the iterator : 7
# [INFO] Iteration number : 8 dataset entry returned by the iterator : 8
# [INFO] Iteration number : 9 dataset entry returned by the iterator : 9


"""
Reinitializable iterators--

Many times, there are multiple datasets that have the same structure.
In such cases, it is ideal to define the iterator based on the structure of the datasets, rather than the dataset itself!
Reinitializable datasets can achieve this.
We need to create a bunch of datasets that have the same structure.
Then, we need to use any one of the datasets to define a re-initializable iterator with the common structure of all the datasets.
This iterator can be used to get the next element from any of the datasets.
"""


print('\n\n####################################################################################################')
print('REINITIALIZABLE ITERATORS')
print('####################################################################################################')
# Create multiple datasets with the same structure
dataset_d_1 = tf.data.Dataset.from_tensor_slices((np.array([int(x) for x in range(10)]), np.array([int(x) for x in range(10)])))
dataset_d_2 = tf.data.Dataset.from_tensor_slices((np.array([int(x) for x in range(5)]), np.array([int(x) for x in range(5)])))
# Create a map that applies to all entries and perturbs the second component of each entry by a random amount
# lambda_d_1_1 = lambda x, y : (x, y + np.random.randint(0, 2)) # DOESN'T WORK!!
# lambda_d_2_1 = lambda x, y : (x, y + np.random.randint(-1, 2)) # DOESN'T WORK!!
# lambda_d_1_1 = lambda x, y : (x, y + tf.random_uniform([], 0, 2, tf.int64))
# lambda_d_2_1 = lambda x, y : (x, y + tf.random_uniform([], -1, 2, tf.int64)) # EVEN THIS WON'T WORK, AS THE DATASET DEFINITION IS NOT ASSIGNED!!
lambda_d_1_1 = lambda x, y : (x, y + np.random.randint(0, 2)) # DOESN'T WORK!!
lambda_d_2_1 = lambda x, y : (x, y + np.random.randint(-1, 2)) # DOESN'T WORK!!
# Map that function on the dataset
dataset_d_1 = dataset_d_1.map(lambda_d_1_1)
dataset_d_2 = dataset_d_2.map(lambda_d_2_1)
# Create an iterator using the structure of any one of the two datasets-- d_1 or d_2. Here, we would use d_2
itr_reinitializable = tf.data.Iterator.from_structure(dataset_d_2.output_types, dataset_d_2.output_shapes)
# Get the op for next elements
op_next_element = itr_reinitializable.get_next()
# Now, we have created the iterator and the datasets and the next element op. However, we need to map the iterator onto each dataset and then initialize it!
sess = tf.Session()
op_itr_d_1_init = itr_reinitializable.make_initializer(dataset_d_1)
op_itr_d_2_init = itr_reinitializable.make_initializer(dataset_d_2)
# Initialize the operations
print('##################################################')
sess.run(op_itr_d_1_init)
# Get elements from the datasets using the SAME NEXT ELEMENT OP!
for i in range(10) :
	print('[INFO] Next entry from dataset d_1 : ' + str(sess.run(op_next_element)))
# The key to remember is that we first need to initialize the mapped iterator on the appropriate dataset and then we can access its entries. This can be done as many number of times as we wish
print('##################################################')
sess.run(op_itr_d_2_init)
for i in range(5) :
	print('[INFO] Next entry from dataset d_2 : ' + str(sess.run(op_next_element)))
print('##################################################')
# Do it once again!
sess.run(op_itr_d_1_init)
for i in range(10) :
	print('[INFO] Next entry from dataset d_1 : ' + str(sess.run(op_next_element)))
print('##################################################')
sess.run(op_itr_d_2_init)
for i in range(5) :
	print('[INFO] Next entry from dataset d_2 : ' + str(sess.run(op_next_element)))


"""
Feedable iterators--

Many a times, we need to make multiple datasets, each of which can have an iterator of its own.
In these cases, it is a good idea to pass the choice of the iterator via feed_dict so that each sess.run can have possibly different iterator
This has the same capabilities as that of the reinitializable iterator, but we can switch between the iterators as and when we want.
Note that in the previous case, we were forced to initialize the mapped iterator on a dataset and then only we could access the entries
"""

print('\n\n####################################################################################################')
print('FEEDABLE ITERATORS')
print('####################################################################################################')
# Create datasets, maps and map the maps on the datasets!
dataset_chunk_1 = tf.data.Dataset.range(10)
dataset_chunk_2 = tf.data.Dataset.range(5)
dataset_d_1 = tf.data.Dataset.zip((dataset_chunk_1, dataset_chunk_1))
dataset_d_2 = tf.data.Dataset.zip((dataset_chunk_2, dataset_chunk_2))
lambda_d_1_1 = lambda x, y : (x, y + tf.random_uniform([], 0, 2, tf.int64))
lambda_d_2_1 = lambda x, y : (x, y + tf.random_uniform([], -1, 2, tf.int64))
dataset_d_1 = dataset_d_1.map(lambda_d_1_1)
dataset_d_2 = dataset_d_2.map(lambda_d_2_1)
# We first need to create a handle, which is a placeholder of type string
handle = tf.placeholder(tf.string, shape = []) # Shape is a single entry
# Now, we need to create an iterator from the handle. It requires the handle placeholder and the structure of the desired dataset. Feed in the common structure of the datasets
itr_from_handle = tf.data.Iterator.from_string_handle(handle, dataset_d_1.output_types, dataset_d_1.output_shapes)
# After getting the iterator, we need to create the next element op
op_next_element = itr_from_handle.get_next()
# Now, we can create as many iterators as we wish on the datasets. For simplicity, let us create a one-shot iterator per dataset
itr_d_1 = dataset_d_1.make_one_shot_iterator()
itr_d_2 = dataset_d_2.make_one_shot_iterator()
# From these iterators, create handles that can be fed to the handle
sess = tf.Session()
itr_d_1_handle = sess.run(itr_d_1.string_handle())
itr_d_2_handle = sess.run(itr_d_2.string_handle())
# Now, we want to get the next element. We do so using the SAME next element op, but by using the handle to choose which iterator to use!
for i in range(15) :
	if i < 10 :
		print('[INFO] The next entry is from d_1 : ' + str(sess.run(op_next_element, { handle : itr_d_1_handle })))
	else :
		print('[INFO] The next entry is from d_2 : ' + str(sess.run(op_next_element, { handle : itr_d_2_handle })))


"""
Creating infinite dataset--

The repeat is a special map that can be applied to a dataset, so that it becomes infinite!
To demonstrate this, we will keep printing the next entry till 10 seconds!
"""


print('\n\n####################################################################################################')
print('ACCESSING DATASETS INFINITE TIMES')
print('####################################################################################################')
# Create a simple dataset
dataset_d_1 = tf.data.Dataset.from_tensor_slices(np.random.random([10, 5])) # 10 entries of size 5
dataset_d_1 = dataset_d_1.repeat(3) # Repeat the whole dataset 5 times
itr_d_1 = dataset_d_1.make_one_shot_iterator()
op_next_element = itr_d_1.get_next()
for i in range(30) :
	print('[INFO] Next element position : ' + str(i + 1) + '\tNext entry : ' + str(sess.run(op_next_element))) 
# Repeat a dataset indefinitely. Use repeat without input
dataset_d_1 = tf.data.Dataset.from_tensor_slices(np.random.random([10, 5])) # 10 entries of size 5
dataset_d_1 = dataset_d_1.repeat() 
itr_d_1 = dataset_d_1.make_one_shot_iterator()
op_next_element = itr_d_1.get_next()
# Keep printing for 10 seconds!!
time_start = time.time()
is_continue = True
i = 0 
while is_continue :
	i += 1
	print('[INFO] Next element position : ' + str(i + 1) + '\tNext entry : ' + str(sess.run(op_next_element))) 
	time_now = time.time()
	time_till_now = time_now - time_start
	# if time_till_now > 10 :
	if time_till_now > 1 :
		is_continue = False
	print('[INFO] Time till now : ' + str(time_till_now))


"""
Using next_element in the code and adhering to range--

Sometimes, it might be required to use the next entry obtained from the iterator inside a code.
Note that only after a run of session can the iterator go to the next position. 
For instance, if we use the op_next_element in several steps in the tf graph, all the locations will have the same value.
Further, if the "iterator goes outside the dataset", then tf.errors.OutOfRangeError is raised. This can be captured and handled
The next elements are obtained in the same fashion as is the definition in the case of a nested dataset.
"""


print('\n\n####################################################################################################')
print('USING NEXT ELEMENTS')
print('####################################################################################################')
# Create a dataset and get an iterator
ph_size = tf.placeholder(tf.int64, []) # Single scalar placeholder-- []
dataset_d_1 = tf.data.Dataset.range(ph_size)
itr_d_1_initializable = dataset_d_1.make_initializable_iterator()
op_next_element = itr_d_1_initializable.get_next()
data_consumer = tf.add(tf.add(op_next_element, op_next_element*2), tf.multiply(op_next_element, 3)) # entry = (x + 2*x) + 3*x
# Initialize the iterator. The only placeholder is ph_size
init_itr_and_dataset = itr_d_1_initializable.initializer
sess = tf.Session()
sess.run(init_itr_and_dataset, feed_dict = { ph_size : 5 })
# Now, we can access the entries and use them in the code as well
for i in range(10) : # We access 10 entries. For first 5 times, we will get the entry, for the next 5 times, we will consume the error
	try :
		op_next_element_, data_consumer_ = sess.run([op_next_element, data_consumer])
		print('[INFO] Next entry : ' + str(op_next_element_) + '\tData Consumer Value : ' + str(data_consumer_))
	except tf.errors.OutOfRangeError :
		print('[ERROR] The dataset is already exhausted.')
		# # A good practice is to break the loop as soon as we hit an error of OutOfRngeError type
		# break
# Create nested datasets
dataset_d_1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([10, 5])) # 10 entries, each of size 5
dataset_d_2 = tf.data.Dataset.from_tensor_slices((np.array([x for x in range(10)]), tf.random_uniform([10, 5])))
dataset_d_3 = tf.data.Dataset.zip((dataset_d_1, dataset_d_2))
itr_d_3_initializable = dataset_d_3.make_initializable_iterator()
op_next_element = itr_d_3_initializable.get_next()
op_next_element_1, (op_next_element_2, op_next_element_3) = op_next_element # Extract elements
sess = tf.Session()
init_itr_d_3_and_d_3 = itr_d_3_initializable.initializer
sess.run(init_itr_d_3_and_d_3)
for i in range(10) :
	if i < 5 : # Print first 5 entries as is
		print('[INFO] Iteration : ' + str(i) + ' Entry : ' + str(sess.run(op_next_element)))
	# else : # Print the next 5 entries as their components. This is TRICKY!! THIS GIVES ERROR AFTER PRINTING Component 2 of 6th indexed entry. Note that each call of sess.run takes the iterator to the next entry!!
	# 	print('[INFO] Iteration : ' + str(i) + ' Component 1 : ' + str(sess.run(op_next_element_1)))
	# 	print('[INFO] Iteration : ' + str(i) + ' Component 2 : ' + str(sess.run(op_next_element_2)))
	# 	print('[INFO] Iteration : ' + str(i) + ' Component 3 : ' + str(sess.run(op_next_element_3)))
	# TRICK : Note that anything that requires the op_next to be evaluated will make it increment after the session run is completed!!
	else :
		an_entry = sess.run(op_next_element)
		print('[INFO] Iteration : ' + str(i) + '\n')
		print('[INFO]\t\tComponent 1 : ' + str(an_entry[0]))
		print('[INFO]\t\tComponent 2 : ' + str(an_entry[1][0]))
		print('[INFO]\t\tComponent 3 : ' + str(an_entry[1][1]))


"""
tf.contrib.data.make_saveable_from_iterator--

This tensorflow object allows the creation of a saver for an iterator's state.
This essentially creates a SaveableObject for the entire input pipeline
In order to be able to store the state, we need to add the save-able object into the collection of save-able objects, named tf.GraphKeys.SAVEABLE_OBJECTS
"""


print('\n\n####################################################################################################')
print('SAVING ITERATORS')
print('####################################################################################################')
# # Create a dataset and an iterator on it
# path_saver = './'
# ph_size = tf.placeholder(tf.int64, [])
# dataset_d_1 = tf.data.Dataset.range(ph_size)
# itr_d_1_initializable = dataset_d_1.make_initializable_iterator() # NOT AVAILABLE IN MY VERSION OF TF!!
# op_next_element = itr_d_1_initializable.get_next()
# # Create a saver for the iterator
# saver_itr_ = tf.contrib.data.make_saveable_from_iterator(itr_d_1_initializable)
# # Add the saver into the collection of save-ables
# tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saver_itr_)
# # Initialize the saver
# init_itr_d_1 = itr_d_1_initializable.initializer
# sess = tf.Session()
# saver = tf.train.Saver()
# # Initialize the dataset
# sess.run(init_itr_d_1, feed_dict = { ph_size : 10 })
# # Run for 5 iterations
# for i in range(5) :
# 	print('[INFO] Iteration : ' + str(i) + '\tEntry : ' + str(sess.run(op_next_element)))
# # Save everything in a saver
# saver.save(path_saver) # Feed as argument the path of checkpoint
# # In order to restore the session, define from ph_size to init_ditr_d_1, but not including the last! (i.e., till the line tf.add_to_collection ...)
# saver = tf.train.Saver()
# sess = tf.Session()
# saver.restore(sess, path_saver)


"""
Creating simple dataset from numpy arrays--

If the memory allows, i.e. if we can load the entire dataset into memory at the same time, then we can create a simple dataset from np arrays itself!!
For instance, we can create the mnist dataset and load it as a tf dataset
"""


print('\n\n####################################################################################################')
print('DATASETS FROM NUMPY ARRAYS')
print('####################################################################################################')
# Input the standard MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# Get all the training, validation and testing data
images_train = mnist.train.images
labels_train = mnist.train.labels
images_validation = mnist.validation.images
labels_validation = mnist.validation.labels
images_test = mnist.test.images
labels_test = mnist.test.labels
# Create the tf dataset for training, validation and testing
dataset_train = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
dataset_validation = tf.data.Dataset.from_tensor_slices((images_validation, labels_validation))
dataset_test = tf.data.Dataset.from_tensor_slices((images_test, labels_test))
# We create iterators on each of the datasets and check with the regular dataset methods. Also, instead, we can create a common iterator from structure
itr_reinitializable = tf.data.Iterator.from_structure(dataset_test.output_types, dataset_test.output_shapes)
op_next_element = itr_reinitializable.get_next()
init_itr_mapped_on_train = itr_reinitializable.make_initializer(dataset_train)
init_itr_mapped_on_validation = itr_reinitializable.make_initializer(dataset_validation)
init_itr_mapped_on_test = itr_reinitializable.make_initializer(dataset_test)
# We can iterate on all the datasets one-by-one by initializing the initializers for each of the splits separately. However, we want to show that the iterator behaves exactly the same as the dataset
sess = tf.Session()
sess.run(init_itr_mapped_on_train)
for i in range(1) :
	batch_from_itr = sess.run(op_next_element)
	batch_from_dataset = (images_train[i], labels_train[i])
	# print('[INFO] Entry from the iteartor : \n')
	# print(str(batch_from_itr))
	# print('[INFO] Entry from the dataset : \n')
	# print(str(batch_from_dataset))
	batch_from_itr_0 = batch_from_itr[0]
	batch_from_itr_1 = batch_from_itr[1]
	batch_from_dataset_0 = batch_from_dataset[0]
	batch_from_dataset_1 = batch_from_dataset[1]
	if np.all(batch_from_itr_0 == batch_from_dataset_0) and np.all(batch_from_itr_1 == batch_from_dataset_1) :
		print('[INFO] The entries from the training dataset and the iterator match!!')
sess.run(init_itr_mapped_on_validation)
for i in range(1) :
	batch_from_itr = sess.run(op_next_element)
	batch_from_dataset = (images_validation[i], labels_validation[i])
	batch_from_itr_0 = batch_from_itr[0]
	batch_from_itr_1 = batch_from_itr[1]
	batch_from_dataset_0 = batch_from_dataset[0]
	batch_from_dataset_1 = batch_from_dataset[1]
	if np.all(batch_from_itr_0 == batch_from_dataset_0) and np.all(batch_from_itr_1 == batch_from_dataset_1) :
		print('[INFO] The entries from the validation dataset and the iterator match!!')
sess.run(init_itr_mapped_on_test)
for i in range(1) :
	batch_from_itr = sess.run(op_next_element)
	batch_from_dataset = (images_test[i], labels_test[i])
	batch_from_itr_0 = batch_from_itr[0]
	batch_from_itr_1 = batch_from_itr[1]
	batch_from_dataset_0 = batch_from_dataset[0]
	batch_from_dataset_1 = batch_from_dataset[1]
	if np.all(batch_from_itr_0 == batch_from_dataset_0) and np.all(batch_from_itr_1 == batch_from_dataset_1) :
		print('[INFO] The entries from the testing dataset and the iterator match!!')
# Clear the memory!!
del images_train
del images_validation
del images_test
del labels_train
del labels_validation
del labels_test


"""
tf.data.TFRecord--

It is a simple record-oriented and binary data format. It is used in many TF applications
It is a type of dataset that can be created from any types of files, by storing their contents as int64. float or byte streams!
Once created, the dataset can be read from a simple list of filenames. 
Once the file names are mentioned, then we need to define a parser function that can read the values 
Then we can have iteartorson the dataset and extract values
"""


print('\n\n####################################################################################################')
print('TFRECORDS')
print('####################################################################################################')
# If we know the filenames already, create a tfrecord dataset from TF records (we are creating a dataset and NOT A TFRECORD ITSELF!!)
path_files = ['TFRecords/train.tfrecords', 'TFRecords/validation.tfrecords', 'TFRecords/test.tfrecords']
dataset_mnist = tf.data.TFRecordDataset(path_files)
# We can check the dataset being created! For simplicity, let us check just the first member
itr_dataset_mnist = dataset_mnist.make_initializable_iterator()
op_next_element = itr_dataset_mnist.get_next()
init_itr_dataset_mnist = itr_dataset_mnist.initializer
sess = tf.Session()
sess.run(init_itr_dataset_mnist)
for i in range(1) :
	print('[INFO] First entry in the TFRecord format of MNIST : \n')
	x_1 = sess.run(op_next_element)
	print(str(x_1)) # This prints a huge pile of garbage! (Actually, it is the biary format stored data. We need to parse it appropriately to convert it to usable format)
# HOWEVER, IN MANY CASES, WE JUST KNOW THE PATH TO TFRECORDS, AND NOT ALL FILES IN IT!! Thus, we need to have a way to initialize the tfrecord dataset using a tf.string typed placeholder
ph_paths_tfrecords = tf.placeholder(tf.string, shape = [None]) # A placeholder to hold a single dimensional list of strings!!
paths_tfrecords = [str('TFRecords/' + str(x)) for x in os.listdir('TFRecords')]
dataset_from_tfrecords = tf.data.TFRecordDataset(ph_paths_tfrecords)
# dataset_from_tfrecords.map(_parser) # THIS IS NOT INCLUDED AS OF NOW. However, there must be a function to parse the binary data into usable entries
dataset_from_tfrecords.repeat() # Create batches indefinitely!!
dataset_from_tfrecords.batch(32) # Create batch-size of 32
# Now, we can define an iterator on this dataset
itr_dataset_from_tfrecords = dataset_from_tfrecords.make_initializable_iterator()
op_next_element = itr_dataset_from_tfrecords.get_next()
init_itr_dataset_from_tfrecords = itr_dataset_from_tfrecords.initializer
# Now, to create the dataset and to initialize the iterator, we need to know the value of the placeholder, which is in paths_tfrecords
sess = tf.Session()
sess.run(init_itr_dataset_from_tfrecords, feed_dict = { ph_paths_tfrecords : paths_tfrecords })
for i in range(1) :
	print('[INFO] First entry in the TFRecord format of MNIST : \n')
	x_2 = sess.run(op_next_element)
	print(str(x_2))


"""
tf.data.TextLineDataset--

It is a simple dataset created from one or more .txt files.
It provides a way to extract lines from one or more text files and procudes a string-valued element per line of the files.
In a text file, usually not all the lines contain data. Some are comments (lines beginning with special characters some are header or info lines)
To address this, we first need to create a dataset from filenames and then map it with .skip and .filter commands in order to pick only the legit lines
"""


print('\n\n####################################################################################################')
print('DATASET FROM .TXT FILES')
print('####################################################################################################')
# Create a list of filenames
path_txt_files = ['Data_File_1.txt', 'Data_File_2.txt']
# We can create a dataset from a single entry of this file
dataset_d_1 = tf.data.TextLineDataset('Data_File_1.txt') # Feed in the file name
dataset_d_1 = dataset_d_1.skip(1) # Skip the first line as it is the header
dataset_d_1 = dataset_d_1.filter(lambda a_line : tf.not_equal(tf.substr(a_line, 0, 1), '#')) # Filter only those lines (by filter, it is meant that these lines must be included) that do not start with '#'
# Create an iterator and access all the data
itr_d_1_initializable = dataset_d_1.make_initializable_iterator()
op_next_element = itr_d_1_initializable.get_next()
init_itr_d_1 = itr_d_1_initializable.initializer
# Create a session and print all the entries
sess = tf.Session()
sess.run(init_itr_d_1)
# Run an iterator to get all values
for i in range(10) :
	try :
		print('[INFO] Iteration : ' + str(i) + ' Entry : ' + str(sess.run(op_next_element))) 
	except tf.errors.OutOfRangeError :
		print('[ERROR] Dataset is finished!!')
# # Carefully observe all the output entries!!
# [INFO] Iteration : 0 Entry : b'1,1.34352'
# [INFO] Iteration : 1 Entry : b'2,4533.3'
# [INFO] Iteration : 2 Entry : b'3,4532.0'
# [INFO] Iteration : 3 Entry : b'4,0.0'
# [INFO] Iteration : 4 Entry : b'5,-0.0'
# [INFO] Iteration : 5 Entry : b'6,-324.2'
# [INFO] Iteration : 6 Entry : b'8,8'
# [ERROR] Dataset is finished!!
# [ERROR] Dataset is finished!!
# [ERROR] Dataset is finished!!
print('##################################################')
# We can create the same dataset from multiple files
path_txt_files = ['Data_File_1.txt', 'Data_File_2.txt']
dataset_d_1 = tf.data.TextLineDataset(path_txt_files)
dataset_d_1 = dataset_d_1.skip(1)
dataset_d_1 = dataset_d_1.filter(lambda a_line : tf.not_equal(tf.substr(a_line, 0, 1), '#'))
itr_d_1_initializable = dataset_d_1.make_initializable_iterator()
op_next_element = itr_d_1_initializable.get_next()
init_itr_d_1 = itr_d_1_initializable.initializer
sess = tf.Session()
sess.run(init_itr_d_1)
for i in range(20) :
	try :
		print('[INFO] Iteration : ' + str(i) + ' Entry : ' + str(sess.run(op_next_element))) 
	except tf.errors.OutOfRangeError :
		print('[ERROR] Dataset is finished!!')
# # Carefully note the values! b'idx,val' appears in the list as well. THIS IS NOT WHAT IS WANTED!
# [INFO] Iteration : 0 Entry : b'1,1.34352'
# [INFO] Iteration : 1 Entry : b'2,4533.3'
# [INFO] Iteration : 2 Entry : b'3,4532.0'
# [INFO] Iteration : 3 Entry : b'4,0.0'
# [INFO] Iteration : 4 Entry : b'5,-0.0'
# [INFO] Iteration : 5 Entry : b'6,-324.2'
# [INFO] Iteration : 6 Entry : b'8,8'
# [INFO] Iteration : 7 Entry : b'idx,val'
# [INFO] Iteration : 8 Entry : b'1,1.34352'
# [INFO] Iteration : 9 Entry : b'4,0.0'
# [INFO] Iteration : 10 Entry : b'5,-0.0'
# [INFO] Iteration : 11 Entry : b'6,-324.2'
# [INFO] Iteration : 12 Entry : b'7,-323'
# [INFO] Iteration : 13 Entry : b'8,8'
# [ERROR] Dataset is finished!!
# [ERROR] Dataset is finished!!
# [ERROR] Dataset is finished!!
# [ERROR] Dataset is finished!!
# [ERROR] Dataset is finished!!
# [ERROR] Dataset is finished!!
print('##################################################')
# We can create the same dataset from multiple files in the following fashion!
path_txt_files = ['Data_File_1.txt', 'Data_File_2.txt']
dataset_d_1 = tf.data.Dataset.from_tensor_slices(path_txt_files)
dataset_d_1 = dataset_d_1.flat_map(lambda a_file : tf.data.TextLineDataset(a_file).skip(1).filter(lambda a_line : tf.not_equal(tf.substr(a_line, 0, 1), '#')))
itr_d_1_initializable = dataset_d_1.make_initializable_iterator()
op_next_element = itr_d_1_initializable.get_next()
init_itr_d_1 = itr_d_1_initializable.initializer
sess = tf.Session()
sess.run(init_itr_d_1)
for i in range(20) :
	try :
		print('[INFO] Iteration : ' + str(i) + ' Entry : ' + str(sess.run(op_next_element))) 
	except tf.errors.OutOfRangeError :
		print('[ERROR] Dataset is finished!!')
# # THIS GIVES THE CORRECT RESULTS, AS WE NEEDED AND EXPECTED!!
# [INFO] Iteration : 0 Entry : b'1,1.34352'
# [INFO] Iteration : 1 Entry : b'2,4533.3'
# [INFO] Iteration : 2 Entry : b'3,4532.0'
# [INFO] Iteration : 3 Entry : b'4,0.0'
# [INFO] Iteration : 4 Entry : b'5,-0.0'
# [INFO] Iteration : 5 Entry : b'6,-324.2'
# [INFO] Iteration : 6 Entry : b'8,8'
# [INFO] Iteration : 7 Entry : b'1,1.34352'
# [INFO] Iteration : 8 Entry : b'4,0.0'
# [INFO] Iteration : 9 Entry : b'5,-0.0'
# [INFO] Iteration : 10 Entry : b'6,-324.2'
# [INFO] Iteration : 11 Entry : b'7,-323'
# [INFO] Iteration : 12 Entry : b'8,8'
# [ERROR] Dataset is finished!!
# [ERROR] Dataset is finished!!
# [ERROR] Dataset is finished!!
# [ERROR] Dataset is finished!!
# [ERROR] Dataset is finished!!
# [ERROR] Dataset is finished!!
# [ERROR] Dataset is finished!!


"""
tf.data.Dataset.map()--

The .map callable is used in order to manipulate each entry in the dataset
It inputs a function f that applies on each of the entries of the dataset and returns a dataset. It inputs a tf.Tensor object (single element of dataset) and returns a tf.Tensor object that represents a single element of the dataset
The map can involve np operations, usual mathematical operations and tf operations, the last being the recommended
We can have simple examples of map, as seen in previous cases, but we will have more common examples in the next sections
"""


"""
tf.train.Example--

The tfrecords is the recommended format of storing datasets for tensorflow applications
It stores all the dataset entries in the form of examples. Each example is of the form tf.train.Example
If we start a dataset with tfrecords, we need to map a function that can parse single example from the binary tfrecords
This is a special example of map, which parses a tf.train.Example and can possibly process it to spit out the data in the desired usable form
"""


print('\n\n####################################################################################################')
print('TF EXAMPLES')
print('####################################################################################################')
# Define a parser
def _ParseSingleRawExampleFromTFRecords(serialized_data) :
	
	"""
	inputs--

	serialized_data :
		Example in binarized tfrecords 
	"""

	"""
	outputs--

	parsed_feat_1 :
		The feature 1 as parsed from the serialized data. This is a binary feature in our case.
	parsed_label_1 :
		The label 1 as parsed from the serialized data. This is an int in our case
	"""

	feature_dict = { 'feat_1' : tf.FixedLenFeature((), tf.string) , 'label_1' : tf.FixedLenFeature((), tf.int64) }
	parsed_ex = tf.parse_single_example(serialized_data, feature_dict)
	feat_1 = parsed_ex['feat_1']
	label_1 = parsed_ex['label_1']

	return feat_1, label_1

# Create a dataset from tfrecords
dataset_train = tf.data.TFRecordDataset(['TFRecords/train.tfrecords'])
dataset_train = dataset_train.map(_ParseSingleRawExampleFromTFRecords) # Apply function that can parse the binary data feature and the label
# Create an iterator and access all the elements
itr_d_train_initializable = dataset_train.make_initializable_iterator()
op_next_element = itr_d_train_initializable.get_next()
init_itr_d_train = itr_d_train_initializable.initializer
sess = tf.Session()
sess.run(init_itr_d_train)
# Print the first entry
for i in range(1) :
	print('[INFO] The first entry in the dataset : ' + str(sess.run(op_next_element)))
# This prints the binary feature of the first MNIST image and the label-7 (which is correct!!)

# Define a parser that an decode the raw binary image features into an image
def _ParseSingleDecodedExampleFromTFRecords(serialized_data) :
	
	"""
	inputs--

	serialized_data :
		Example in binarized tfrecords 
	"""

	"""
	outputs--

	parsed_feat_1 :
		The feature 1 as parsed from the serialized data. This is a binary feature in our case.
	parsed_label_1 :
		The label 1 as parsed from the serialized data. This is an int in our case
	"""

	feature_dict = { 'feat_1' : tf.FixedLenFeature((), tf.string) , 'label_1' : tf.FixedLenFeature((), tf.int64) }
	parsed_ex = tf.parse_single_example(serialized_data, feature_dict)
	feat_1 = parsed_ex['feat_1']
	feat_1 = tf.decode_raw(feat_1, tf.uint8) # Convert using decode_raw. The first argument is the feature to convert and the second input is the data type to which we need to convert
	label_1 = parsed_ex['label_1']

	return feat_1, label_1

# Create a dataset from tfrecords
dataset_train = tf.data.TFRecordDataset(['TFRecords/train.tfrecords'])
dataset_train = dataset_train.map(_ParseSingleDecodedExampleFromTFRecords) # Apply function that can parse the binary data feature and the label
# Create an iterator and access all the elements
itr_d_train_initializable = dataset_train.make_initializable_iterator()
op_next_element = itr_d_train_initializable.get_next()
init_itr_d_train = itr_d_train_initializable.initializer
sess = tf.Session()
sess.run(init_itr_d_train)
# Print the first entry
for i in range(1) :
	print('[INFO] The first entry in the dataset : ' + str(sess.run(op_next_element)))
	print('[INFO] The first entry in the dataset : ' + str(sess.run(op_next_element)[0]))
	print('[INFO] The first entry in the dataset : ' + str(sess.run(op_next_element)[0].shape))
	print('[INFO] The first entry in the dataset : ' + str(sess.run(op_next_element)[1]))
	print('[INFO] The first entry in the dataset : ' + str(sess.run(op_next_element)[1].shape))
# # This prints the decoded feature. However, the size is not correct, as it is recorded as 2352.
# [INFO] The first entry in the dataset : (array([0, 0, 0, ..., 0, 0, 0], dtype=uint8), 7)
# [INFO] The first entry in the dataset : [0 0 0 ... 0 0 0]
# [INFO] The first entry in the dataset : (2352,)
# [INFO] The first entry in the dataset : 1
# [INFO] The first entry in the dataset : ()
# To get the correct decoded output, we need to KNOW the exact encoding of the inputs. THIS SEEMS TO BE A SHORTCOMING OF THE API! Use the following code to correctly use the API

# Define a parser that an decode the raw binary image features into an image
def _ParseSingleDecodedExampleFromTFRecords(serialized_data) :
	
	"""
	inputs--

	serialized_data :
		Example in binarized tfrecords 
	"""

	"""
	outputs--

	parsed_feat_1 :
		The feature 1 as parsed from the serialized data. This is a binary feature in our case.
	parsed_label_1 :
		The label 1 as parsed from the serialized data. This is an int in our case
	"""

	feature_dict = { 'feat_1' : tf.FixedLenFeature((), tf.string) , 'label_1' : tf.FixedLenFeature((), tf.int64) }
	parsed_ex = tf.parse_single_example(serialized_data, feature_dict)
	feat_1 = parsed_ex['feat_1']
	feat_1 = tf.decode_raw(feat_1, tf.int32) # Another encoding! FIND THE CORRECT ENCODING BY HIT-AND-TRIAL!!!!
	label_1 = parsed_ex['label_1']

	return feat_1, label_1

dataset_train = tf.data.TFRecordDataset(['TFRecords/train.tfrecords'])
dataset_train = dataset_train.map(_ParseSingleDecodedExampleFromTFRecords) 
itr_d_train_initializable = dataset_train.make_initializable_iterator()
op_next_element = itr_d_train_initializable.get_next()
init_itr_d_train = itr_d_train_initializable.initializer
sess = tf.Session()
sess.run(init_itr_d_train)
for i in range(1) :
	print('[INFO] The first entry in the dataset : ' + str(sess.run(op_next_element)))
	print('[INFO] The first entry in the dataset : ' + str(sess.run(op_next_element)[0]))
	print('[INFO] The first entry in the dataset : ' + str(sess.run(op_next_element)[0].shape))
	print('[INFO] The first entry in the dataset : ' + str(sess.run(op_next_element)[1]))
	print('[INFO] The first entry in the dataset : ' + str(sess.run(op_next_element)[1].shape))

# Another way to use this parser is to use ANY kind of files and then define the desired functionality. The following example is hypothetical--
file_names = tf.constant(['Images/im_1.jpg', 'Images/im_2.jpg'])
labels = tf.constant([32, 1])
dataset_images = tf.data.Dataset.from_tensor_slices((file_names, labels))
def _ParseImagesFromFiles(a_file_name, a_label) : # NOTE THAT AN ENTRY OF THE DATASET IS a_file_name AND a_label, which mimics the entries of the dataset as of now

	"""
	inputs--

	a_file_name :
		Example of the name of file
	a_label :
		The corresponding label
	"""

	"""
	outputs--

	parsed_loaded_image :
		Loaded and parsed image
	parsed_label :
		Parsed label, which will be identical to the input label
	"""

	parsed_loaded_image = cv2.imread(a_file_name.decode()) # We need to decode the file name and then read it via cv2
	parsed_label = a_label

	return parsed_loaded_image, parsed_label
# Since this parser function uses functions that are not using tf, we need to map it as a tf.py_func
# We are mapping an in-place lambda function. It must input a_file_name and a_label. They must be CONVERTED to an image and a label. This output is a tuple of the image and the label. The image and the label will be obtained from a regular python function. The call tf.py_func allows to do that, with first argument as the function name, the second being the input list and the third being the output datatypes
dataset_images = dataset_images.map(lambda a_file_name, a_label : tuple(tf.py_func(_ParseImagesFromFiles, [a_file_name, a_label], [tf.uint8, a_label.dtype])))


"""
Creating batches of data--

Almost always, we need batches of data from a dataset.
This is achieved using the .batch() method for the class of datasets
"""


print('\n\n####################################################################################################')
print('BATCHING THE DATASET')
print('####################################################################################################')
# dataset_d_1 = tf.data.Dataset.from_tensor_slices(np.array([x for x in range(10)]).astype(np.float32))
# dataset_d_2 = tf.data.Dataset.from_tensor_slices(tf.random_uniform(shape = [10, 5]))
# dataset_d_3 = tf.data.Dataset.zip((dataset_d_1, dataset_d_2))
# dataset_d_3_batch = dataset_d_3.batch(4)
# itr_d_3_one_shot = dataset_d_3_batch.make_initializable_iterator()
# op_next_element = itr_d_3_one_shot.get_next()
# sess = tf.Session()
# sess.run(itr_d_3_one_shot.initializer)
# for i in range(10) :
# 	print('[INFO] Iteration : ' + str(i) + ' Batch of size 4 : ' + str(sess.run(op_next_element)))
# # # TRICK! The dataset gets exhausted after 2 iterations and half the next!
# # [INFO] Iteration : 0 Batch of size 4 : (array([0., 1., 2., 3.], dtype=float32), array([[0.8732004 , 0.0504725 , 0.93043816, 0.8808429 , 0.3385867 ],
# #        [0.43248057, 0.11033738, 0.83129835, 0.4386369 , 0.64893126],
# #        [0.6333921 , 0.8307407 , 0.95417285, 0.14215422, 0.48749363],
# #        [0.8133067 , 0.68265915, 0.8404523 , 0.32187164, 0.2798376 ]],
# #       dtype=float32))
# # [INFO] Iteration : 1 Batch of size 4 : (array([4., 5., 6., 7.], dtype=float32), array([[0.5500783 , 0.68858874, 0.28348708, 0.4319272 , 0.50177824],
# #        [0.73060036, 0.8014771 , 0.286044  , 0.17267597, 0.09057891],
# #        [0.3873173 , 0.7133751 , 0.64562154, 0.350726  , 0.61373734],
# #        [0.7431874 , 0.46514666, 0.4127797 , 0.33395493, 0.58312273]],
# #       dtype=float32))
# # [INFO] Iteration : 2 Batch of size 4 : (array([8., 9.], dtype=float32), array([[0.04777563, 0.04185534, 0.34166193, 0.6186359 , 0.8059368 ],
# #        [0.39856946, 0.00511253, 0.92232347, 0.4880129 , 0.24416947]],
# # OutOfRangeError (see above for traceback): End of sequence
print('##################################################')
print('THE CORRECT WAY--')
print('##################################################')
dataset_d_1 = tf.data.Dataset.from_tensor_slices(np.array([x for x in range(10)]).astype(np.float32))
dataset_d_2 = tf.data.Dataset.from_tensor_slices(tf.random_uniform(shape = [10, 5]))
dataset_d_3 = tf.data.Dataset.zip((dataset_d_1, dataset_d_2))
dataset_d_3 = dataset_d_3.repeat()
dataset_d_3_batch = dataset_d_3.batch(4)
itr_d_3_one_shot = dataset_d_3_batch.make_initializable_iterator()
op_next_element = itr_d_3_one_shot.get_next()
sess = tf.Session()
sess.run(itr_d_3_one_shot.initializer)
for i in range(10) :
	print('[INFO] Iteration : ' + str(i) + ' Batch of size 4 : ' + str(sess.run(op_next_element)))
print('##################################################')
print('THE INCORRECT WAY--\nThe batching of 10 sized dataset into 4 sized batches creates a 2-sized batch, which is reflected in the values generated from the iterator!! Check the output manually!!')
print('##################################################')
dataset_d_1 = tf.data.Dataset.from_tensor_slices(np.array([x for x in range(10)]).astype(np.float32))
dataset_d_2 = tf.data.Dataset.from_tensor_slices(tf.random_uniform(shape = [10, 5]))
dataset_d_3 = tf.data.Dataset.zip((dataset_d_1, dataset_d_2))
dataset_d_3_batch = dataset_d_3.batch(4)
dataset_d_3_batch = dataset_d_3_batch.repeat() # Repeat indefinitely!! THIS WAS MISSING EARLIER!
itr_d_3_one_shot = dataset_d_3_batch.make_initializable_iterator()
op_next_element = itr_d_3_one_shot.get_next()
sess = tf.Session()
sess.run(itr_d_3_one_shot.initializer)
for i in range(10) :
	print('[INFO] Iteration : ' + str(i) + ' Batch of size 4 : ' + str(sess.run(op_next_element)))


"""
Creating padded batches of data--

Applications oriented about sequence modelling involve variable sized inputs. 
This is achieved using the .padded_batch() method of the Dataset class.
We need to provide as the first argument the batch size, the second argument as the padded_shapes.
Setting it None takes care of the variable size that might exist in the data and pads the content appropriately
"""


print('\n\n####################################################################################################')
print('PADDED BATCHES')
print('####################################################################################################')
# # Create a 2-D dataset with variable shapes
# dataset_d_1 = tf.data.Dataset.range(5)
# def _GetChunkOfIdentityMatrix(x) :

# 	"""
# 	inputs--

# 	x :
# 		Input, entry of the dataset d_1
# 	"""

# 	"""
# 	outputs--

# 	chunk (implicit) :
# 		The matrix Identity_5(0:x, 0:x)
# 	"""

# 	# return np.eye(5)[0:int(x + 1), 0:int(x + 1)].astype(np.float32)
# 	return tf.fill([tf.cast(x, tf.int32), tf.cast(x, tf.int32)], x)

# dataset_d_1 = dataset_d_1.map(lambda x : tf.py_func(_GetChunkOfIdentityMatrix, [x], [tf.int32]))
# dataset_d_1 = dataset_d_1.repeat()
# # dataset_d_1 = dataset_d_1.batch(2) # THIS WILL GIVE ERROR!
# print('[DEBUG] Output shape of the dataset : ' + str(dataset_d_1.output_shapes))
# # dataset_d_1 = dataset_d_1.padded_batch(2, padded_shapes = (None, None), padding_values = 0)

# itr_d_1_initializable = dataset_d_1.make_initializable_iterator()
# op_next_element = itr_d_1_initializable.get_next()
# init_itr_d_1 = itr_d_1_initializable.initializer
# sess = tf.Session()
# sess.run(init_itr_d_1)
# for i in range(3) :
# 	print('[INFO] Iteration : ' + str(i) + ' Batch with padding : ' + str(sess.run(op_next_element)))
# THE WHOLE EXAMPLE IS A PROBLEM. Advice : Create single dimensional lata and work!!
# Create a 1-D dataset with variable shapes
dataset_d_1 = tf.data.Dataset.range(5)
dataset_d_1 = dataset_d_1.map(lambda x : tf.fill([tf.cast(x, tf.int32)], x))
dataset_d_1 = dataset_d_1.repeat()
# dataset_d_1 = dataset_d_1.batch(2) # THIS WILL GIVE ERROR!
print('[DEBUG] Output shape of the dataset : ' + str(dataset_d_1.output_shapes))
dataset_d_1 = dataset_d_1.padded_batch(2, padded_shapes = [None], padding_values = tf.constant(7, dtype = tf.int64))

itr_d_1_initializable = dataset_d_1.make_initializable_iterator()
op_next_element = itr_d_1_initializable.get_next()
init_itr_d_1 = itr_d_1_initializable.initializer
sess = tf.Session()
sess.run(init_itr_d_1)
for i in range(3) :
	print('[INFO] Iteration : ' + str(i) + ' Batch with padding : ' + str(sess.run(op_next_element)))


"""
tf.data.Dataset.shuffle(buffer_size)--

Another crucial aspect for creating batches from any dataset is the random shuffling of the data.
This can be achieved by the .shuffle method of the Dataset class. 
The method requires the argument buffer_size, which is the number of elements of the dataset that are loaded into the buffer. It also requires optional seed value as an integer to create the pseudo random entries and the optional reshuffle_each_iteration, which as the name suggests, reshuffles the dataset per iteration and is defaulted to True
"""


print('\n\n####################################################################################################')
print('PADDED BATCHES')
print('####################################################################################################')
# We just want to see that whether this consumes all the elements of the dataset, even if the buffer size is low
dataset_d_1 = tf.data.Dataset.range(10)
dataset_d_1 = dataset_d_1.shuffle(buffer_size = 2)
dataset_d_1 = dataset_d_1.repeat()
itr_d_1_initializable = dataset_d_1.make_initializable_iterator()
op_next_element = itr_d_1_initializable.get_next()
init_itr_d_1 = itr_d_1_initializable.initializer
sess = tf.Session()
sess.run(init_itr_d_1)
for i in range(20) :
	print('[INFO] Next entry : ' + str(sess.run(op_next_element)))
# # THIS DOES CONSUME ALL THE ENTRIES! It only loads a few of them onto the RAM so as to avoid the high memory consumption
# [INFO] Next entry : 0
# [INFO] Next entry : 2
# [INFO] Next entry : 1
# [INFO] Next entry : 3
# [INFO] Next entry : 4
# [INFO] Next entry : 6
# [INFO] Next entry : 7
# [INFO] Next entry : 8
# [INFO] Next entry : 9
# [INFO] Next entry : 5
# [INFO] Next entry : 1
# [INFO] Next entry : 0
# [INFO] Next entry : 3
# [INFO] Next entry : 2
# [INFO] Next entry : 4
# [INFO] Next entry : 6
# [INFO] Next entry : 5
# [INFO] Next entry : 8
# [INFO] Next entry : 7
# [INFO] Next entry : 9


"""
TO CONCLUCE WITH A DUMMY EXAMPLE--

It is recommended to use the Dataset.make_one_shot_iterator() for creating a new iterator over the entire dataset
It is also recommended to use the tf.train.MonitoredTrainingSession() so as to avoid getting the OutOfRangeError when dataset exhausts
If that error is hit, the session's .should_stop() method returns True. This can be exploited to stop the training
"""


print('\n\n####################################################################################################')
print('FINAL EXAMPLE--')
print('####################################################################################################')
# Create a simple dataset
dataset_1 = tf.data.Dataset.range(1000) 
dataset_1 = dataset_1.map(lambda x : tf.cast(x, tf.float32))
dataset_2 = dataset_1.map(lambda x : ((tf.cast(x, tf.float32)*2) + tf.random_uniform([]))) # 2*x + e
dataset = tf.data.Dataset.zip((dataset_1, dataset_2))
dataset = dataset.shuffle(buffer_size = 10)
dataset = dataset.repeat(10) # 10 epochs!
dataset = dataset.batch(2) # 128 sized batch
itr_dataset = dataset.make_one_shot_iterator()
op_next_element = itr_dataset.get_next()
# This gives us the next element
W = tf.Variable(tf.truncated_normal([]))
print(W)
x = op_next_element[0]
y = op_next_element[1]
loss = tf.reduce_mean((y - tf.multiply(x, W))*(y - tf.multiply(x, W)))
opt = tf.train.AdamOptimizer(1e-10)
op_training_step = opt.minimize(loss)
# Create a session
sess = tf.train.MonitoredTrainingSession() 
index = 0
print('[INFO]')
while not sess.should_stop() :
	try :
		_, loss_, x_, y_ = sess.run([op_training_step, loss, x, y])
		sys.stdout.write('\r[INFO] Iteration : ' + str(index) + ' Loss : ' + str(loss_) + ' x : ' + str(x_) + ' y : ' + str(y_))
		sys.stdout.flush()
		index += 1
		time.sleep(0.5)
	except tf.errors.OutOfRangeError : # UNNCESSARY!!
		print('\n[ERROR] Dataset Exhausted!!')
		break
# Print the value of the variables
W_ = sess.run(W)
print('[INFO] Learned weight W : ' + str(W_)) # NOTE ONLY THE CODE!!