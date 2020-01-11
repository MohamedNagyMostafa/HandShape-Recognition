import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

def load_data():
	train_dataset = h5py.File('data/train_signs.h5', 'r')
	train_set = np.array(train_dataset['train_set_x'][:])
	train_labels = np.array(train_dataset['train_set_y'][:])

	test_dataset = h5py.File('data/test_signs.h5', 'r')
	test_set = np.array(test_dataset['test_set_x'][:])
	test_labels = np.array(test_dataset['test_set_y'][:])

	classes = np.array(test_dataset['list_classes'][:])

	train_labels= train_labels.reshape((1, train_labels.shape[0]))
	test_labels = test_labels.reshape((1,test_labels.shape[0]))

	return train_set, train_labels, test_set, test_labels, classes 


def show_example(index, dataset, labels):
	image = dataset[index]
	label = labels[0,index]
	print(label)

	plt.imshow(image)
	plt.show()

def to_one_hot(labels, classes):
	return np.eye(classes)[labels[0,:]]

def define_placeholders(*args):
	placeholders=[]
	for arg in args:
		placeholders.append(tf.placeholder(tf.float32, shape=(None, *arg.shape[1:])))
	return placeholders[:]

def initialize_parameters():
	# The same randome seed - match previous random
	tf.set_random_seed(1)
	W1= tf.get_variable("W1", [4,4,3,8], initializer= tf.contrib.layers.xavier_initializer(seed=0))
	W2= tf.get_variable("W2", [2,2,8,16], initializer= tf.contrib.layers.xavier_initializer(seed=0))

	parameters = {'W1':W1,'W2':W2}

	return parameters

def forward_propagation(X, parameters):
	W1, W2 = parameters['W1'], parameters['W2']

	Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1], padding='SAME')
	A1 = tf.nn.relu(Z1)
	P1 = tf.nn.max_pool(A1, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME')
	Z2 = tf.nn.conv2d(P1,W2, strides=[1,1,1,1], padding='SAME')
	A2 = tf.nn.relu(Z2)
	P2 = tf.nn.max_pool(A2, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')
	F = tf.contrib.layers.flatten(P2)
	Z3 = tf.contrib.layers.fully_connected(F, 6, activation_fn=None)

	return Z3

def compute_cost(Z3, Y):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

def to_batch(X, Y, batch_size):
	m = np.arange(0,X.shape[0],1, dtype=np.int32)
	num_m = len(m)
	np.random.shuffle(m)
	r = int(num_m/batch_size * (num_m//batch_size))
	
	for end in range(batch_size,m.shape[0],batch_size + r):
		start = end - batch_size

		yield np.array(X[m[start:end]]), np.array(Y[m[start:end]])


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
	num_epochs= 300, minibatch_size= 64, print_cost=True):
	
	ops.reset_default_graph()
	tf.set_random_seed(1)
	seed = 3

	X, Y = define_placeholders(X_train, Y_train)
	costs=[]

	parameters = initialize_parameters()

	Z3 = forward_propagation(X, parameters)

	cost = compute_cost(Z3, Y)
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epochs):
			minibatch_cost = 0
			for X_batch, y_batch in to_batch(X_train, Y_train, batch_size=minibatch_size):
				_, temp = sess.run([optimizer, cost], feed_dict={X: X_batch, Y: y_batch})

				minibatch_cost += temp/minibatch_size
			if print_cost ==True and epoch % 5 == 0:
				print('Cost {} at epoch {} '.format(minibatch_cost, epoch))
			if print_cost == True and epoch % 1 == 0:
				costs.append(minibatch_cost)
		
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()
		predict = tf.argmax(Z3, 1)
		correct_prediction = tf.equal(predict, tf.argmax(Y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print(accuracy)
		train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
		test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
		print("Train Accuracy:", train_accuracy)
		print("Test Accuracy:", test_accuracy)
		return train_accuracy, test_accuracy, parameters
