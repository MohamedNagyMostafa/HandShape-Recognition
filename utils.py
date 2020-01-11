import h5py
import numpy as np
import matplotlib.pyplot as plt

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
