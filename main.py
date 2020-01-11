import math
import numpy as np
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from utils import *
import tensorflow as tf
from tensorflow.python.framework import ops

np.random.seed(1)

train_dataset, train_labels, test_dataset, test_labels, classes = load_data()
#display an example
show_example(dataset=train_dataset, labels=train_labels, index= 5)

#to binary image
train_dataset = train_dataset/255.
test_dataset = test_dataset/255.
train_labels = to_one_hot(train_labels, len(classes))
test_labels = to_one_hot(test_labels, len(classes))
print(test_labels.shape)