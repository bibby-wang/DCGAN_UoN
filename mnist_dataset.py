"""
mnist_dataset.py
- provide abstraction to the mnist dataset
- abstract image preprocessing
- abstract mini-batch?
"""
import os
import numpy as np
import tensorflow as tf

class MNIST():

    def __init__(self, data_dir=None, batch_size=None, sample_num=64):
        self.y_dim = 10 # TODO: Hardcoded number of classifications
        self.data_dir = data_dir
        self.data_x, self.data_y, self.c_dim = self.load_mnist()
        self.batch_size = batch_size
        self.input_height = 28
        self.output_height = 28
        self.input_width = 28
        self.output_width = 28
        self.crop = False
        self.sample_num = sample_num

        self.tf_dataset, self.tf_sample = self.create_tf_dataset()

    def load_mnist(self):

        # training data
        fd = open(os.path.join(self.data_dir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

        # training labels
        fd = open(os.path.join(self.data_dir,'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)
    
        # test data
        fd = open(os.path.join(self.data_dir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

        # test labels
        fd = open(os.path.join(self.data_dir,'t10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)


        trY = np.asarray(trY)
        teY = np.asarray(teY)

        # Combine Training and Test data
        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0).astype(np.int)

        seed = 547    # TODO: hardcoded seed number?
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)

        for i, label in enumerate(y):
          y_vec[i,y[i]] = 1.0

        return X/255., y_vec, 1 # normalized data_x, data_y, image channel

    def create_tf_dataset(self):
        # Tensorflow Dataset
        x_ph = tf.placeholder(self.data_x.dtype, self.data_x.shape, name="x_ph")
        y_ph = tf.placeholder(self.data_y.dtype, self.data_y.shape, name="y_ph")

        dataset_x = tf.data.Dataset.from_tensor_slices((x_ph, y_ph))
        # dataset_x = tf.data.Dataset.from_tensors((self.data_x, self.data_y))

        sample_dim_x = self.data_x[:self.sample_num]
        sample_dim_y = self.data_y[:self.sample_num]
        sample = tf.data.Dataset.from_tensors((sample_dim_x, sample_dim_y))

        return dataset_x, sample

    def get_sample_dataset(self, sess):
        it = self.tf_sample.make_one_shot_iterator()
        next = it.get_next()
        data = sess.run(next)
        return data
