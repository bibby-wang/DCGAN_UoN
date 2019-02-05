import os
import numpy as np
import tensorflow as tf
import abc


class TFDataset(abc.ABC):

    def __init__(self, data_dir,
                       batch_size,
                       sample_num,
                       epoch,
                       input_height,
                       input_width,
                       output_height,
                       output_width,
                       ):

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.epoch = epoch
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        # to be used as placeholders by the implementing datasets
        self.data_x_ph = None
        self.label_y_ph = None

        self.data_x, self.data_y, self.c_dim = self.load_data()
        self.tf_dataset, self.tf_sample = self.create_tf_dataset

        # set the number of category labels
        if self.data_y is None:
            self.y_dim = None
        else:
            self.y_dim = len(self.data_y[0])

    # TODO: data encapsulation
    #  @property
    #  def label_dim(self):
    #      return self.__label_dim

    #  @label_dim.setter
    #  def label_dim(self, label_dim):
    #      self.__label_dim = label_dim

    #  @property
    #  def data_dir(self):
    #      return self.__data_dir

    @abc.abstractmethod
    def load_data(self):
        raise NotImplementedError("load data is not implemented yet")

    @abc.abstractmethod
    def create_tf_dataset(self, scope=None):
        raise NotImplementedError("load data is not implemented yet")

    def get_sample_dataset(self, sess):
        it = self.tf_sample.make_one_shot_iterator()
        next_elem = it.get_next()
        data = sess.run(next_elem)
        return data

    def get_batch_dataset(self, sess):

        tf_dataset = self.tf_dataset.shuffle(
            100, reshuffle_each_iteration=True)
        tf_dataset = tf_dataset.batch(self.batch_size, drop_remainder=True)
        tf_dataset = tf_dataset.repeat(self.epoch)
        it = tf_dataset.make_initializable_iterator()
        if self.y_dim is not None:
            data_dict = {
                self.data_x_ph: self.data_x,
                self.label_y_ph: self.data_y
            }
        else:
            data_dict = {
                self.data_x_ph: self.data_x,
            }
        sess.run(it.initializer, feed_dict=data_dict)

        return it.get_next()