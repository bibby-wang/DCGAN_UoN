"""
celeba_dataset.py
- provide abstraction to the mnist dataset
- abstract image preprocessing
- abstract mini-batch?
"""
import os
import numpy as np
import tensorflow as tf

from utils import *
from glob import glob

class CelebA():

    def __init__(self, data_dir, batch_size=64, grayscale=False, sample_num=64, crop=False):
        # self.y_dim = 10 # TODO: Hardcoded number of classifications
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.fname_extension = "*.jpg"
        self.input_height = 108
        self.output_height = 64
        self.input_width = 108
        self.output_width = 64
        self.crop = crop
        self.grayscale = grayscale
        self.sample_num = sample_num
        self.data_x, self.data_y, self.c_dim = self.load_data()

        self.tf_dataset, self.tf_sample = self.create_tf_dataset()

    def load_data(self):
        data_x = None
        data_y = None
        c_dim = None

        data_path = os.path.join(self.data_dir, self.fname_extension)
        data_x = glob(data_path)  # Get all filenames of images in data_path

        # image preprocessing
        # data_x = [get_image(image,
        #               input_height=self.input_height,
        #               input_width=self.input_width,
        #               resize_height=self.output_height,
        #               resize_width=self.output_width,
        #               crop=self.crop,
        #               grayscale=self.grayscale) for image in data_x]

        if len(data_x) == 0:
            raise Exception("[!] No data found in '" + data_path + "'")

        np.random.shuffle(data_x)

        imreadImg = imread(data_x[0])

        # if self.grayscale:
        #   sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
        # else:
        #   sample_inputs = np.array(sample).astype(np.float32)

        # check if image is a non-grayscale image by checking channel number
        if len(imreadImg.shape) >= 3: 
            c_dim = imread(data_x[0]).shape[-1]
        else:
            c_dim = 1

        # if c_dim == 1 and not self.grayscale:
        #     self.grayscale = True
        #     data_x = np.array(data_x).astype(np.float32)
        # elif not self.grayscale:
        #     data_x = np.array(sample).astype(np.float32)[:, :, :, None]

        if len(data_x) < self.batch_size:
            raise Exception("[!] Entire dataset size is less than the configured batch_size")

        return data_x, data_y, c_dim

    def create_tf_dataset(self, scope=None):
        """
        Create the tf.data.Dataset for the training and samples datasets
        """
        with tf.variable_scope(scope or "datasets"):
            # x_ph = tf.placeholder(tf.string, np.array(self.data_x).shape, name="x_ph")
            dataset_x = tf.data.Dataset.from_tensor_slices(self.data_x)
            dataset_x = dataset_x.map(self._read_transform)
            # dataset_x = dataset_x.map(lambda x: get_image(
            #   x,
            #   input_height=self.input_height,
            #   input_width=self.input_width,
            #   resize_height=self.output_height,
            #   resize_width=self.output_width,
            #   crop=self.crop,
            #   grayscale=self.grayscale))

            sample_x = tf.constant(self.data_x[:self.sample_num])
            sample = tf.data.Dataset.from_tensor_slices(sample_x)
            sample = sample.map(self._read_transform)
            # sample = sample.map(lambda x: get_image(
            #   x,
            #   input_height=self.input_height,
            #   input_width=self.input_width,
            #   resize_height=self.output_height,
            #   resize_width=self.output_width,
            #   crop=self.crop,
            #   grayscale=self.grayscale))

        return dataset_x, sample

    def get_sample_dataset(self, sess, scope=None):
        with tf.variable_scope(scope or "datasets"):
            it = self.tf_sample.make_one_shot_iterator()
            next_elem = it.get_next()
            data = np.array([sess.run(next_elem) for _ in range(self.sample_num)])
        return data

    def get_batch_dataset(self, sess, epoch, scope=None):
        with tf.variable_scope(scope or "datasets"):
            # x_ph = tf.placeholder(tf.string, self.data_x, name="x_ph")
            # y_ph = tf.placeholder(self.data_y.dtype, self.data_y.shape, name="y_ph")
            # tf_dataset = tf.data.Dataset.from_tensor_slices(x_ph)
            tf_dataset = self.tf_dataset.batch(self.batch_size, drop_remainder=True)
            tf_dataset = tf_dataset.repeat(epoch)
            # it = tf_dataset.make_initializable_iterator()
            it = tf_dataset.make_one_shot_iterator()
            # sess.run(it.initializer, feed_dict={
            #     x_ph: self.data_x,
            #     # y_ph: self.data_y
            # })
        return it.get_next()
    
    def _read_transform(self, filename,scope=None):
        with tf.variable_scope(scope or "datasets"):
            img_string = tf.read_file(filename)
            img_decoded = tf.image.decode_jpeg(img_string)
            if self.crop:
                # img = tf.image.central_crop(img_decoded, 0.5)
                img_decoded = tf.image.crop_to_bounding_box(
                    image=img_decoded,
                    offset_height=55,
                    offset_width=35,
                    target_height=108,
                    target_width=108
                )
            img = tf.image.resize_images(img_decoded, [self.output_height, self.output_width])
        return tf.subtract(tf.divide(img, 127.5), 1)
