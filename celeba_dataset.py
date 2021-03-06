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
from tf_dataset import TFDataset


class CelebA(TFDataset):

    def __init__(self,
                 data_dir,
                 epoch,
                 sess,
                 batch_size=64,
                 grayscale=False,
                 sample_num=64,
                 crop=True):

        self.crop = crop
        self.grayscale = grayscale
        self.fname_extension = "*.jpg"
        self.sess = sess

        super(CelebA, self).__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            sample_num=sample_num,
            epoch=epoch,
            input_height=108,
            input_width=108,
            output_height=64,
            output_width=64
        )

    def load_data(self):
        data_x = None
        data_y = None
        c_dim = None

        data_path = os.path.join(self.data_dir, self.fname_extension)

        # Get all filenames of images in data_path
        data_x = np.array(glob(data_path))

        if len(data_x) == 0:
            raise Exception("[!] No data found in '" + data_path + "'")

        imreadImg = imread(data_x[0])

        # check if image is a non-grayscale image by checking channel number
        if len(imreadImg.shape) >= 3:
            c_dim = imread(data_x[0]).shape[-1]
        else:
            c_dim = 1

        if len(data_x) < self.batch_size:
            raise Exception(
                "[!] Entire dataset size is less than the configured batch_size")

        return data_x, data_y, c_dim

    @property
    def create_tf_dataset(self, scope=None):
        """
        Create the tf.data.Dataset for the training and samples datasets
        """

        self.data_x_ph = tf.placeholder(
            self.data_x.dtype, self.data_x.shape, name="data_x_ph")
        dataset_x = tf.data.Dataset.from_tensor_slices((self.data_x_ph,))
        dataset_x = dataset_x.map(self.__read_transform)

        sample_x = self.data_x[:self.sample_num]
        sample = tf.data.Dataset.from_tensor_slices((sample_x,))
        sample = sample.map(self.__read_transform)
        sample_it = sample.make_one_shot_iterator()
        sample_next = sample_it.get_next()

        sample_images = np.array([self.sess.run(sample_next)
                                  for _ in range(self.sample_num)]).astype(np.float)

        sample_images = tf.data.Dataset.from_tensors(sample_images)

        return dataset_x, sample_images

    def __read_transform(self, filename, scope=None):
        with tf.variable_scope(scope or "datasets"):
            img_string = tf.read_file(filename)
            img_decoded = tf.image.decode_jpeg(img_string)
            if self.crop:
                img_decoded = tf.image.crop_to_bounding_box(
                    image=img_decoded,
                    offset_height=55,
                    offset_width=35,
                    target_height=108,
                    target_width=108
                )
            img = tf.image.resize_images(
                img_decoded, [self.output_height, self.output_width])
        # compansate for difference in decoding jpeg images
        return tf.subtract(tf.divide(img, 125.38), 1.)
