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

    def __init__(self, data_dir, batch_size=64, grayscale=False):
        # self.y_dim = 10 # TODO: Hardcoded number of classifications
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.fname_extension = "*.jpg"
        self.input_height = 108
        self.output_height = 108
        self.input_width = 108
        self.output_width = 108
        self.crop = False
        self.grayscale = grayscale
        self.data_x, self.data_y, self.c_dim = self.load_data()

    def load_data(self):
        data_x = None
        data_y = None
        c_dim = None
        data_path = os.path.join(self.data_dir, self.fname_extension)
        data_x = glob(data_path)

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

        if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
          c_dim = imread(data_x[0]).shape[-1]
        else:
          c_dim = 1

        if c_dim == 1 and not self.grayscale:
            self.grayscale = True
            data_x = np.array(data_x).astype(np.float32)
        elif not self.grayscale:
            data_x = np.array(sample).astype(np.float32)[:, :, :, None]

        if len(data_x) < self.batch_size:
          raise Exception("[!] Entire dataset size is less than the configured batch_size")

        return data_x, data_y, c_dim