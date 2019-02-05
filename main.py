import os
import sys
import argparse
import numpy as np
from runner import Runner

import tensorflow as tf


class ModelConfig():
    def __init__(self, epoch=25,
                 learning_rate=0.0002,
                 beta1=0.5,
                 train_size=np.inf,
                 batch_size=64,
                 input_height=108,
                 input_width=None,
                 output_height=64,
                 output_width=None,
                 dataset="celebA",
                 input_fname_pattern="*.jpg",
                 checkpoint_dir="checkpoint",
                 data_dir="../DCGAN-tensorflow/data",
                 sample_dir="samples",
                 train=True,
                 crop=False,   # TODO: put this in dataset
                 visualize=False,  # Not being used
                 generate_test_images=100,
                 y_dim=None):

        self.epoch = epoch
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.train_size = train_size
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.dataset = dataset
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.sample_dir = sample_dir
        self.train = train
        self.crop = crop
        self.visualize = visualize
        self.generate_test_images = generate_test_images
        self.y_dim = y_dim


def main():

    model_config = ModelConfig()

    # Create directory for saving checkpoints
    if not os.path.exists(model_config.checkpoint_dir):
        os.makedirs(model_config.checkpoint_dir)
    if not os.path.exists(model_config.sample_dir):
        os.makedirs(model_config.sample_dir)

    # Run the training
    # TODO: Runner class not really needed, put all things in this main
    runner = Runner(model_config)
    runner.start_training()


if __name__ == '__main__':
    main()
