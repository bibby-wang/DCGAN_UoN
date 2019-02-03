import os
import sys
import argparse
import numpy as np
from runner import Runner
from mnist_dataset import MNIST
from celeba_dataset import CelebA

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
                       data_dir="data",
                       sample_dir="samples",
                       train=True,
                       crop=False,   # TODO: put this in dataset
                       visualize=False, # Not being used
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

    # process input arguments
    # --dataset [mnist, celebA]
    # --input_height=[size]
    # --output_height=[size]
    # --train
    # --crop

    # create a parser of the command line
    # parser = argparse.ArgumentParser(description="Runs DCGAN on either mnist or celebA dataset")
    # parser.add_argument('--dataset', dest='dataset', choices=["mnist", "celebA"], type=str, default='celebA')
    # parser.add_argument('--train', dest="train", action='store_true')
    # parser.add_argument('--crop', dest="crop", action='store_true')
    # args = parser.parse_args()


    model_config = ModelConfig()
    # model_config.dataset = args.dataset
    # model_config.train = args.train

    if model_config.dataset == 'mnist':
        model_config.y_dim = 10
        model_config.dataset = 'mnist'
        data_dir = os.path.join(os.path.abspath(model_config.data_dir), 'mnist')
        dataset = MNIST(data_dir,
                        batch_size=64)
    elif model_config.dataset == 'celebA':
        model_config.crop = True
        data_dir = os.path.join(os.path.abspath(model_config.data_dir), 'celebA')
        dataset = CelebA(data_dir,
                         crop=model_config.crop)

    model_config.input_height = dataset.input_height
    model_config.output_height = dataset.output_height
    model_config.crop = dataset.crop

    if not os.path.exists(model_config.checkpoint_dir):
        os.makedirs(model_config.checkpoint_dir)
    if not os.path.exists(model_config.sample_dir):
        os.makedirs(model_config.sample_dir)

    if model_config.input_width is None:
        model_config.input_width = model_config.input_height

    if model_config.output_width is None:
        model_config.output_width = model_config.output_height

    runner = Runner(model_config, dataset)
    runner.start_training()


if __name__ == '__main__':
    main()
