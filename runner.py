"""
runner.py
- manage dataset input for the model, which includes data pre-processing
- manage model parameters to set when training the model
- execute model training and testing
- manage model output and visualization
"""
import os
import tensorflow as tf
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables

from model import DCGAN
from mnist_dataset import MNIST
from celeba_dataset import CelebA


class Runner():

    def __init__(self, model_config):
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True

        self.model_config = model_config

        # reset tensorflow graph, does not require kernel restart in spyder
        tf.reset_default_graph()

        self.sess = tf.Session(config=run_config)

        # mnist dataset specific settings
        if model_config.dataset == 'mnist':
            model_config.y_dim = 10
            model_config.dataset = 'mnist'
            data_dir = os.path.join(os.path.abspath(
                model_config.data_dir), 'mnist')
            dataset = MNIST(data_dir, epoch=model_config.epoch,
                            batch_size=64)
        # celebA dataset specific settings
        elif model_config.dataset == 'celebA':
            model_config.crop = True
            data_dir = os.path.join(os.path.abspath(
                model_config.data_dir), 'celebA')
            dataset = CelebA(data_dir, epoch=model_config.epoch,
                             crop=model_config.crop, sess=self.sess)

        # Set the input height for the model to be dependent on the dataset's input_height
        # TODO: This should be in the dataset
        # TODO: model should refer to the dataset for the input_height and input_width to be used all through out
        model_config.input_height = dataset.input_height
        model_config.output_height = dataset.output_height

        # set input data to be a square
        if model_config.input_width is None:
            model_config.input_width = model_config.input_height
        if model_config.output_width is None:
            model_config.output_width = model_config.output_height

        # if model_config.dataset == 'mnist':
        self.model = DCGAN(
            self.sess,
            input_width=model_config.input_width,
            input_height=model_config.input_height,
            output_width=model_config.output_width,
            output_height=model_config.output_height,
            batch_size=model_config.batch_size,
            sample_num=model_config.batch_size,
            y_dim=model_config.y_dim,
            z_dim=model_config.generate_test_images,
            dataset_name=model_config.dataset,  # TODO: make this to refer the dataset instead
            # TODO: make this to refer the dataset
            input_fname_pattern=model_config.input_fname_pattern,
            crop=model_config.crop,  # TODO: Not used in the model, only used in the dataset
            checkpoint_dir=model_config.checkpoint_dir,
            sample_dir=model_config.sample_dir,
            data_dir=model_config.data_dir,
            dataset=dataset
        )

    def start_training(self):

        show_all_variables()
        if self.model_config.train:
            self.model.train(self.model_config)
        else:
            if not self.model.load(self.model_config.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")

          # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
          #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
          #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
          #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
          #                 [dcgan.h4_w, dcgan.h4_b, None])

        OPTION = 1
        visualize(self.sess, self.model, self.model_config, OPTION)

        self.sess.close()
