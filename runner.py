"""
runner.py
- manage dataset input for the model, which includes data pre-processing
- manage model parameters to set when training the model
- execute model training and testing
- manage model output and visualization
"""
import tensorflow as tf
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables

from model import DCGAN

class Runner():

    def __init__(self, model_config, dataset):
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth=True

        # reset tensorflow graph, enables re-running model withour restarting the kernel
        tf.reset_default_graph()

        self.sess = tf.Session(config=run_config)
        # self.model = model
        # self.dataset = dataset
        self.model_config = model_config

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
            dataset_name=model_config.dataset,
            input_fname_pattern=model_config.input_fname_pattern,
            crop=model_config.crop,
            checkpoint_dir=model_config.checkpoint_dir,
            sample_dir=model_config.sample_dir,
            data_dir=model_config.data_dir,
            dataset=dataset)
        # else:
        #   self.model = DCGAN(
        #       self.sess,
        #       input_width=model_config.input_width,
        #       input_height=model_config.input_height,
        #       output_width=model_config.output_width,
        #       output_height=model_config.output_height,
        #       batch_size=model_config.batch_size,
        #       sample_num=model_config.batch_size,
        #       z_dim=model_config.generate_test_images,
        #       dataset_name=model_config.dataset,
        #       input_fname_pattern=model_config.input_fname_pattern,
        #       crop=model_config.crop,
        #       checkpoint_dir=model_config.checkpoint_dir,
        #       sample_dir=model_config.sample_dir,
        #       data_dir=model_config.data_dir)

    def start_training(self):

        show_all_variables()
        if self.model_config.train:
          self.model.train(self.model_config)
        else:
          if not self.model.load(self.model_config.checkpoint_dir)[0]:
            raise Exception("[!] Train a model first, then run test mode")

        OPTION = 1
        visualize(sess, dcgan, model_config, OPTION)

        self.sess.close()
