import os
import sys
import argparse
import numpy as np
from runner import Runner
from mnist_dataset import MNIST

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
                     train=False,
                     crop=False,
                     visualize=False,
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
  parser = argparse.ArgumentParser(description="Runs DCGAN on either mnist or celebA dataset")
  parser.add_argument('--dataset', dest='dataset', choices=["mnist", "celebA"], type=str, default='celebA')
  parser.add_argument('--input_height=', dest="input_height", default=108, type=int)
  parser.add_argument('--output_height=', dest="output_height", default=108, type=int)
  parser.add_argument('--train', dest="train", action='store_true')
  parser.add_argument('--crop', dest="crop", action='store_true')
  args = parser.parse_args()

  model_config = ModelConfig()
  model_config.dataset = args.dataset
  model_config.input_height = args.input_height
  model_config.output_height = args.output_height
  model_config.train = args.train
  model_config.crop = args.crop

  if model_config.input_width is None:
    model_config.input_width = model_config.input_height

  if model_config.output_width is None:
    model_config.output_width = model_config.output_height

  if model_config.dataset == 'mnist':
    model_config.y_dim = 10

  if not os.path.exists(model_config.checkpoint_dir):
    os.makedirs(model_config.checkpoint_dir)
  if not os.path.exists(model_config.sample_dir):
    os.makedirs(model_config.sample_dir)

  # #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  # run_config = tf.ConfigProto()
  # run_config.gpu_options.allow_growth=True

  # # reset tensorflow graph, enables re-running model withour restarting the kernel
  # tf.reset_default_graph()

  dataset = MNIST("./data/mnist")
  runner = Runner(model_config, dataset)
  runner.start_training()

  # with tf.Session(config=run_config) as sess:
  #   if model_config.dataset == 'mnist':
  #     dcgan = DCGAN(
  #         sess,
  #         input_width=model_config.input_width,
  #         input_height=model_config.input_height,
  #         output_width=model_config.output_width,
  #         output_height=model_config.output_height,
  #         batch_size=model_config.batch_size,
  #         sample_num=model_config.batch_size,
  #         y_dim=10,
  #         z_dim=model_config.generate_test_images,
  #         dataset_name=model_config.dataset,
  #         input_fname_pattern=model_config.input_fname_pattern,
  #         crop=model_config.crop,
  #         checkpoint_dir=model_config.checkpoint_dir,
  #         sample_dir=model_config.sample_dir,
  #         data_dir=model_config.data_dir)
  #   else:
  #     dcgan = DCGAN(
  #         sess,
  #         input_width=model_config.input_width,
  #         input_height=model_config.input_height,
  #         output_width=model_config.output_width,
  #         output_height=model_config.output_height,
  #         batch_size=model_config.batch_size,
  #         sample_num=model_config.batch_size,
  #         z_dim=model_config.generate_test_images,
  #         dataset_name=model_config.dataset,
  #         input_fname_pattern=model_config.input_fname_pattern,
  #         crop=model_config.crop,
  #         checkpoint_dir=model_config.checkpoint_dir,
  #         sample_dir=model_config.sample_dir,
  #         data_dir=model_config.data_dir)

    # show_all_variables()

    # if model_config.train:
    #   dcgan.train(model_config)
    # else:
    #   if not dcgan.load(model_config.checkpoint_dir)[0]:
    #     raise Exception("[!] Train a model first, then run test mode")
      

    # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
    #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
    #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
    #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
    #                 [dcgan.h4_w, dcgan.h4_b, None])

    # Below is codes for visualization
    # OPTION = 1
    # visualize(sess, dcgan, model_config, OPTION)

if __name__ == '__main__':
  main()
