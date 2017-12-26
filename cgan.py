import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import visualizer
import evaluate
import tf_nets


# TFGAN MNIST examples from `tensorflow/models`.
from mnist import data_provider
from mnist import util

# TF-Slim data provider.
from slim.datasets import download_and_convert_mnist

import tensorflow as tf


DATA_DIR = "assets/data/"
RESULT_DIR = "assets/results/"


# short cuts
tfgan = tf.contrib.gan
slim = tf.contrib.slim

if not tf.gfile.Exists(DATA_DIR):  # check if data directory already exists
    tf.gfile.MakeDirs(DATA_DIR)

download_and_convert_mnist.run(DATA_DIR)  # download data if missing

# input pipeline
batch_size = 32
with tf.device('/cpu:0'):  # pin it to the cpu to free up gpu for propagation
    images, one_hot_labels, _ = data_provider.provide_data('train', batch_size, DATA_DIR)

# Sanity check that we're getting images.
imgs_to_visualize = tfgan.eval.image_reshaper(images[:20,...], num_cols=10)
visualizer.pre_train_image(imgs_to_visualize, save=False)

noise_dims = 64
conditional_gan_model = tfgan.gan_model(generator_fn=tf_nets.generator,
                                        discriminator_fn=tf_nets.discriminator,
                                        real_data=images,
                                        generator_inputs=(tf.random_normal([batch_size, noise_dims]), one_hot_labels))

# Sanity check that currently generated images are garbage.
cond_generated_data_to_visualize = tfgan.eval.image_reshaper(conditional_gan_model.generated_data[:20,...], num_cols=10)
visualizer.pre_train_image(cond_generated_data_to_visualize, save=True)

loss = tfgan.gan_loss(conditional_gan_model, gradient_penalty_weight=1.0)

evaluate.gan_loss(loss)  # test loss function

generator_optimizer = tf.train.AdamOptimizer(0.0009, beta1=0.5)  # instantiate optimizers
discriminator_optimizer = tf.train.AdamOptimizer(0.00009, beta1=0.5)

gan_train_ops = tfgan.gan_train_ops(conditional_gan_model,
                                    loss,
                                    generator_optimizer,
                                    discriminator_optimizer)

# Set up class-conditional visualization. We feed class labels to the generator
# so that the the first column is `0`, the second column is `1`, etc.
images_to_eval = 1000
assert images_to_eval % 10 == 0  # ensure multiples of 10

random_noise = tf.random_normal([images_to_eval, 64])
one_hot_labels = tf.one_hot([i for _ in range(images_to_eval // 10) for i in range(10)], depth=10)

with tf.variable_scope(conditional_gan_model.generator_scope, reuse=True):
    eval_images = conditional_gan_model.generator_fn((random_noise, one_hot_labels))

reshaped_eval_imgs = tfgan.eval.image_reshaper(eval_images[:20, ...], num_cols=10)

# We will use a pretrained classifier to measure the progress of our generator.
# Specifically, the cross-entropy loss between the generated image and the target label will be the metric we track.
MNIST_CLASSIFIER_FROZEN_GRAPH = 'mnist/data/classify_mnist_graph_def.pb'
xent_score = util.mnist_cross_entropy(eval_images, one_hot_labels, MNIST_CLASSIFIER_FROZEN_GRAPH)

global_step = tf.train.get_or_create_global_step()
train_step_fn = tfgan.get_sequential_train_steps()

loss_values, xent_score_values = [], []

#
# start session
#
with tf.Session() as sess:
    saver = tf.train.Saver()  # TODO (@messiest) this needs to try and load an existing model

    sess.run(tf.global_variables_initializer())  # run!

    with slim.queues.QueueRunners(sess):
        start_time = time.time()  # start timer
        for i in range(3001):  # number of steps
            cur_loss, _ = train_step_fn(sess, gan_train_ops, global_step, train_step_kwargs={})
            loss_values.append((i, cur_loss))

            if not i % 10:
                xent_score_values.append((i, sess.run(xent_score)))

            if not i % 500:
                print(f'Current loss: {cur_loss:.2f}')
                print(f'Current cross entropy score: {xent_score_values[-1][1]:.2f}')
                visualizer.generated_image(i, start_time, sess.run(reshaped_eval_imgs), save=True)


        #  program complete
        save_path = saver.save(sess, "assets/saved_models/model.ckpt")
        print(f"Model saved in file: {save_path}")

        k = zip(*xent_score_values)
        print(type(k), k)

        plt.plot(*zip(*xent_score_values))
        plt.title('Cross entropy score per step')
        result_num = os.listdir(RESULT_DIR)
        plt.savefig(RESULT_DIR + f'xent_score_{len(result_num)}.png', dpi=250)

        plt.title('Training Loss Per Step')
        plt.plot(*zip(*loss_values))
        plt.savefig(RESULT_DIR + f'training_loss_{len(result_num)}.png', dpi=250)
