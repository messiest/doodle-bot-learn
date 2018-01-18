import os
import time
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt


IMAGE_DIR = "assets/images/"

slim = tf.contrib.slim  # simplified reference


def generated_image(train_step_num, start_time, data_np, save=True):
    """
    visualize generator outputs during training

    check for model output

    :param train_step_num:
    :type train_step_num:
    :param start_time:
    :type start_time:
    :param data_np:
    :type data_np:
    :param save:
    :type save:
    :return:
    :rtype:
    """
    print('Training step: %i' % train_step_num)
    time_since_start = (time.time() - start_time) / 60.0
    print('Time since start: %f m' % time_since_start)
    print('Steps per min: %f' % (train_step_num / time_since_start))
    plt.axis('off')
#     plt.title(f"Step {train_step_num}")

    plt.imshow(np.squeeze(data_np), cmap='gray')

    if save:
        path = IMAGE_DIR + 'generated/'
        plt.savefig(IMAGE_DIR + f'/generated/{len(os.listdir(path))}-gen_step_{train_step_num}.png')


def pre_train_image(tensor_to_visualize, save=True):
    """
    used to visualize generated image pre-training

    check for proper image input

    :param tensor_to_visualize:
    :type tensor_to_visualize:
    :param save:
    :type save:
    :return:
    :rtype:
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with slim.queues.QueueRunners(sess):
            images_np = sess.run(tensor_to_visualize)

    plt.axis('off')
    plt.imshow(np.squeeze(images_np), cmap='gray')

    if save:
        path = IMAGE_DIR + "raw/"
        plt.savefig(IMAGE_DIR + f'raw/result_{len(os.listdir(path))}.png')


def cross_entropy(scores, save=True):  # TODO (@messiest) these aren't working due to the *zip(*object) passed
    """
    plot cross entropy score

    :param scores:
    :type scores:
    :return:
    :rtype:
    """
    plt.plot(scores)
    plt.title('Cross entropy score per step')
    result_num = os.listdir(IMAGE_DIR)

    if save:
        path = IMAGE_DIR + "evaluation/"
        plt.savefig(IMAGE_DIR + f'evaluation/xent_score_{len(os.listdir(path))}.png', dpi=250)


def loss(loss_values, save_image=None):  # TODO (@messiest) see the comment above
    """
    plot the overall loss for the model

    :param loss_values:
    :type loss_values:
    :return:
    :rtype:
    """
    plt.title('Training Loss Per Step')
    plt.plot(loss_values)
    if save_image:
        path = IMAGE_DIR + "evaluation/"
        plt.savefig(IMAGE_DIR + f'evaluation/training_loss_{len(os.listdir(path))}.png', dpi=250)
