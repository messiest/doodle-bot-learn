import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os


RESULT_DIR = "assets/results/"

slim = tf.contrib.slim


def generated_image(train_step_num, start_time, data_np, save=True):
    """
    Visualize generator outputs during training.

    Args:
        train_step_num: The training step number. A python integer.
        start_time: Time when training started. The output of `time.time()`. A
            python float.
        data: Data to plot. A numpy array, most likely from an evaluated TensorFlow
            tensor.
    """
    print('Training step: %i' % train_step_num)
    time_since_start = (time.time() - start_time) / 60.0
    print('Time since start: %f m' % time_since_start)
    print('Steps per min: %f' % (train_step_num / time_since_start))
    plt.axis('off')

    plt.imshow(np.squeeze(data_np), cmap='gray')

    if save:
        result_num = os.listdir(RESULT_DIR)
        plt.savefig(RESULT_DIR + f'gen_{len(result_num)}.png')


def image(tensor_to_visualize, save=True):
    """
    used to visualize generated image pre-training.

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
        result_num = os.listdir(RESULT_DIR)
        plt.savefig(RESULT_DIR + f'result_{len(result_num)}.png')


def cross_entropy(xent_scores):
    """
    plot cross entropy score

    :param xent_scores:
    :type xent_scores:
    :return:
    :rtype:
    """
    plt.plot(xent_scores)
    plt.title('Cross entropy score per step')
    result_num = os.listdir(RESULT_DIR)
    plt.savefig(RESULT_DIR + f'xent_score_{len(result_num)}.png', dpi=250)


def loss(loss_values):
    plt.title('Training Loss Per Step')
    plt.plot(loss_values)
    result_num = os.listdir(RESULT_DIR)
    plt.savefig(RESULT_DIR + f'training_loss_{len(result_num)}.png', dpi=250)
