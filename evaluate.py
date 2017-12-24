import tensorflow as tf


slim = tf.contrib.slim


def gan_loss(gan_loss, name=None):
    """
    evaluate loss

    :param gan_loss:  GANLoss tuple
    :type gan_loss: GANtuple
    :param name:
    :type name:
    :return:
    :rtype:
    """
    """
    Evaluate GAN losses. Used to check that the graph is correct.

    Args:
        gan_loss: A GANLoss tuple.
        name: Optional. If present, append to debug output.
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with slim.queues.QueueRunners(sess):
            gen_loss_np = sess.run(gan_loss.generator_loss)
            dis_loss_np = sess.run(gan_loss.discriminator_loss)
    if name:
        print('%s generator loss: %f' % (name, gen_loss_np))
        print('%s discriminator loss: %f' % (name, dis_loss_np))
    else:
        print('Generator loss: %f' % gen_loss_np)
        print('Discriminator loss: %f' % dis_loss_np)

    return gen_loss_np, dis_loss_np
