# Model created to compare z from VAE and AAE ( Between groups for this lab course )


from src.mode_base import ModelBase

import tensorflow as tf
from src.utils import lrelu_conv, lrelu_tconv, bn_lrelu_conv, bn_lrelu_tconv
from src.utils import tconv, conv, bn_conv, lin


class ModelCompareMnist(ModelBase):
    def __init__(self, batch_size, z_dim, y_dim=None, is_training=True):
        super().__init__(batch_size, z_dim, y_dim, is_training)
        self.input_dim = 784
        self.x_image = tf.placeholder(tf.float32, [batch_size, self.input_dim], name='x_image')
        self.channels = 1

    def encoder(self):
        c_i = self.x_image
        c_i = tf.reshape(c_i, shape=[self.batch_size, 28, 28, 1])
        # Now c_i has shape batch_size x 28 x 28 x 1

        c_i = conv(c_i, 5, 1, 16, name="enc_conv_1")
        c_i = tf.maximum(0., c_i)
        # Now c_i has shape batch_size x 28 x 28 x 16

        c_i = conv(c_i, 5, 1, 16, name="enc_conv_2")
        c_i = tf.maximum(0., c_i)
        # Now c_i has shape batch_size x 28 x 28 x 16

        c_i = tf.reshape(c_i, shape=[self.batch_size, 16*784])
        # Now c_i has shape batch_size x 784

        c_i = lin(c_i, 400, name="enc_lin_1")
        c_i = tf.maximum(0., c_i)
        # Now c_i has shape batch_size x 400

        z = lin(c_i, self.z_dim, name="enc_lin_2")

        return z

    def decoder(self, z, reuse=False):
        with tf.variable_scope('decoder') as scope:
            if reuse:
                scope.reuse_variables()
            z_dim = self.z_dim
            if self.y_dim:
                z = tf.concat(1, [z, self.y_labels])
                z_dim = self.z_dim + self.y_dim

            c_i = z
            c_i = lin(c_i, 784, name='dec_lin_1')
            # Now c_i has shape batch_size x 784
            c_i = tf.maximum(0., c_i)
            c_i = tf.reshape(c_i, [self.batch_size, 28, 28, 1])
            # Now c_i has shape batch_size x 28 x 28 x 1

            c_i = conv(c_i, 5, 1, 16, name="dec_conv_1")
            c_i = tf.maximum(0., c_i)
            # Now c_i has shape batch_size x 28 x 28 x 16

            c_i = conv(c_i, 5, 1, 1, name="dec_conv_2")
            # Now c_i has shape batch_size x 28 x 28 x 1

            c_i = tf.reshape(c_i, shape=[self.batch_size, 784])
            # Now c_i has shape batch_size x 784

            y_image = tf.nn.sigmoid(c_i)
            return y_image

    def gan(self, x_image, reuse=False, features=False):
        pass
