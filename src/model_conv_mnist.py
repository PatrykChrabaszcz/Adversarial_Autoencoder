from src.mode_base import ModelBase

import tensorflow as tf
from src.utils import lrelu_conv, bn_lrelu_conv, bn_lrelu_tconv
from src.utils import conv, PS


class ModelConvMnist(ModelBase):

    def __init__(self, batch_size, z_dim, y_dim=None, is_training=True):
        super().__init__(batch_size, z_dim, y_dim, is_training)
        self.input_dim = 784
        self.x_image = tf.placeholder(tf.float32, [batch_size, self.input_dim], name='x_image')
        self.k = 1
        self.channels = 1

    def encoder(self):
        c_i = self.x_image
        c_i = tf.reshape(c_i, shape=[self.batch_size, 28, 28, 1])
        # Now c_i has shape batch_size x 28 x 28 x 1

        c_i = conv(c_i, filter_size=3, stride=1, out_channels=16, name="enc_conv_1")
        # Now c_i has shape batch_size x 28 x 28 x 16

        c_i = bn_lrelu_conv(c_i, 4, 2, 32*self.k, self.bn_settings, name="enc_conv_2")
        # Now c_i has shape batch_size x 14 x 14 x 32*k

        c_i = bn_lrelu_conv(c_i, 4, 2, 64*self.k, self.bn_settings, name="enc_conv_3")
        # Now c_i has shape batch_size x 7 x 7 x 64*k

        c_i = bn_lrelu_conv(c_i, 7, 7, self.z_dim, self.bn_settings, name="enc_conv_fin")
        # Now c_i has shape batch_size x 1 x 1 x z_dim

        z = tf.reshape(c_i, shape=[self.batch_size, self.z_dim])
        return z

    def decoder(self, z, reuse=False):
        with tf.variable_scope('decoder') as scope:
            if reuse:
                scope.reuse_variables()
            z_dim = self.z_dim
            if self.y_dim:
                z = tf.concat(1, [z, self.y_labels])
                z_dim = self.z_dim + self.y_dim

            c_i = tf.reshape(z, shape=[self.batch_size, 1, 1, z_dim])
            # Now c_i has shape batch_size x 1 x 1 x z_dim(+y_dim)

            c_i = conv(c_i, 1, 1, 7*7*64*self.k, name="dec_conv_1")
            c_i = tf.reshape(c_i, shape=[self.batch_size, 7, 7, 64*self.k])
            # Now c_i has shape batch_size x 7 x 7 x 64*k

            c_i = bn_lrelu_tconv(c_i, 4, 2, self.batch_size, 32 * self.k,
                                 self.bn_settings, name="dec_tconv_2")
            # Now c_i has shape batch_size x 14 x 14 x 32*k

            c_i = bn_lrelu_tconv(c_i, 4, 2, self.batch_size, 1,
                                 self.bn_settings, name="dec_tconv_fin")
            # Now c_i has shape batch_size x 28 x 28 x 1

            c_i = tf.reshape(c_i, shape=[self.batch_size, 784])
            y_image = tf.nn.sigmoid(c_i)
            return y_image

    def gan(self, x_image, reuse=False, features=False):
        with tf.variable_scope('gan') as scope:
            if reuse:
                scope.reuse_variables()

            c_i = x_image
            c_i = tf.reshape(c_i, shape=[self.batch_size, 28, 28, 1])
            # Now c_i has shape batch_size x 28 x 28 x 1

            c_i = bn_lrelu_conv(c_i, 4, 2, 32 * self.k, self.bn_settings, name="gan_conv_1")
            # Now c_i has shape batch_size x 14 x 14 x 32*k
            if features:
                return c_i
            c_i = bn_lrelu_conv(c_i, 4, 2, 64 * self.k, self.bn_settings, name="gan_conv_2")
            # Now c_i has shape batch_size x 7 x 7 x 64*k

            c_i = bn_lrelu_conv(c_i, 7, 7, 1, self.bn_settings, name="gan_conv_fin")
            # Now c_i has shape batch_size x 1 x 1 x z_dim

            y = tf.reshape(c_i, shape=[self.batch_size, 1])
            return y
