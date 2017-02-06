from src.mode_base import ModelBase

import tensorflow as tf
from src.utils import bn_lrelu_conv, lrelu_conv, bn_lrelu_tconv
from src.utils import tconv, conv, PS


class ModelConv128(ModelBase):

    def __init__(self, batch_size, z_dim, channels=3, y_dim=None, is_training=True):
        super().__init__(batch_size, z_dim, y_dim, is_training)
        self.channels = channels
        self.x_image = tf.placeholder(tf.float32, [batch_size, 128, 128, self.channels], name='x_image')

        self.k = 1

    def encoder(self):
        c_i = self.x_image
        # c_i shape is batch_size x 128 x 128 x self.channels

        c_i = conv(c_i, 4, 2, 32*self.k, name="enc_conv_1")
        # c_i shape is batch_size x 64 x 64 x 32*k

        c_i = bn_lrelu_conv(c_i, 4, 2, 64*self.k, self.bn_settings, name="enc_conv_2")
        # c_i shape is batch_size x 32 x 32 x 64*k

        c_i = bn_lrelu_conv(c_i, 4, 2, 128*self.k, self.bn_settings, name="enc_conv_3")
        # c_i shape is batch_size x 16 x 16 x 128*k

        c_i = bn_lrelu_conv(c_i, 4, 2, 256*self.k, self.bn_settings, name="enc_conv_4")
        # c_i shape is batch_size x 8 x 8 x 256*k

        c_i = bn_lrelu_conv(c_i, 4, 2, 512*self.k, self.bn_settings, name="enc_conv_5")
        # c_i shape is batch_size x 4 x 4 x 512*k

        c_i = bn_lrelu_conv(c_i, 4, 4, self.z_dim, self.bn_settings, name="enc_conv_6")
        # c_i shape is batch_size x 1 x 1 x z_dim

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

            c_i = conv(c_i, filter_size=1, stride=1, out_channels=8192*self.k, name="dec_conv_1")
            c_i = tf.reshape(c_i, shape=[self.batch_size, 4, 4, 512*self.k])
            # Now c_i has shape batch_size x 4 x 4 x 512*k

            c_i = bn_lrelu_tconv(c_i, 4, 2, self.batch_size, 256*self.k, self.bn_settings, name="dec_tconv_2")
            # Now c_i has shape batch_size x 8 x 8 x 256*k

            c_i = bn_lrelu_tconv(c_i, 4, 2, self.batch_size, 128*self.k, self.bn_settings,  name="dec_tconv_3")
            # Now c_i has shape batch_size x 16 x 16 x 128*k

            c_i = bn_lrelu_tconv(c_i, 4, 2, self.batch_size, 64*self.k, self.bn_settings, name="dec_tconv_4")
            # Now c_i has shape batch_size x 32 x 32 x 64*k

            c_i = bn_lrelu_tconv(c_i, 4, 2, self.batch_size, 12, self.bn_settings, name="dec_tconv_5")
            # Now c_i has shape batch_size x 64 x 64 x 12

            c_i = PS(c_i, 2, 3)

            y_image = tf.nn.tanh(c_i)
            return y_image

    def gan(self, x_image, reuse=False, features=False):
        with tf.variable_scope('gan') as scope:
            if reuse:
                scope.reuse_variables()

            c_i = x_image
            # c_i shape is batch_size x 128 x 128 x self.channels

            c_i = conv(c_i, 4, 2, 32*self.k, name="gan_conv_1")
            # c_i shape is batch_size x 64 x 64 x 32*k

            c_i = lrelu_conv(c_i, 4, 2, 64*self.k, self.bn_settings, name="gan_conv_2")
            # c_i shape is batch_size x 32 x 32 x 64*k

            c_i = bn_lrelu_conv(c_i, 4, 2, 128*self.k, self.bn_settings, name="gan_conv_3")
            # c_i shape is batch_size x 16 x 16 x 128*k
            if features:
                return c_i
            c_i = bn_lrelu_conv(c_i, 4, 2, 256*self.k, self.bn_settings, name="gan_conv_4")
            # c_i shape is batch_size x 8 x 8 x 256*k

            c_i = bn_lrelu_conv(c_i, 4, 2, 512*self.k, self.bn_settings, name="gan_conv_5")
            # c_i shape is batch_size x 4 x 4 x 512*k

            c_i = bn_lrelu_conv(c_i, 4, 4, 1, self.bn_settings, name="gan_conv_6")
            # c_i shape is batch_size x 1 x 1 x 1

            z = tf.reshape(c_i, shape=[self.batch_size, 1])

            return z
