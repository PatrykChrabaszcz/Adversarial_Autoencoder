from src.mode_base import ModelBase

import tensorflow as tf
from src.utils import relu_bn_conv, relu_bn_tconv
from src.utils import conv, tconv


class ModelConvMnist(ModelBase):
    def __init__(self, batch_size, z_dim, y_dim=None, is_training=True):
        super().__init__(batch_size, z_dim, y_dim, is_training)
        self.input_dim = 784
        self.x_image = tf.placeholder(tf.float32, [batch_size, self.input_dim], name='x_image')

    def encoder(self):
        c_i = tf.reshape(self.x_image, [-1, 28, 28, 1])
        # c_i shape is batch_size x 28 x 28 x 1

        c_i = conv(c_i, filter_size=3, stride=1, out_channels=16, name="enc_conv_1")
        # c_i shape is batch_size x 28 x 28 x 16

        c_i = relu_bn_conv(c_i, 4, 2, 32, self.bn_settings, name="enc_conv_2")
        # c_i shape is batch_size x 14 x 14 x 32

        c_i = relu_bn_conv(c_i, 4, 2, 64, self.bn_settings, name="enc_conv_3")
        # c_i shape is batch_size x 7 x 7 x 64

        c_i = relu_bn_conv(c_i, 7, 7, self.z_dim, self.bn_settings, name="enc_conv_4")
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
            # c_i shape is batch_size x 1 x 1 x z_dim(+y_dim)

            c_i = tconv(c_i, filter_size=7, stride=7, batch_size=self.batch_size,
                        out_channels=64, name="dec_tconv_1")
            # c_i shape is batch_size x 7 x 7 x 64

            c_i = relu_bn_tconv(c_i, 4, 2, self.batch_size, 32,  self.bn_settings, name="dec_tconv_2")
            # c_i shape is batch_size x 14 x 14 x 32

            c_i = relu_bn_tconv(c_i, 4, 2, self.batch_size, 16, self.bn_settings, name="dec_tconv_3")
            # c_i shape is batch_size x 28 x 28 x 16

            c_i = relu_bn_tconv(c_i, 3, 1, self.batch_size, 1, self.bn_settings, name="dec_tconv_4")
            # c_i shape is batch_size x 28 x 28 x 1

            c_i = tf.nn.sigmoid(c_i)
            y_image = tf.reshape(c_i, shape=[self.batch_size, self.input_dim])
            return y_image
