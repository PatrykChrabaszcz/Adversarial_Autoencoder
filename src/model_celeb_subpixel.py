from src.mode_base import ModelBase

import tensorflow as tf
from src.utils import conv, relu_bn_conv, PS


class ModelSubpixelCeleb(ModelBase):

    def __init__(self, batch_size, z_dim, y_dim=None, is_training=True):
        super().__init__(batch_size, z_dim, y_dim, is_training)
        self.x_image = tf.placeholder(tf.float32, [batch_size, 32, 32, 3], name='x_image')

    def encoder(self):
        c_i = self.x_image
        # c_i shape is batch_size x 32 x 32 x 3

        c_i = conv(c_i, filter_size=3, stride=1, out_channels=16, name="enc_conv_1")
        # c_i shape is batch_size x 32 x 32 x 16

        c_i = relu_bn_conv(c_i, 4, 2, 32, self.bn_settings, name="enc_conv_2")
        # c_i shape is batch_size x 16 x 16 x 32

        c_i = relu_bn_conv(c_i, 4, 2, 64, self.bn_settings, name="enc_conv_3")
        # c_i shape is batch_size x 8 x 8 x 64

        c_i = relu_bn_conv(c_i, 4, 2, 128, self.bn_settings, name="enc_conv_4")
        # c_i shape is batch_size x 4 x 4 x 128

        c_i = relu_bn_conv(c_i, 4, 4, self.z_dim, self.bn_settings, name="enc_conv_5")
        # c_i shape is batch_size x 1 x 1 x z_dim

        z = tf.reshape(c_i, shape=[self.batch_size, self.z_dim])
        return z

    def decoder(self, z, reuse=False, hq=False):
        with tf.variable_scope('decoder') as scope:
            if reuse:
                scope.reuse_variables()
            z_dim = self.z_dim
            if self.y_dim:
                z = tf.concat(1, [z, self.y_labels])
                z_dim = self.z_dim + self.y_dim

            c_i = tf.reshape(z, shape=[self.batch_size, 1, 1, z_dim])
            # c_i shape is batch_size x 1 x 1 x z_dim(+y_dim)

            c_i = conv(c_i, filter_size=1, stride=1, out_channels=1024, name="dec_conv_1")
            c_i = PS(c_i, 4, out_dim=64)
            # c_i shape is batch_size x 4 x 4 x 64

            c_i = relu_bn_conv(c_i, 3, 1, 256, self.bn_settings, name="dec_conv_2")
            c_i = PS(c_i, 2, out_dim=64)
            # c_i shape is batch_size x 8 x 8 x 64

            c_i = relu_bn_conv(c_i, 3, 1, 256, self.bn_settings, name="dec_conv_3")
            c_i = PS(c_i, 2, out_dim=64)
            # c_i shape is batch_size x 16 x 16 x 64

            c_i = relu_bn_conv(c_i, 3, 1, 128, self.bn_settings, name="dec_conv_4")
            c_i = PS(c_i, 2, out_dim=32)
            # c_i shape is batch_size x 32 x 32 x 32

            c_i = relu_bn_conv(c_i, 3, 1, 3, self.bn_settings, name="dec_conv_5")

            y_image = tf.nn.tanh(c_i)
            return y_image
