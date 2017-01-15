from src.mode_base import ModelBase

import tensorflow as tf
from src.utils import relu_bn_conv, relu_bn_tconv
from src.utils import conv, tconv, PS


class ModelConvMnist(ModelBase):
    def __init__(self, batch_size, z_dim, y_dim=None, is_training=True):
        super().__init__(batch_size, z_dim, y_dim, is_training)
        self.input_dim = 784
        self.x_image = tf.placeholder(tf.float32, [batch_size, self.input_dim], name='x_image')

        self.n = 4
        self.k = 2

    def encoder(self):
        c_i = tf.reshape(self.x_image, [-1, 28, 28, 1])
        # c_i shape is batch_size x 28 x 28 x 1

        c_i = conv(c_i, filter_size=3, stride=1, out_channels=16*self.k, name="enc_conv_1")
        # c_i shape is batch_size x 28 x 28 x 16

        c_i = self._block_enc(c_i, fn=16 * self.k, name="enc_b1")
        # Now c_i has shape batch_size x 14 x 14 x 32*k

        c_i = self._block_enc(c_i, fn=32 * self.k, name="enc_b2")
        # Now c_i has shape batch_size x 7 x 7 x 64*k

        c_i = self._block_enc(c_i, fn=64 * self.k, name="enc_b1")
        # Now c_i has shape batch_size x 4 x 4 x 128*k

        c_i = relu_bn_conv(c_i, 4, 4, self.z_dim, self.bn_settings, name="enc_end")
        # Now c_i has shape batch_size x 1 x 1 x self.dim

        z = tf.reshape(c_i, shape=[self.batch_size, self.z_dim])
        return z

    def _block_enc(self, c_i, fn, name):
        for i in range(1, self.n):
            l_i = c_i
            c_i = relu_bn_conv(c_i, 3, 1, fn, self.bn_settings, name="%s_blockA%d" % (name, i))
            c_i = relu_bn_conv(c_i, 3, 1, fn, self.bn_settings, name="%s_blockB%d" % (name, i))
            c_i = c_i + l_i

        # Decrease dimension
        l_i = c_i
        c_i = relu_bn_conv(c_i, 3, 1, fn, self.bn_settings, name="%s_blockA_f" % name)
        c_i = relu_bn_conv(c_i, 4, 2, fn*2, self.bn_settings, name="%s_blockB_f" % name)

        l_i = conv(l_i, 4, 2, fn*2, name="%s_proj" % name)
        c_i = c_i + l_i

        return c_i

    def decoder(self, z, hq=False, reuse=False):
        with tf.variable_scope('decoder') as scope:
            if reuse:
                scope.reuse_variables()
            z_dim = self.z_dim
            if self.y_dim:
                z = tf.concat(1, [z, self.y_labels])
                z_dim = self.z_dim + self.y_dim

            c_i = tf.reshape(z, shape=[self.batch_size, 1, 1, z_dim])
            # c_i shape is batch_size x 1 x 1 x z_dim(+y_dim)

            # TODO: Change to 512*self.k
            c_i = tconv(c_i, filter_size=7, stride=7, batch_size=self.batch_size,
                        out_channels=512, name="dec_tconv_1")
            # c_i shape is batch_size x 7 x 7 x 64

            c_i = conv(c_i, filter_size=1, stride=1, out_channels=256 * 4 * self.k, name="dec_conv_1")
            c_i = PS(c_i, 4, out_dim=256 * self.k)
            # c_i shape is batch_size x 14 x 14 x 256*k

            c_i = self._block_dec(c_i, fn=256 * self.k, name="dec_b1")
            c_i = PS(c_i, 2, out_dim=64 * self.k)
            # c_i shape is batch_size x 28 x 28 x 64*k

            c_i = relu_bn_tconv(c_i, 3, 1, self.batch_size, 1, self.bn_settings, name="dec_tconv_4")
            # c_i shape is batch_size x 28 x 28 x 1

            c_i = tf.nn.sigmoid(c_i)
            y_image = tf.reshape(c_i, shape=[self.batch_size, self.input_dim])
            return y_image

    def _block_dec(self, c_i, fn, name):
        l_i = conv(c_i, 3, 1, fn, name="%s_proj" % name)

        for i in range(self.n):
            c_i = relu_bn_conv(c_i, 3, 1, fn, self.bn_settings, name="%s_blockA%d" % (name, i))
            c_i = relu_bn_conv(c_i, 3, 1, fn, self.bn_settings, name="%s_blockB%d" % (name, i))
            c_i = c_i + l_i
            l_i = c_i

        return c_i
