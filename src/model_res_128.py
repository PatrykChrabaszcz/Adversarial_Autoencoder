from src.mode_base import ModelBase

import tensorflow as tf
from src.utils import relu_bn_conv, conv
from src.utils import relu_bn_tconv, tconv
from src.utils import block_res_conv, block_res_tconv


class ModelRes128(ModelBase):

    def __init__(self, batch_size, z_dim, channels=3, y_dim=None, is_training=True):
        super().__init__(batch_size, z_dim, y_dim, is_training)
        self.channels = channels
        self.x_image = tf.placeholder(tf.float32, [batch_size, 128, 128, channels], name='x_image')
        self.n = 2
        self.k = 1

    def encoder(self):

        c_i = self.x_image
        # Now c_i has shape batch_size x 128 x 128 x 3

        c_i = conv(input=c_i, filter_size=3, stride=1, out_channels=16*self.k, name="enc_conv_in", )
        # Now c_i has shape batch_size x 128 x 128 x 16*k

        c_i = block_res_conv(c_i, 16*self.k, self.n, self.bn_settings, reduce=True, name="enc_b1")
        # Now c_i has shape batch_size x 64 x 64 x 32*k

        c_i = block_res_conv(c_i, 32*self.k, self.n, self.bn_settings, reduce=True, name="enc_b2")
        # Now c_i has shape batch_size x 32 x 32 x 64*k

        c_i = block_res_conv(c_i, 64*self.k, self.n, self.bn_settings, reduce=True, name="enc_b3")
        # Now c_i has shape batch_size x 16 x 16 x 128*k

        c_i = block_res_conv(c_i, 128*self.k, self.n, self.bn_settings, reduce=True, name="enc_b4")
        # Now c_i has shape batch_size x 8 x 8 x 256*k

        c_i = block_res_conv(c_i, 256*self.k, self.n, self.bn_settings, reduce=True, name="enc_b5")
        # Now c_i has shape batch_size x 4 x 4 x 512*k

        c_i = relu_bn_conv(c_i, 4, 4, self.z_dim, self.bn_settings, name="enc_end")
        # Now c_i has shape batch_size x 1 x 1 x self.z_dim

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
            # Now c_i has shape batch_size x 1 x 1 x self.z_dim (+ self.y_dim)

            c_i = tconv(c_i, filter_size=1, stride=4, batch_size=self.batch_size,
                        out_channels=512*self.k, name='dec_tconv_in')
            # Now c_i has shape batch_size x 4 x 4 x 512*k

            c_i = block_res_tconv(c_i, 512*self.k, self.n, self.batch_size, self.bn_settings, name='dec_b1')
            # Now c_i has shape batch_size x 8 x 8 x 256*k

            c_i = block_res_tconv(c_i, 256 * self.k, self.n, self.batch_size, self.bn_settings, name='dec_b2')
            # Now c_i has shape batch_size x 16 x 16 x 128*k

            c_i = block_res_tconv(c_i, 128 * self.k, self.n, self.batch_size, self.bn_settings, name='dec_b3')
            # Now c_i has shape batch_size x 32 x 32 x 64*k

            c_i = block_res_tconv(c_i, 64 * self.k, self.n, self.batch_size, self.bn_settings, name='dec_b4')
            # Now c_i has shape batch_size x 64 x 64 x 32*k

            c_i = block_res_tconv(c_i, 32 * self.k, self.n, self.batch_size, self.bn_settings, name='dec_b5')
            # Now c_i has shape batch_size x 128 x 128 x 16*k

            c_i = relu_bn_conv(c_i, 3, 1, self.channels, self.bn_settings, name="dec_end")
            # c_i shape is batch_size x 128 x 128 x 3

            y_image = tf.nn.tanh(c_i)
            return y_image

    def gan(self, x_image):
        c_i = x_image
        # Now c_i has shape batch_size x 128 x 128 x 3

        c_i = conv(input=c_i, filter_size=3, stride=1, out_channels=16 * self.k, name="gan_conv_in", )
        # Now c_i has shape batch_size x 128 x 128 x 16*k

        c_i = self._block_enc(c_i, fn=16 * self.k, name="gan_b1")
        # Now c_i has shape batch_size x 64 x 64 x 32*k

        c_i = self._block_enc(c_i, fn=32 * self.k, name="gan_b2")
        # Now c_i has shape batch_size x 32 x 32 x 64*k

        c_i = self._block_enc(c_i, fn=64 * self.k, name="gan_b3")
        # Now c_i has shape batch_size x 16 x 16 x 128*k

        # c_i = self._block_enc(c_i, fn=128 * self.k, name="gan_b4")
        # # Now c_i has shape batch_size x 8 x 8 x 256*k
        #
        # c_i = self._block_enc(c_i, fn=256 * self.k, name="gan_b5")
        # # Now c_i has shape batch_size x 4 x 4 x 512*k

        c_i = relu_bn_conv(c_i, 4, 4, 1, self.bn_settings, name="gan_end")
        # Now c_i has shape batch_size x 1 x 1 x self.dim

        y = tf.reshape(c_i, shape=[self.batch_size, 1])

        return y
