from src.mode_base import ModelBase

import tensorflow as tf
from src.utils import relu_bn_tconv, relu_bn_conv, conv, tconv


class ModelResCeleb(ModelBase):

    def __init__(self, batch_size, z_dim, y_dim=None, is_training=True):
        super().__init__(batch_size, z_dim, y_dim, is_training)
        self.x_image = tf.placeholder(tf.float32, [batch_size, 32, 32, 3], name='x_image')
        self.n = 1
        self.k = 1

    def encoder(self):

        c_i = self.x_image
        # Now c_i has shape batch_size x 32 x 32 x 3

        c_i = conv(input=c_i, filter_size=3, stride=1, out_channels=16*self.k, name="enc_conv_in", )
        # Now c_i has shape batch_size x 32 x 32 x 16*k

        c_i = self._block_enc(c_i, fn=16*self.k, name="enc_b1")
        # Now c_i has shape batch_size x 16 x 16 x 32*k

        c_i = self._block_enc(c_i, fn=32*self.k, name="enc_b2")
        # Now c_i has shape batch_size x 8 x 8 x 64*k

        c_i = self._block_enc(c_i, fn=64*self.k, name="enc_b3")
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

            c_i = tconv(c_i, filter_size=4, stride=4, batch_size=self.batch_size,
                        out_channels=128*self.k, name="dec_tconv_1")
            # c_i shape is batch_size x 4 x 4 x 128*k

            c_i = self._block_dec(c_i, name="dec_b1", fn=128*self.k)
            # c_i shape is batch_size x 8 x 8 x 64*k

            c_i = self._block_dec(c_i, name="dec_b2", fn=64*self.k)
            # c_i shape is batch_size x 16 x 16 x 32*k

            c_i = self._block_dec(c_i, name="dec_b3", fn=32*self.k)
            # c_i shape is batch_size x 32 x 32 x 16*k

            c_i = relu_bn_tconv(c_i, 3, 1, self.batch_size, 3, self.bn_settings, name="dec_end")
            # c_i shape is batch_size x 32 x 32 x 3

            y_image = tf.nn.tanh(c_i)
            return y_image

    def _block_dec(self, c_i, fn, name):
        for i in range(1, self.n):
            l_i = c_i
            c_i = relu_bn_tconv(c_i, 3, 1, self.batch_size, fn, self.bn_settings, name="%s_blockA%d" % (name, i))
            c_i = relu_bn_tconv(c_i, 3, 1, self.batch_size, fn, self.bn_settings, name="%s_blockB%d" % (name, i))
            c_i = c_i + l_i

        l_i = c_i
        # Increase dimension
        c_i = relu_bn_tconv(c_i, 3, 1, self.batch_size, fn, self.bn_settings, name="%s_blockA_f" % name)
        c_i = relu_bn_tconv(c_i, 4, 2, self.batch_size, fn//2, self.bn_settings, name="%s_blockB_f" % name)

        l_i = tconv(l_i, 4, 2, self.batch_size, fn//2, name="%s_proj" % name)
        c_i = c_i + l_i

        return c_i
