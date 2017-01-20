from src.mode_base import ModelBase
from src.utils import *


#   Implements encoder and decoder for 32x32x3 images
#   Uses residual architecture with subpixel upscaling
class ModelSubpixRes32(ModelBase):

    def __init__(self, batch_size, z_dim, channels=3, y_dim=None, is_training=True):
        super().__init__(batch_size, z_dim, y_dim, is_training)
        self.channels = channels
        self.x_image = tf.placeholder(tf.float32, [batch_size, 32, 32, channels], name='x_image')
        self.n = 2
        self.k = 2

    def encoder(self):

        c_i = self.x_image
        # Now c_i has shape batch_size x 32 x 32 x self.channels

        c_i = conv(input=c_i, filter_size=3, stride=1, out_channels=16*self.k, name="enc_conv_in", )
        # Now c_i has shape batch_size x 32 x 32 x 16*k

        c_i = block_res_conv(c_i, 16*self.k, self.n, self.bn_settings, reduce=True, name="enc_b1")
        # Now c_i has shape batch_size x 16 x 16 x 32*k

        c_i = block_res_conv(c_i, 32*self.k, self.n, self.bn_settings, reduce=True, name="enc_b2")
        # Now c_i has shape batch_size x 8 x 8 x 64*k

        c_i = block_res_conv(c_i, 64*self.k, self.n, self.bn_settings, reduce=True, name="enc_b3")
        # Now c_i has shape batch_size x 4 x 4 x 128*k

        c_i = block_res_conv(c_i, 128*self.k, self.n, self.bn_settings, reduce=True, name="enc_b4")
        # Now c_i has shape batch_size x 2 x 2 x 256*k

        c_i = relu_bn_conv(c_i, 2, 2, self.z_dim, self.bn_settings, name="enc_end")
        # Now c_i has shape batch_size x 1 x 1 x self.dim

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

            c_i = conv(c_i, 1, 1, 1024 * self.k, name="dec_conv_1")
            c_i = PS(c_i, 2, 256*self.k)
            # c_i shape is batch_size x 2 x 2 x 256*k

            c_i = block_res_subpix(c_i, 256*self.k, self.n, self.bn_settings, name="dec_b1")
            # c_i shape is batch_size x 4 x 4 x 128*k

            c_i = block_res_subpix(c_i, 128*self.k, self.n, self.bn_settings, name="dec_b2")
            # c_i shape is batch_size x 8 x 8 x 64*k

            c_i = block_res_subpix(c_i, 64*self.k, self.n, self.bn_settings, name="dec_b3")
            # c_i shape is batch_size x 16 x 16 x 32*k

            c_i = block_res_subpix(c_i, 32*self.k, self.n, self.bn_settings, name="dec_b4")
            # c_i shape is batch_size x 32 x 32 x 16*k

            c_i = relu_bn_conv(c_i, 3, 1, self.channels, self.bn_settings, name="dec_end")
            # c_i shape is batch_size x 32 x 32 x self.channels

            y_image = tf.nn.tanh(c_i)
            return y_image

    def gan(self, x_image, reuse, discriminator):
        if discriminator:
            bs = self.batch_size*2
        else:
            bs = self.batch_size

        with tf.variable_scope('gan') as scope:
            if reuse:
                scope.reuse_variables()
            c_i = x_image
            c_i = conv(input=c_i, filter_size=3, stride=1, out_channels=16 * self.k, name="gan_conv_in", )
            # Now c_i has shape bs x 32 x 32 x 16*k

            c_i = block_res_conv(c_i, 16 * self.k, self.n, self.bn_settings, reduce=True, name="gan_b1")
            # Now c_i has shape bs x 16 x 16 x 32*k

            c_i = block_res_conv(c_i, 32 * self.k, self.n, self.bn_settings, reduce=True, name="gan_b2")
            # Now c_i has shape bs x 8 x 8 x 64*k

            c_i = block_res_conv(c_i, 64 * self.k, self.n, self.bn_settings, reduce=True, name="gan_b3")
            # Now c_i has shape bs x 4 x 4 x 128*k

            c_i = block_res_conv(c_i, 128 * self.k, self.n, self.bn_settings, reduce=True, name="gan_b4")
            # Now c_i has shape batch_size x 2 x 2 x 256*k

            c_i = relu_bn_conv(c_i, 2, 2, 1, self.bn_settings, name="gan_end")
            # Now c_i has shape batch_size x 1 x 1 x 1

            y = tf.reshape(c_i, shape=[bs, 1])

            return y
