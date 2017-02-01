from src.mode_base import ModelBase

import tensorflow as tf
from src.utils import lin, lin_bn_lrelu


class ModelDenseMnist(ModelBase):
    def __init__(self, batch_size, z_dim, y_dim=None, is_training=True):
        super().__init__(batch_size=batch_size, z_dim=z_dim, y_dim=y_dim, is_training=is_training)
        self.neuron_numbers = [1000, 1000]
        self.input_dim = 784
        self.x_image = tf.placeholder(tf.float32, [batch_size, self.input_dim], name='x_image')

    def encoder(self):
        c_i = self.x_image
        c_i = lin_bn_lrelu(c_i, 1000, self.bn_settings, name="enc_dens_%d" % 0)
        c_i = lin_bn_lrelu(c_i, 1000, self.bn_settings, name="enc_dens_%d" % 1)

        z = lin(c_i, self.z_dim, name="enc_out")
        return z

    def decoder(self, z, reuse=False):
        with tf.variable_scope('decoder') as scope:
            c_i = z
            if self.y_dim:
                c_i = tf.concat(1, [z, self.y_labels])

            if reuse:
                scope.reuse_variables()

            c_i = lin_bn_lrelu(c_i, 1000, self.bn_settings, name="dec_dens_%d" % 0)
            c_i = lin_bn_lrelu(c_i, 1000, self.bn_settings, name="dec_dens_%d" % 1)

            c_i = lin(c_i, self.input_dim, name="dec_out")
            x_reconstructed = tf.nn.sigmoid(c_i)
            return x_reconstructed

    def gan(self, x_image, reuse=False, features=False):

        with tf.variable_scope('gan') as scope:
            if reuse:
                scope.reuse_variables()
            c_i = x_image

            c_i = lin_bn_lrelu(c_i, 1000, self.bn_settings, name="gan_dens_%d" % 0)
            if features:
                return c_i
            c_i = lin_bn_lrelu(c_i, 1000, self.bn_settings, name="gan_dens_%d" % 1)
            y = lin(c_i, 1, "gan_out")

            return y
