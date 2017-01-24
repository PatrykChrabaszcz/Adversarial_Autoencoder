from src.mode_base import ModelBase

import tensorflow as tf
from src.utils import lin, lin_relu_bn


class ModelDenseCell(ModelBase):
    def __init__(self, batch_size, z_dim, y_dim=None, is_training=True):
        super().__init__(batch_size=batch_size, z_dim=z_dim, y_dim=y_dim, is_training=is_training)
        self.neuron_numbers = [2048, 2048]
        self.input_dim = 4096
        self.x_image = tf.placeholder(tf.float32, [batch_size, self.input_dim], name='x_image')

    def encoder(self):
        c_i = self.x_image
        for i, n in enumerate(self.neuron_numbers):
            c_i = lin_relu_bn(c_i, n, self.bn_settings,  name="enc_dens_%d" % i)

        z = lin(c_i, self.z_dim, name="enc_out")
        return z

    def decoder(self, z, reuse=False):
        with tf.variable_scope('decoder') as scope:
            c_i = z
            if self.y_dim:
                c_i = tf.concat(1, [z, self.y_labels])

            if reuse:
                scope.reuse_variables()
            for i, n in enumerate(self.neuron_numbers[::-1]):
                c_i = lin_relu_bn(c_i, n, self.bn_settings, name="dec_dens_%d" % i)

            c_i = lin(c_i, self.input_dim, name="dec_out")
            x_reconstructed = tf.nn.sigmoid(c_i)
            return x_reconstructed
