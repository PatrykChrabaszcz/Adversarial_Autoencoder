import tensorflow as tf
import numpy as np

from src.mode_base import ModelBase
from src.utils import lin, lin_relu_bn


class ModelHqMnist(ModelBase):

    def __init__(self, batch_size, z_dim, y_dim=None, is_training=False):
        super().__init__(batch_size, z_dim, y_dim, is_training)
        self.neuron_numbers = [1000, 1000]
        self.input_dim = 784
        self.x_image = tf.placeholder(tf.float32, [batch_size, self.input_dim], name='x_image')

    def encoder(self):
        c_i = self.x_image
        for i, n in enumerate(self.neuron_numbers):
            c_i = lin_relu_bn(c_i, n, self.bn_settings, "enc_%d" % i)

        z = lin(c_i, self.z_dim, "enc_fin")

        return z

    def decoder(self, z, reuse=False):
        with tf.variable_scope('decoder') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()

            n_n = 1000
            gen_n_points = 28 * 28
            scale = 1.0
            x, y, r = self.coordinates(28, 28, 1.0)
            if hq:
                scale = 10.0
                gen_n_points = 280*280
                x, y, r = self.coordinates(280, 280, 10.0)

            z_scaled = tf.reshape(z, [self.batch_size, 1, self.z_dim]) * \
                       tf.ones([gen_n_points, 1], dtype=tf.float32) * scale
            z_unroll = tf.reshape(z_scaled, [self.batch_size * gen_n_points, self.z_dim])
            x_unroll = tf.reshape(x, [self.batch_size * gen_n_points, 1])
            y_unroll = tf.reshape(y, [self.batch_size * gen_n_points, 1])
            r_unroll = tf.reshape(r, [self.batch_size * gen_n_points, 1])
            x_unroll = tf.cast(x_unroll, dtype=tf.float32)
            y_unroll = tf.cast(y_unroll, dtype=tf.float32)
            r_unroll = tf.cast(r_unroll, dtype=tf.float32)

            U = lin(z_unroll, n_n, 'dec_g_0_z') + \
                lin(x_unroll, n_n, 'dec_g_0_x') + \
                lin(y_unroll, n_n, 'dec_g_0_y') + \
                lin(r_unroll, n_n, 'dec_g_0_r')

            H = tf.nn.softplus(U)

            for i in range(1, 2):
                H = tf.nn.tanh(lin(H, n_n, 'dec_g_tanh_%d' % i))

            output = tf.sigmoid(lin(H, 1, 'dec_g_fin'))
            x_rec = tf.reshape(output, [self.batch_size, gen_n_points])

            return x_rec

    def coordinates(self, x_dim=32, y_dim=32, scale=1.0):
        n_pixel = x_dim * y_dim
        x_range = scale * (np.arange(x_dim) - (x_dim - 1) / 2.0) / (x_dim - 1) / 0.5
        y_range = scale * (np.arange(y_dim) - (y_dim - 1) / 2.0) / (y_dim - 1) / 0.5
        x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
        y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
        r_mat = np.sqrt(x_mat * x_mat + y_mat * y_mat)
        x_mat = np.tile(x_mat.flatten(), self.batch_size).reshape(self.batch_size, n_pixel, 1)
        y_mat = np.tile(y_mat.flatten(), self.batch_size).reshape(self.batch_size, n_pixel, 1)
        r_mat = np.tile(r_mat.flatten(), self.batch_size).reshape(self.batch_size, n_pixel, 1)
        return x_mat, y_mat, r_mat
