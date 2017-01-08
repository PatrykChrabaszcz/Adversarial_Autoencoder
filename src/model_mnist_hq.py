import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import batch_norm


class ModelHqMnist:

    def __init__(self, batch_size, z_dim):
        self.neuron_numbers = [250, 100]
        self.batch_size = batch_size
        self.input_dim = 784
        self.z_dim = z_dim
        self.x_image = tf.placeholder(tf.float32, [batch_size, self.input_dim], name='x_image')
        self.bn_settings = {'decay': 0.9,
                            'updates_collections': None,
                            'scale': True,
                            'epsilon': 1e-05}

    def encoder(self):
        current_input = self.x_image
        input_dim = self.input_dim
        for i, n in enumerate(self.neuron_numbers):
            w = tf.get_variable('W_enc_dens%d' % i, shape=[input_dim, n],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b_enc_dens%d' % i, shape=[n], initializer=tf.constant_initializer())
            current_input = tf.matmul(current_input, w) + b
            current_input = batch_norm(current_input, scope=('batch_norm_enc%d' % i), **self.bn_settings)
            current_input = tf.nn.relu(current_input)
            input_dim = n

        w = tf.get_variable('W_enc_out', shape=[input_dim, self.z_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b_enc_out', shape=[self.z_dim],
                            initializer=tf.constant_initializer())
        z = tf.matmul(current_input, w) + b
        return z

    def decoder(self, z, reuse=False, hq=False):
        with tf.variable_scope('decoder') as scope:
            if reuse:
                scope.reuse_variables()
            scale = 1
            if hq:
                scale = 4
            coordinates = tf.convert_to_tensor(self._coordinates(scale=scale), dtype=tf.float32)
            z = tf.tile(z, [1, 784*scale*scale])
            z = tf.reshape(z, shape=[784*scale*scale*self.batch_size, self.z_dim])
            x = tf.concat(1, [coordinates, z])
            current_input = x
            input_dim = self.z_dim + 2
            for i, n in enumerate(self.neuron_numbers[::-1]):
                w = tf.get_variable('W_dec_dens%d' % i, shape=[input_dim, n],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('b_dec_dens%d' % i, shape=[n], initializer=tf.constant_initializer())
                current_input = tf.matmul(current_input, w) + b
                current_input = batch_norm(current_input, scope=('batch_norm_dec%d' % i), **self.bn_settings)
                current_input = tf.nn.relu(current_input)
                input_dim = n

            w = tf.get_variable('W_dec_out', shape=[input_dim, 1],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b_dec_out', shape=[1], initializer=tf.constant_initializer())
            x_reconstructed = tf.matmul(current_input, w) + b
            x_reconstructed = tf.nn.sigmoid(x_reconstructed)

            x_reconstructed = tf.reshape(x_reconstructed, [self.batch_size, self.input_dim*scale*scale])

            return x_reconstructed

    def discriminator(self, z, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            neuron_numbers = [
             250, 250]
            current_input = z
            input_dim = self.z_dim
            for i, n in enumerate(neuron_numbers):
                w = tf.get_variable('W_disc_dens%d' % i, shape=[input_dim, n],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('b_disc_dens%d' % i, shape=[n],
                                    initializer=tf.constant_initializer())
                current_input = tf.matmul(current_input, w) + b
                #current_input = batch_norm(current_input, scope=('batch_norm_disc%d' % i), **self.bn_settings)
                current_input = tf.nn.relu(current_input)
                input_dim = n

            w = tf.get_variable('W_disc_out', shape=[input_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b_disc_out', shape=[1], initializer=tf.constant_initializer())
            y = tf.matmul(current_input, w) + b
        return y

    def sampler(self):
        z = tf.random_uniform([self.batch_size, self.z_dim], -1, 1, name='sampled_z')
        return z

    def _coordinates(self, scale=1):
        px = 28
        scale_f = float(scale)
        x = np.arange(px * scale) / scale_f
        x = np.array([[i] * self.batch_size for i in x]).reshape([px * scale * self.batch_size, 1])
        x = np.tile(x, [px*scale, 1])
        y = np.array([[i / scale_f] * px * scale * self.batch_size for i in range(px * scale)]).\
            reshape([px * px * scale * scale * self.batch_size, 1])
        return np.concatenate([x, y], axis=1)
