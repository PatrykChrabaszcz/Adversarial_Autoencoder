import tensorflow as tf
from utils import lrelu


class ModelDenseMnist:
    def __init__(self, input_shape, z_dim):

        self.neuron_numbers = [1000, 1000]
        self.z_dim = z_dim
        self.input_shape = input_shape

    # TODO: Add batch normalization

    def encoder(self, x_image):
        current_input = x_image
        input_dim = self.input_shape[1]

        # Dense layer part
        for i, n in enumerate(self.neuron_numbers):
            w = tf.get_variable('W_enc_dens%d' % i, shape=[input_dim, n],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b_enc_dens%d' % i, shape=[n],
                                initializer=tf.constant_initializer())
            current_input = lrelu(tf.matmul(current_input, w) + b)
            input_dim = n

        w = tf.get_variable('W_enc_out', shape=[input_dim, self.z_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b_enc_out', shape=[self.z_dim],
                            initializer=tf.constant_initializer())
        z = tf.matmul(current_input, w) + b

        return z

    def decoder(self, z):
        current_input = z
        input_dim = self.z_dim

        # Dense layer part
        for i, n in enumerate(self.neuron_numbers[::-1]):
            w = tf.get_variable('W_dec_dens%d' % i, shape=[input_dim, n],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b_dec_dens%d' % i, shape=[n],
                                initializer=tf.constant_initializer())
            current_input = lrelu(tf.matmul(current_input, w) + b)
            input_dim = n

        w = tf.get_variable('W_dec_out', shape=[input_dim, self.input_shape[1]],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b_dec_out', shape=[self.input_shape[1]],
                            initializer=tf.constant_initializer())
        x_reconstructed = tf.sigmoid(tf.matmul(current_input, w) + b)

        return x_reconstructed

    def discriminator(self, z, reuse=False):
        neuron_numbers = [100, 100]

        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            current_input = z
            input_dim = self.z_dim
            for i, n in enumerate(neuron_numbers):
                w = tf.get_variable('W_disc_dens%d' % i, shape=[input_dim, n],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('b_disc_dens%d' % i, shape=[n],
                                    initializer=tf.constant_initializer())
                current_input = lrelu(tf.matmul(current_input, w) + b)
                input_dim = n

            w = tf.get_variable('W_disc_out', shape=[input_dim, 1],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b_disc_out', shape=[1],
                                initializer=tf.constant_initializer())
            y = tf.nn.sigmoid(tf.matmul(current_input, w) + b)

        return y

    def sampler(self):
        batch_size = self.input_shape[0]
        z = tf.random_uniform([batch_size, self.z_dim], -5, 5, name='sampled_z')

        return z
