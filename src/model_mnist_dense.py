import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from src.mode_base import ModelBase


class ModelDenseMnist(ModelBase):
    def __init__(self, batch_size, z_dim, y_dim=None, is_training=True):
        super().__init__(batch_size=batch_size, z_dim=z_dim, y_dim=y_dim, is_training=is_training)
        self.neuron_numbers = [1000, 1000]
        self.input_dim = 784
        self.x_image = tf.placeholder(tf.float32, [batch_size, self.input_dim], name='x_image')

    def encoder(self):
        current_input = self.x_image
        input_dim = self.input_dim
        for i, n in enumerate(self.neuron_numbers):
            w = tf.get_variable('W_enc_dens%d' % i, shape=[input_dim, n],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b_enc_dens%d' % i, shape=[n],
                                initializer=tf.constant_initializer())
            current_input = tf.matmul(current_input, w) + b
            current_input = batch_norm(current_input, scope=('batch_norm_enc%d' % i), **self.bn_settings)
            current_input = tf.maximum(0.2 * current_input, current_input)
            input_dim = n

        w = tf.get_variable('W_enc_out', shape=[input_dim, self.z_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b_enc_out', shape=[self.z_dim],
                            initializer=tf.constant_initializer())
        z = tf.matmul(current_input, w) + b
        return z

    def decoder(self, z, hq=False, reuse=False):
        with tf.variable_scope('decoder') as scope:
            if reuse:
                scope.reuse_variables()
            if self.y_dim:
                current_input = tf.concat(1, [z, self.y_labels])
                input_dim = self.z_dim + self.y_dim
            else:
                current_input = z
                input_dim = self.z_dim

            for i, n in enumerate(self.neuron_numbers[::-1]):
                w = tf.get_variable('W_dec_dens%d' % i, shape=[input_dim, n],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('b_dec_dens%d' % i, shape=[n],
                                    initializer=tf.constant_initializer())
                current_input = tf.matmul(current_input, w) + b
                current_input = batch_norm(current_input, scope=('batch_norm_dec%d' % i), **self.bn_settings)
                current_input = tf.maximum(0.2 * current_input, current_input)
                input_dim = n

            w = tf.get_variable('W_dec_out', shape=[input_dim, self.input_dim],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b_dec_out', shape=[self.input_dim], initializer=tf.constant_initializer())
            x_reconstructed = tf.matmul(current_input, w) + b
            x_reconstructed = tf.nn.sigmoid(x_reconstructed)
            return x_reconstructed
