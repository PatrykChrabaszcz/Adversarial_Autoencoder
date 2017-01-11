import tensorflow as tf


class ModelBase:
    def __init__(self, batch_size, z_dim, y_dim=None, is_training=True):
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.y_labels = tf.placeholder(tf.float32, [batch_size, y_dim], name='y_labels')
        self.bn_settings = {'decay': 0.9,
                            'updates_collections': None,
                            'scale': True,
                            'epsilon': 1e-05,
                            "is_training": is_training}

    # Has to be implemented in subclass
    def encoder(self):
        pass

    # Has to be implemented in subclass
    def decoder(self, z, hq=False, reuse=False):
        pass

    def discriminator(self, z, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            if self.y_dim:
                current_input = tf.concat(1, [z, self.y_labels])
                input_dim = self.z_dim + self.y_dim
            else:
                current_input = z
                input_dim = self.z_dim

            neuron_numbers = [500, 500]
            for i, n in enumerate(neuron_numbers):
                w = tf.get_variable('W_disc_dens%d' % i, shape=[input_dim, n],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('b_disc_dens%d' % i, shape=[n], initializer=tf.constant_initializer())
                current_input = tf.matmul(current_input, w) + b
                current_input = tf.maximum(0.2 * current_input, current_input)

                input_dim = n

            w = tf.get_variable('W_disc_out', shape=[input_dim, 1],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b_disc_out', shape=[1],
                                initializer=tf.constant_initializer())
            y = tf.matmul(current_input, w) + b
        return y

    def sampler(self):
        z = tf.truncated_normal([self.batch_size, self.z_dim], name='sampled_z')
        return z
