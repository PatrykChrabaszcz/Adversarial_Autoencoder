import tensorflow as tf
from src.utils import lin


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
    def decoder(self, z, reuse=False):
        pass

    def discriminator(self, z, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            c_i = z
            if self.y_dim:
                c_i = tf.concat(1, [z, self.y_labels])

            neuron_numbers = [500, 500]
            for i, n in enumerate(neuron_numbers):
                c_i = lin(c_i, n, name="disc_dens_%d" % i)
                c_i = tf.maximum(0.2 * c_i, c_i)

            y = lin(c_i, 1, name="disc_out")

        return y

    def sampler(self):
        z = tf.truncated_normal([self.batch_size, self.z_dim], name='sampled_z')
        return z
