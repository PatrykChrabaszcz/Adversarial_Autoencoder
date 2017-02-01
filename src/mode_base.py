import tensorflow as tf
from src.utils import lin, lin_lrelu


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

    # Has to be implemented in subclass
    def gan(self, x_image, reuse=False, features=False):
        pass

    def discriminator(self, z, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            c_i = z
            if self.y_dim:
                c_i = tf.concat(1, [z, self.y_labels])

            c_i = lin_lrelu(c_i, 500, name="disc_dens_%d" % 0)
            c_i = lin_lrelu(c_i, 500, name="disc_dens_%d" % 1)

            y = lin(c_i, 1, name="disc_out")

        return y

    def sampler(self):
        z = tf.truncated_normal([self.batch_size, self.z_dim], name='sampled_z')
        return z
