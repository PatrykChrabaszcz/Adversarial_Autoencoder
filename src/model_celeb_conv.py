import tensorflow as tf
from tensorflow.contrib.layers import batch_norm


class ModelConvCeleb:

    def __init__(self, batch_size, z_dim, y_dim):
        self.batch_size = batch_size
        self.input_dim = [32, 32, 3]
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.x_image = tf.placeholder(tf.float32, [batch_size, 32, 32, 3], name='x_image')
        self.y_labels = tf.placeholder(tf.float32, [batch_size, y_dim], name='y_labels')
        self.bn_settings = {'decay': 0.9,
                            'updates_collections': None,
                            'scale': True,
                            'epsilon': 1e-05}

    def encoder(self):
        # Image size in next layers
        # i_s = [32, 16, 8, 4, 1]
        # Channels in next layers
        f_n = [3, 128, 256, 512, self.z_dim]
        # Filter sizes (3x3xfn[i] for each i-th layer)
        f_s = [None, 3, 3, 3, 4]
        padding = [None, 'SAME', 'SAME', 'SAME', 'VALID']

        current_input = self.x_image
        for i in range(1, len(f_n)):
            w = tf.get_variable('W_enc_conv%d' % i, shape=[f_s[i], f_s[i], f_n[i-1], f_n[i]],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b_enc_conv%d' % i, shape=[f_n[i]],
                                initializer=tf.constant_initializer(0))
            current_input = tf.nn.conv2d(current_input, w, strides=[1, 2, 2, 1], padding=padding[i]) + b
            current_input = batch_norm(current_input, scope=('batch_norm_enc_conv%d' % i), **self.bn_settings)
            current_input = tf.maximum(0.2*current_input, current_input)

        z = tf.reshape(current_input, shape=[self.batch_size, self.z_dim])
        return z

    def decoder(self, z, reuse=False, hq=False):
        with tf.variable_scope('decoder') as scope:
            if reuse:
                scope.reuse_variables()

            if self.y_dim:
                current_input = tf.concat(1, [z, self.y_labels])
                input_dim = self.z_dim + self.y_dim
            else:
                current_input = z
                input_dim = self.z_dim

            i_s = [1, 4, 8, 16, 32]
            f_n = [input_dim, 512, 256, 128, 3]
            f_s = [None, 4, 3, 3, 3]
            padding = [None, 'VALID', 'SAME', 'SAME', 'SAME']
            current_input = tf.reshape(current_input, [self.batch_size, 1, 1, input_dim])
            for i in range(1, len(i_s)):
                w = tf.get_variable('W_dec_tconv%d' % i, shape=[f_s[i], f_s[i], f_n[i], f_n[i - 1]],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('b_dec_tconv%d' % i, shape=f_n[i],
                                    initializer=tf.constant_initializer(0))
                current_input = tf.nn.conv2d_transpose(current_input, w, [self.batch_size, i_s[i], i_s[i], f_n[i]],
                                                       strides=[1, 2, 2, 1], padding=padding[i])
                current_input = current_input + b
                if i != len(i_s) - 1:
                    current_input = batch_norm(current_input,
                                               scope=('batch_norm_dec_conv%d' % (i - 1)), **self.bn_settings)
                    current_input = tf.maximum(0.2*current_input, current_input)

            current_input = tf.nn.tanh(current_input)
            y_image = tf.reshape(current_input, shape=[self.batch_size, 32, 32, 3])
            return y_image

    def discriminator(self, z, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            neuron_numbers = [250, 250]
            current_input = z
            input_dim = self.z_dim
            for i, n in enumerate(neuron_numbers):
                w = tf.get_variable('W_disc_dens%d' % i, shape=[input_dim, n],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('b_disc_dens%d' % i, shape=[n], initializer=tf.constant_initializer())
                current_input = tf.matmul(current_input, w) + b
                current_input = tf.maximum(0.2 * current_input, current_input)
                input_dim = n

            w = tf.get_variable('W_disc_out', shape=[input_dim, 1],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b_disc_out', shape=[1], initializer=tf.constant_initializer())
            y = tf.matmul(current_input, w) + b
        return y

    def sampler(self):
        z = tf.truncated_normal([self.batch_size, self.z_dim], name='sampled_z')
        return z
