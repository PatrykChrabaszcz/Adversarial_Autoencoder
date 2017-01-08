import tensorflow as tf
from tensorflow.contrib.layers import batch_norm


class ModelConvMnist:

    def __init__(self, batch_size, z_dim):
        self.batch_size = batch_size
        self.input_dim = 784
        self.z_dim = z_dim
        self.x_image = tf.placeholder(tf.float32, [batch_size, self.input_dim], name='x_image')
        self.bn_settings = {'decay': 0.9,
                            'updates_collections': None,
                            'scale': True,
                            'epsilon': 1e-05}

    def encoder(self):
        f_n = [64, 128, 256]
        f_s = [5, 5, 4]
        x_image = tf.reshape(self.x_image, [-1, 28, 28, 1])
        current_input = x_image
        for i, n_output in enumerate(f_n):
            n_input = current_input.get_shape().as_list()[3]
            w_shape = [
             f_s[i], f_s[i], n_input, n_output]
            w = tf.get_variable('W_enc_conv%d' % i, shape=w_shape, initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b_enc_conv%d' % i, shape=[n_output], initializer=tf.constant_initializer(0))
            current_input = tf.nn.conv2d(current_input, w, strides=[1, 2, 2, 1], padding='VALID') + b
            current_input = batch_norm(current_input, scope=('batch_norm_enc_conv%d' % i), **self.bn_settings)
            current_input = tf.nn.relu(current_input)

        current_input = tf.reshape(current_input, shape=[-1, f_n[2]])
        w = tf.get_variable('W_dec_full%d' % i, shape=[f_n[2], self.z_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b_dec_full%d' % i, shape=[self.z_dim],
                            initializer=tf.constant_initializer(0))
        z = tf.matmul(current_input, w) + b
        return z

    def decoder(self, z, reuse=False, hq=False):
        with tf.variable_scope('decoder') as scope:
            if reuse:
                scope.reuse_variables()
            f_s = [7, 14, 28]
            f_n = [128, 64, 1]
            full_layer_size = [1024, f_s[0] * f_s[0] * f_n[0]]
            current_input = z
            n_prev = self.z_dim
            for i, neuron_num in enumerate(full_layer_size):
                w = tf.get_variable('W_dec_full%d' % i, shape=[n_prev, neuron_num],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('b_dec_full%d' % i, shape=[neuron_num],
                                    initializer=tf.constant_initializer(0))
                n_prev = neuron_num
                current_input = tf.matmul(current_input, w) + b
                current_input = batch_norm(current_input, scope=('batch_norm_dec_full%d' % i), **self.bn_settings)
                current_input = tf.nn.relu(current_input)

            current_input = tf.reshape(current_input, shape=[self.batch_size, f_s[0], f_s[0], f_n[0]])
            for i in range(1, len(f_s)):
                w = tf.get_variable('W_dec_tconv%d' % i, shape=[5, 5, f_n[i], f_n[i - 1]],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('b_dec_tconv%d' % i, shape=f_n[i], initializer=tf.constant_initializer(0))
                current_input = tf.nn.conv2d_transpose(current_input, w, [self.batch_size, f_s[i], f_s[i], f_n[i]],
                                                       strides=[1, 2, 2, 1])
                current_input = current_input + b
                if i != len(f_s) - 1:
                    current_input = batch_norm(current_input,
                                               scope=('batch_norm_dec_conv%d' % (i - 1)), **self.bn_settings)
                    current_input = tf.nn.relu(current_input)

            current_input = tf.nn.sigmoid(current_input)
            y_image = tf.reshape(current_input, shape=[-1, self.input_dim])
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
                current_input = tf.nn.relu(current_input)
                input_dim = n

            w = tf.get_variable('W_disc_out', shape=[input_dim, 1],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b_disc_out', shape=[1], initializer=tf.constant_initializer())
            y = tf.matmul(current_input, w) + b
        return y

    def sampler(self):
        z = tf.random_uniform([self.batch_size, self.z_dim], -1, 1, name='sampled_z')
        return z
