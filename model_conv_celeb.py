import tensorflow as tf
from utils import lrelu


class ModelConvCeleb:
    def __init__(self, batch_size, z_dim):
        # Shape of data after each conv
        self.data_shapes = []
        # Shape of filters in each conv
        self.filter_shapes = []

        # Network Architecture
        self.filter_numbers = [16, 32, 64, z_dim]
        self.strides = [2, 2, 1, 1]
        self.filter_sizes = [4, 3, 4, 4]

        self.batch_size = batch_size
        self.z_dim = z_dim

        # Input image
        self.x_image = tf.placeholder(tf.float32, [batch_size, 32, 32, 3], name='x_image')

    # TODO: Add batch normalization

    def encoder(self):
        current_input = self.x_image
        # Encoder part
        for i, n_output in enumerate(self.filter_numbers):
            # Extract number of channels/filters in current input
            n_input = current_input.get_shape().as_list()[3]

            # Remember shapes, decoder will use this info to create symmetric structure
            self.data_shapes.append(current_input.get_shape().as_list())
            w_shape = [self.filter_sizes[i], self.filter_sizes[i], n_input, n_output]
            self.filter_shapes.append(w_shape)
            w = tf.get_variable("W_enc_conv%d" % i, shape=w_shape,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b_enc_conv%d" % i, shape=[n_output],
                                initializer=tf.constant_initializer(0))
            s = self.strides[i]
            output = lrelu(tf.nn.conv2d(current_input, w, strides=[1, s, s, 1], padding='VALID') + b)
            current_input = output

        # Latent vector
        # We reshape it from [batch_size, 1, 1, z_dim] to [batch_size, n_dim]
        z = tf.reshape(current_input, shape=[-1, self.z_dim])

        return z

    # TODO: Add batch normalization
    # TODO: Change last activation to tanh
    # Can be called after encoder was called
    def decoder(self, z):

        # Reshape z vector so it looks like output from encoder before reshaping was done
        z = tf.reshape(z, shape=[-1, 1, 1, self.z_dim])
        # Decoder part
        self.data_shapes.reverse()
        self.filter_shapes.reverse()
        self.strides.reverse()
        current_input = z
        for i, shape in enumerate(self.data_shapes):
            w = tf.get_variable("W_dec_tconv%d" % i, shape=self.filter_shapes[i],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b_dec_tconv%d" % i, shape=self.filter_shapes[i][2],
                                initializer=tf.constant_initializer(0))
            s = self.strides[i]
            output = lrelu(tf.nn.conv2d_transpose(current_input, w, shape,
                                                  strides=[1, s, s, 1], padding='VALID') + b)
            current_input = output

        y_image = current_input

        return y_image

    def discriminator(self, z, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            neuron_numbers = [100, 100]
            current_input = z
            input_dim = self.z_dim
            for i, n in enumerate(neuron_numbers):
                w = tf.get_variable('W_disc_dens%d' % i, shape=[input_dim, n],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('b_disc_dens%d' % i, shape=[n],
                                    initializer=tf.constant_initializer())
                current_input = lrelu(tf.matmul(current_input, w) + b)
                input_dim = n

            w = tf.get_variable('W_disc_out', shape=[input_dim, 2],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b_disc_out', shape=[2],
                                initializer=tf.constant_initializer())
            y = tf.matmul(current_input, w) + b

        return y

    def sampler(self):
        z = tf.random_uniform([self.batch_size, self.z_dim], 0, 1, name='sampled_z')

        return z
