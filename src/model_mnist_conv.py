import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from src.mode_base import ModelBase


class ModelConvMnist(ModelBase):
    def __init__(self, batch_size, z_dim, y_dim=None, is_training=True):
        super().__init__(batch_size, z_dim, y_dim, is_training)
        self.input_dim = 784
        self.x_image = tf.placeholder(tf.float32, [batch_size, self.input_dim], name='x_image')

    def encoder(self):
        # Size of image SxS after i-th convolution (No need to specify for forward pass)
        # i_s = [28, 14, 7, 3, 1]
        # Number of features/channels in image after i-th convolution
        f_n = [1, 32, 64, 128, self.z_dim]
        # Padding algorithm for i-th convolution
        padding = ['', 'SAME', 'SAME', 'VALID', 'VALID']
        # Filter shape for i-th convolution (Not needed to specify for forward pass)
        # f_s = [_, 3, 3, 3, 3]

        x_image = tf.reshape(self.x_image, [-1, 28, 28, 1])
        current_input = x_image
        for i in range(1, len(f_n)):
            w = tf.get_variable('W_enc_conv%d' % i, shape=[3, 3, f_n[i-1], f_n[i]],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b_enc_conv%d' % i, shape=[f_n[i]],
                                initializer=tf.constant_initializer(0))
            current_input = tf.nn.conv2d(current_input, w, strides=[1, 2, 2, 1], padding=padding[i]) + b
            if i != len(f_n) - 1:
                current_input = batch_norm(current_input,
                                           scope=('batch_norm_dec_conv%d' % (i - 1)), **self.bn_settings)
                current_input = tf.maximum(0.2 * current_input, current_input)

        z = tf.reshape(current_input, shape=[self.batch_size, self.z_dim])
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

            # Image size after i-th deconvolution
            i_s = [1, 3, 5, 12, 26, 28]
            # Number of filters/channels in image after i-th deconvolution
            f_n = [input_dim, 128, 128, 64, 32, 1]
            # Filter shape for i-th convolution
            f_s = [None, 3, 3, 4, 4, 3]
            # Stride value for i-th convolution
            s_s = [None, 1, 1, 2, 2, 1]
            current_input = tf.reshape(current_input, [self.batch_size, 1, 1, input_dim])
            for i in range(1, len(f_n)):
                w = tf.get_variable('W_dec_tconv%d' % i, shape=[f_s[i], f_s[i], f_n[i], f_n[i-1]],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('b_dec_tconv%d' % i, shape=f_n[i],
                                    initializer=tf.constant_initializer(0))
                current_input = tf.nn.conv2d_transpose(current_input, w, [self.batch_size, i_s[i], i_s[i], f_n[i]],
                                                       strides=[1, s_s[i], s_s[i], 1], padding='VALID')
                current_input = current_input + b
                if i != len(f_n) - 1:
                    current_input = batch_norm(current_input,
                                               scope=('batch_norm_dec_conv%d' % (i-1)), **self.bn_settings)
                    current_input = tf.maximum(0.2 * current_input, current_input)

            current_input = tf.nn.sigmoid(current_input)
            y_image = tf.reshape(current_input, shape=[self.batch_size, self.input_dim])
            return y_image
