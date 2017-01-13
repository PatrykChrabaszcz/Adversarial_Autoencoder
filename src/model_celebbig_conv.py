import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from src.mode_base import ModelBase


# CODE IT
class ModelConvCelebBig(ModelBase):

    def __init__(self, batch_size, z_dim, y_dim=None, is_training=True):
        super().__init__(batch_size, z_dim, y_dim, is_training)
        self.x_image = tf.placeholder(tf.float32, [batch_size, 128, 128, 3], name='x_image')

    def encoder(self):
        # Image size in next layers
        # i_s = [128, 64, 32, 16, 8, 4, 1]
        # Channels in next layers
        f_n = [3, 32, 64, 128,  256, 512, self.z_dim]
        # Filter sizes (3x3xfn[i] for each i-th layer)
        f_s = [None, 3, 3, 3, 3, 3, 4]
        padding = [None, 'SAME', 'SAME', 'SAME', 'VALID']

        current_input = self.x_image
        for i in range(1, len(f_n)):
            w = tf.get_variable('W_enc_conv%d' % i, shape=[f_s[i], f_s[i], f_n[i-1], f_n[i]],
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
