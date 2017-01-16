from src.aae_solver import AaeSolver

import tensorflow as tf


class AaeGanSolver(AaeSolver):
    def __init__(self, model):
        super().__init__(model)

        # Learning rates for different parts of training
        self.gan_d_lr = tf.placeholder(tf.float32, shape=[])
        self.gan_g_lr = tf.placeholder(tf.float32, shape=[])

        # Gan Discriminator
        gan_d_input = tf.concat(0, [self.x_image, self.x_reconstructed])
        gan_d_pred = model.gan(gan_d_input, reuse=False, discriminator=True)

        gan_d_target = tf.concat(0, [tf.ones([model.batch_size, 1]), tf.zeros([model.batch_size, 1])])

        gan_d_loss = gan_d_target*tf.square(gan_d_pred - 1) + (1 - gan_d_target)*tf.square(gan_d_pred)
        self.gan_d_loss = tf.reduce_mean(gan_d_loss)

        t_vars = tf.trainable_variables()
        rec_vars = [var for var in t_vars if 'gan' in var.name]
        self.gan_d_optimizer = tf.train.AdamOptimizer(learning_rate=self.gan_d_lr, beta1=0.5). \
            minimize(self.gan_d_loss, var_list=rec_vars)

        # Gan Generator
        gan_g_pred = model.gan(self.x_reconstructed, reuse=True, discriminator=False)

        gan_g_loss = tf.square(gan_g_pred - 1)
        self.gan_g_loss = tf.reduce_mean(gan_g_loss)

        t_vars = tf.trainable_variables()
        rec_vars = [var for var in t_vars if 'dec' or 'enc' in var.name]

        self.gan_g_optimizer = tf.train.AdamOptimizer(learning_rate=self.gan_g_lr, beta1=0.5). \
            minimize(self.gan_g_loss, var_list=rec_vars)
