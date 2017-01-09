import tensorflow as tf
from model_conv_mnist import ModelConvMnist
from model_dense_mnist import ModelDenseMnist
from model_conv_celeb import ModelConvCeleb


class Solver:
    def __init__(self,  model):

        batch_size = model.batch_size

        self.x_image = model.x_image

        # Labels for predicting the source of latent variables 'z' (generated or encoded from real data)
        self.y_merged = tf.placeholder(tf.float32, [batch_size*2, 2], name='y_merged')
        self.y_real = tf.placeholder(tf.float32, [batch_size, 2], name='y_real')

        self.z_sampled = model.sampler()
        self.z_encoded = model.encoder()

        z_merged = tf.concat(0, [self.z_sampled, self.z_encoded], name='z_merged')

        # Reconstruction loss (Input: x_image )
        self.x_reconstructed = model.decoder(self.z_encoded)
        self.rec_loss = tf.reduce_mean(tf.square(self.x_reconstructed-self.x_image))

        t_vars = tf.trainable_variables()
        rec_vars = [var for var in t_vars if 'dec' or 'enc' in var.name]

        self.rec_optimizer = tf.train.AdamOptimizer(learning_rate=0.02).\
            minimize(self.rec_loss, var_list=rec_vars)


        # Discriminator loss (Inputs: z_1, z_2, y_merged)
        y_pred_d = model.discriminator(z_merged)
        disc_loss = tf.nn.softmax_cross_entropy_with_logits(y_pred_d, self.y_merged)
        self.disc_loss = tf.reduce_mean(disc_loss)

        #disc_correct_prediction = tf.equal(tf.argmax(y_pred_d, 1), tf.argmax(self.y_merged, 1))
        #self.disc_accuracy = tf.reduce_mean(tf.cast(disc_correct_prediction, tf.float32))

        t_vars = tf.trainable_variables()
        disc_vars = [var for var in t_vars if 'disc' in var.name]
        self.disc_optimizer = tf.train.AdamOptimizer(learning_rate=0.1).\
            minimize(disc_loss, var_list=disc_vars)

        # Encoder loss (Input: x_image, y_real)
        y_pred_e = model.discriminator(self.z_encoded, reuse=True)
        enc_loss = tf.nn.softmax_cross_entropy_with_logits(y_pred_e, self.y_real)
        self.enc_loss = tf.reduce_mean(enc_loss)

        #enc_correct_prediction = tf.equal(tf.argmax(y_pred_e, 1), tf.argmax(self.y_real, 1))
        #self.enc_accuracy = tf.reduce_mean(tf.cast(enc_correct_prediction, tf.float32))

        t_vars = tf.trainable_variables()
        enc_vars = [var for var in t_vars if 'enc' in var.name]
        self.enc_optimizer = tf.train.AdamOptimizer(learning_rate=0.1).\
            minimize(enc_loss, var_list=enc_vars)
