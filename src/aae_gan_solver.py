import tensorflow as tf
from tensorflow.python.ops.nn import sigmoid_cross_entropy_with_logits as ce_loss


class AaeGanSolver:
    def __init__(self, model):
        # Tensor with images provided by the user
        self.x_image = model.x_image
        self.y_labels = model.y_labels

        # Two sources of latent variables (encoded and sampled from selected distribution)
        self.z_sampled = model.sampler()
        self.z_encoded = model.encoder()

        # Getting images from latent variables provided by the user
        self.z_provided = tf.placeholder(tf.float32, shape=[model.batch_size, model.z_dim])
        self.x_from_z = model.decoder(self.z_provided)

        # Learning rates for different parts of training
        self.rec_lr = tf.placeholder(tf.float32, shape=[])
        self.disc_lr = tf.placeholder(tf.float32, shape=[])
        self.enc_lr = tf.placeholder(tf.float32, shape=[])

        # Discriminator
        self.y_pred_sam = model.discriminator(self.z_sampled)
        self.y_pred_enc = model.discriminator(self.z_encoded, reuse=True)

        disc_loss_sam = ce_loss(self.y_pred_sam, tf.ones_like(self.y_pred_sam))
        disc_loss_enc = ce_loss(self.y_pred_enc, tf.zeros_like(self.y_pred_enc))
        disc_loss = tf.reduce_mean(disc_loss_sam) + tf.reduce_mean(disc_loss_enc)
        self.disc_loss = disc_loss / 2.0

        t_vars = tf.trainable_variables()
        disc_vars = [var for var in t_vars if 'disc' in var.name]
        self.disc_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.disc_lr).\
            minimize(self.disc_loss, var_list=disc_vars)

        # Encoder
        enc_loss = ce_loss(self.y_pred_enc, tf.ones_like(self.y_pred_enc))
        self.enc_loss = tf.reduce_mean(enc_loss)

        t_vars = tf.trainable_variables()
        enc_vars = [var for var in t_vars if 'enc' in var.name]

        self.enc_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.enc_lr).\
            minimize(self.enc_loss, var_list=enc_vars)

        # Learning rates for different parts of training
        self.gan_d_lr = tf.placeholder(tf.float32, shape=[])
        self.gan_g_lr = tf.placeholder(tf.float32, shape=[])

        self.x_reconstructed = model.decoder(self.z_encoded, reuse=True)
        self.x_sampled = model.decoder(self.z_sampled, reuse=True)

        # Gan Discriminator
        # For fake images use reconstructed images and images
        # generated from samples (This should perform better)
        gan_real_pred = model.gan(self.x_image, reuse=False)
        gan_rec_pred = model.gan(self.x_reconstructed, reuse=True)
        gan_sam_pred = model.gan(self.x_sampled, reuse=True)

        # Reformulate loss in terms of L2
        # gan_d_loss_real = tf.square(1-gan_real_pred)
        # gan_d_loss_rec = tf.square(gan_rec_pred)
        # gan_d_loss_sam = tf.square(gan_sam_pred)

        # "Normal" loss
        gan_d_loss_real = ce_loss(gan_real_pred, tf.ones_like(gan_real_pred))
        gan_d_loss_rec = ce_loss(gan_rec_pred, tf.zeros_like(gan_rec_pred))
        gan_d_loss_sam = ce_loss(gan_sam_pred, tf.zeros_like(gan_sam_pred))

        gan_d_loss = 2*tf.reduce_mean(gan_d_loss_real) + tf.reduce_mean(gan_d_loss_rec)\
                     + tf.reduce_mean(gan_d_loss_sam)
        self.gan_d_loss = gan_d_loss / 4.0

        t_vars = tf.trainable_variables()
        gan_d_vars = [var for var in t_vars if 'gan' in var.name]
        self.gan_d_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.gan_d_lr). \
            minimize(self.gan_d_loss, var_list=gan_d_vars)

        # Gan Generator

        # Reformulate loss in terms of L2
        # gan_g_loss_rec = tf.square(1-gan_rec_pred)
        # gan_g_loss_sam = tf.square(1-gan_sam_pred)

        # "Normal" loss
        gan_g_loss_rec = ce_loss(gan_rec_pred, tf.ones_like(gan_rec_pred))
        gan_g_loss_sam = ce_loss(gan_sam_pred, tf.ones_like(gan_sam_pred))
        gan_g_loss = tf.reduce_mean(gan_g_loss_rec) + tf.reduce_mean(gan_g_loss_sam)
        self.gan_g_loss = gan_g_loss / 2.0

        t_vars = tf.trainable_variables()
        gan_g_vars = [var for var in t_vars if 'dec' in var.name]
        self.gan_g_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.gan_g_lr). \
            minimize(self.gan_g_loss, var_list=gan_g_vars)

        # Reconstruction
        self.features = model.gan(self.x_image, reuse=True, features=True)
        self.features_reconstructed = model.gan(self.x_reconstructed, reuse=True, features=True)

        self.rec_loss = 0
        for i in range(len(self.features)):
            self.rec_loss += tf.reduce_mean(tf.square(self.features[i] - self.features_reconstructed[i]))

        t_vars = tf.trainable_variables()
        rec_vars = [var for var in t_vars if 'dec' in var.name or 'enc' in var.name]

        self.rec_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.rec_lr).\
            minimize(self.rec_loss, var_list=rec_vars)
