import tensorflow as tf
import numpy as np
from model_conv import ModelConv
from model_dense import ModelDense

from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc

# Prepare dataset
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
mean_img = np.mean(mnist.train.images, axis=0)

n_input = 28*28
batch_size = 100
z_dim = 1

# Input image
x_image = tf.placeholder(tf.float32, [batch_size, n_input], name='x_image')

# Labels for z describing the source (generated or encoded from real data)
y_merged = tf.placeholder(tf.float32, [batch_size*2,1], name='y_merged')
y_real = tf.placeholder(tf.float32, [batch_size,1], name='y_real')

model = ModelDense(input_shape=[batch_size, n_input], z_dim=z_dim)

z_sampled = model.sampler()
z_encoded = model.encoder(x_image=x_image)

z_merged = tf.concat(0, [z_sampled, z_encoded], name='z_merged')


# Reconstruction loss (Input: x_image )
x_reconstructed = model.decoder(z_encoded)
rec_loss = tf.reduce_mean(tf.square(x_reconstructed-x_image))
rec_optimizer = tf.train.MomentumOptimizer(learning_rate=0.05, momentum=0.9).minimize(rec_loss)


# Discriminator loss (Inputs: z_1, z_2, y_merged)
y_pred = model.discriminator(z_merged)
disc_loss = tf.nn.sigmoid_cross_entropy_with_logits(y_pred, y_merged)
disc_loss = tf.reduce_mean(disc_loss)

t_vars = tf.trainable_variables()
disc_vars = [var for var in t_vars if 'disc' in var.name]

disc_optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(disc_loss, var_list=disc_vars)


# Encoder loss (Input: x_image, y_real)
y_pred = model.discriminator(z_encoded, reuse=True)
enc_loss = tf.nn.sigmoid_cross_entropy_with_logits(y_pred, y_real)
enc_loss = tf.reduce_mean(enc_loss)

t_vars = tf.trainable_variables()
enc_vars = [var for var in t_vars if 'enc' in var.name]

enc_optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(enc_loss, var_list=enc_vars)

# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Saver
saver = tf.train.Saver()

# One batch used later for analyzing reconstruction during training
test_x, _ = mnist.test.next_batch(batch_size)
test_x_norm = np.array([img - mean_img for img in test_x])


# Training part
n_epochs = 200
for epoch in range(n_epochs):
    print("Epoch %d" % epoch)

    rec_loss_sum = 0
    disc_loss_sum = 0
    enc_loss_sum = 0

    steps = mnist.train.num_examples // batch_size

    # Sampled z are labeled with 0, encoded z are labeled with 1
    y_d = np.array([[0]] * batch_size + [[1]] * batch_size)
    # We want encoder to fool discriminator so we put label 0 for encoded values
    y_e = np.array([[0]] * batch_size)

    for batch in range(steps):
        batch_x, _ = mnist.train.next_batch(batch_size)
        train = np.array([img - mean_img for img in batch_x])

        # Reconstruction update
        rec_loss_for_batch, _ = sess.run([rec_loss, rec_optimizer], feed_dict={x_image: train})
        rec_loss_sum += rec_loss_for_batch

        # Discriminator update
        disc_loss_for_batch, _ = sess.run([disc_loss, disc_optimizer], feed_dict={x_image: train, y_merged: y_d})
        disc_loss_sum += disc_loss_for_batch

        # Encoder update
        enc_loss_for_batch, _ = sess.run([enc_loss, enc_optimizer], feed_dict={x_image: train, y_real: y_e})
        enc_loss_sum += enc_loss_for_batch

    print("Reconstruction Cost %f" % (rec_loss_sum/steps))
    print("Discrimination Cost %f" % (disc_loss_sum/steps))
    print("Encoder Cost %f" % (enc_loss_sum/steps))

    # recon = sess.run(x_reconstructed, feed_dict={x: test_x_norm})
    #
    # im = np.reshape(test_x[0], (28, 28))
    # rec = np.reshape(np.reshape(recon[0], (784,)) + mean_img, (28,28))
    # scipy.misc.imsave('output/train.jpg', im)
    # scipy.misc.imsave('output/rec%d.jpg' % epoch, rec)




