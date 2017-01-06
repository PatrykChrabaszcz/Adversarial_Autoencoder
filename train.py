from solver import *
import scipy.misc
from datasets import MNIST, CelebA
import numpy as np

# All ops are defined in solver and model classes
dataset = 'mnist'
#dataset = 'celeb'

if dataset == 'mnist':
    model = ModelConvMnist(batch_size=100, z_dim=4)
    data = MNIST()
    print('MNIST')
elif dataset == 'celeb':
    model = ModelConvCeleb(batch_size=100, z_dim=10)
    data = CelebA()
    print('CELEB')
else:
    raise NotImplementedError

# One batch used later for analyzing reconstruction during training
test_x = data.train_images[:100]

solver = Solver(model=model)

# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Saver
saver = tf.train.Saver()


# Training part
n_epochs = 200

# To restore previous
# saver.restore(sess, '/home/chrabasp/Workspace/MNIST_AA/model_70.ckpt')


for epoch in range(n_epochs):
    print("Epoch %d" % epoch)

    rec_loss_sum = 0
    disc_loss_sum = 0
    enc_loss_sum = 0
    steps = 0

    # Sampled z are labeled with 0, encoded z are labeled with 1
    y_d = np.array([[1, 0]] * model.batch_size + [[0, 1]] * model.batch_size)
    # We want encoder to fool discriminator so we put label 0 for encoded values
    y_e = np.array([[1, 0]] * model.batch_size)

    for batch_x, batch_y in data.iterate_minibatches(model.batch_size, shuffle=True):
        # Reconstruction update
        for i in range(1):
            rec_loss_for_batch, _ = sess.run([solver.rec_loss, solver.rec_optimizer],
                                             feed_dict={solver.x_image: batch_x})
        rec_loss_sum += rec_loss_for_batch

        # Encoder update
        for i in range(4):
            enc_loss_b, _ = sess.run([solver.enc_loss, solver.enc_optimizer],
                                     feed_dict={solver.x_image: batch_x, solver.y_real: y_e})

        enc_loss_sum += enc_loss_b

        disc_loss_b = sess.run([solver.disc_loss],
                               feed_dict={solver.x_image: batch_x, solver.y_merged: y_d})
        # TODO: Why this is a table
        disc_loss_b = disc_loss_b[0]

        if disc_loss_b > 0.4 and enc_loss_b < 0.8:
            sess.run([solver.disc_optimizer],
                     feed_dict={solver.x_image: batch_x, solver.y_merged: y_d})

        disc_loss_sum += disc_loss_b
        steps += 1

    print("Reconstruction Cost %f" % (rec_loss_sum/steps))

    print("Discrimination Cost %f" % (disc_loss_sum/steps))
    print("Encoder Cost %f" % (enc_loss_sum/steps))

    recon = sess.run(solver.x_reconstructed, feed_dict={solver.x_image: test_x})

    if dataset == 'mnist':
        im = np.reshape(test_x[0], (28, 28))
        rec = np.reshape(np.reshape(recon[0], (784,)), (28, 28))

    elif dataset == 'celeb':
        im = test_x[0]
        rec = recon[0]
    scipy.misc.imsave('output/train_%s2.jpg' % dataset, im)
    scipy.misc.imsave('output/rec%d_%s2.jpg' % (epoch, dataset), rec)

    if (epoch+1) % 10 == 0:
        saver.save(sess, "models/%s2_model_%d.ckpt" % (dataset, epoch+1))
