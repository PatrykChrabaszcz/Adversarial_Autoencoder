import time
import tensorflow as tf
import numpy as np

from src.model_mnist_conv import ModelConvMnist
from src.model_mnist_dense import ModelDenseMnist
from src.model_mnist_hq import ModelHqMnist
from src.model_celeb_conv import ModelConvCeleb

from src.datasets import MNIST, CelebA
from src.solver import Solver


def train(model, data, name, restore=False):
    # Solver
    solver = Solver(model=model)
    # Session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # Saver
    saver = tf.train.Saver()

    # Training part
    n_epochs = 2000

    # To restore previous
    if restore:
        saver.restore(sess, 'models/model_%s.ckpt' % name)

    for epoch in range(n_epochs):
        start_time = time.time()

        print("Starting epoch %d" % epoch)
        loss_rec_sum = 0
        loss_disc_sum = 0
        loss_enc_sum = 0
        steps = 0

        for batch_x, batch_y in data.iterate_minibatches(model.batch_size, shuffle=True):
            # Reconstruction update
            l_r, _ = sess.run([solver.rec_loss, solver.rec_optimizer],
                              feed_dict={solver.x_image: batch_x, solver.y_labels: batch_y, solver.rec_lr: 0.00002})
            l_r /= model.batch_size
            loss_rec_sum += l_r

            l_e, l_d = sess.run([solver.enc_loss, solver.disc_loss],
                                feed_dict={solver.x_image: batch_x, solver.y_labels: batch_y})
            loss_enc_sum += l_e
            loss_disc_sum += l_d

            # Discriminator update (Trick to keep it in balance with encoder
            # log(0.5) = 0.69 (Random guessing)
            if l_e < 0.95 or l_d > 0.45:
                sess.run([solver.disc_loss, solver.disc_optimizer],
                         feed_dict={solver.x_image: batch_x, solver.y_labels: batch_y, solver.disc_lr: 0.00002})
            # Encoder update
            if l_d < 0.95 or l_e > 0.45:
                sess.run(solver.enc_optimizer,
                         feed_dict={solver.x_image: batch_x, solver.y_labels: batch_y, solver.enc_lr: 0.00002})
            if steps % 10 == 0:
                print("step %d, Current loss: Rec %.4f, Disc %.4f, Enc %.4f" % (steps, l_r, l_d, l_e), end='\r')
            steps += 1

        print('Epoch took %d seconds.                                        ' % (time.time()-start_time))

        print("Reconstruction Lost %f" % (loss_rec_sum/steps))
        print("Discrimination Lost %f" % (loss_disc_sum/steps))
        print("Encoder Lost %f \n" % (loss_enc_sum/steps))

        saver.save(sess, 'models/model_%s.ckpt' % name)


if __name__ == '__main__':
    scenario = 4
    y_dim = 40
    if scenario == 1:
        model = ModelDenseMnist(batch_size=128, z_dim=5, y_dim=y_dim)
        data = MNIST(mean=False)
        train(model, data, name='Mnist_Dense_Adam_noy', restore=False)
    if scenario == 2:
        model = ModelConvMnist(batch_size=128, z_dim=5, y_dim=y_dim)
        data = MNIST(mean=False)
        train(model, data, name='Mnist_Conv_Adam_noy', restore=False)
    if scenario == 3:
        model = ModelHqMnist(batch_size=1, z_dim=5, y_dim=y_dim)
        data = MNIST(mean=False)
        train(model, data, name='Mnist_Hq')
    if scenario == 4:
        model = ModelConvCeleb(batch_size=128, z_dim=25, y_dim=y_dim)
        data = CelebA(mean=False)
        train(model, data, name='Celeb_Conv_Adam_sigmoid_25')
