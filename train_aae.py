import time
import tensorflow as tf

from src.model_dense_mnist import ModelDenseMnist

from src.model_conv_32 import ModelConv32
from src.model_subpix_32 import ModelSubpix32
from src.model_subpix_res32 import ModelSubpixRes32

from src.model_dense_cell import ModelDenseCell

from src.model_subpix_128 import ModelSubpix128

from src.datasets import MNIST, CelebA, CelebBig, Cell

from src.aae_solver import AaeSolver
from src.aae_gan_solver import AaeGanSolver

from src.utils import count_params


def train(solver, data, name, restore=False):
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
                              feed_dict={solver.x_image: batch_x, solver.y_labels: batch_y, solver.rec_lr: 0.0002})
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
                         feed_dict={solver.x_image: batch_x, solver.y_labels: batch_y, solver.disc_lr: 0.0002})
            # Encoder update
            if l_d < 0.95 or l_e > 0.45:
                sess.run(solver.enc_optimizer,
                         feed_dict={solver.x_image: batch_x, solver.y_labels: batch_y, solver.enc_lr: 0.0002})
            if steps % 10 == 0:
                print("step %d, Current loss: Rec %.4f, Disc %.4f, Enc %.4f" % (steps, l_r, l_d, l_e), end='\r')
            steps += 1

        print('Epoch took %d seconds.                                        ' % (time.time()-start_time))

        print("Reconstruction Lost %f" % (loss_rec_sum/steps))
        print("Discrimination Lost %f" % (loss_disc_sum/steps))
        print("Encoder Lost %f \n" % (loss_enc_sum/steps))

        saver.save(sess, 'models/model_%s.ckpt' % name)
        print('Model saved as models/model_%s.ckpt' % name)


def train_gan(solver, data, name, restore=False):
    # Session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # Saver
    #t_vars = tf.trainable_variables()
    #rec_vars = [var for var in t_vars if 'enc' in var.name or 'disc' in var.name or 'dec' in var.name]
    #saver = tf.train.Saver(rec_vars)
    saver = tf.train.Saver()

    # Training part
    n_epochs = 2000

    # To restore previous
    if restore:
        saver.restore(sess, 'models/model_%s.ckpt' % name)
        #saver.restore(sess, 'models/model_Mnist_Dense_noy.ckpt')

    for epoch in range(n_epochs):
        start_time = time.time()

        print("Starting epoch %d" % epoch)
        loss_rec_sum = 0
        loss_disc_sum = 0
        loss_enc_sum = 0
        loss_gan_disc_sq_sum = 0
        loss_gan_gen_sq_sum = 0
        loss_gan_disc_ce_sum = 0
        loss_gan_gen_ce_sum = 0
        steps = 0

        for batch_x, batch_y in data.iterate_minibatches(model.batch_size, shuffle=True):

            l_e, l_d = sess.run([solver.enc_loss, solver.disc_loss],
                               feed_dict={solver.x_image: batch_x, solver.y_labels: batch_y})
            loss_enc_sum += l_e
            loss_disc_sum += l_d

            # Discriminator update (Trick to keep it in balance with encoder
            # log(0.5) = 0.69 (Random guessing)
            if l_e < 0.95 or l_d > 0.45:
               sess.run([solver.disc_loss, solver.disc_optimizer],
                        feed_dict={solver.x_image: batch_x, solver.y_labels: batch_y, solver.disc_lr: 0.0002})
            # Encoder update
            if l_d < 0.95 or l_e > 0.45:
               sess.run(solver.enc_optimizer,
                        feed_dict={solver.x_image: batch_x, solver.y_labels: batch_y, solver.enc_lr: 0.0002})

            # GAN update

            #l_g_sq_d, l_g_sq_g = sess.run([solver.gan_d_sq_loss, solver.gan_g_sq_loss],
            #                               feed_dict={solver.x_image: batch_x})
            l_g_ce_d, l_g_ce_g = sess.run([solver.gan_d_loss, solver.gan_g_loss],
                                          feed_dict={solver.x_image: batch_x})

            #loss_gan_disc_sq_sum += l_g_sq_d
            #loss_gan_gen_sq_sum += l_g_sq_g
            loss_gan_disc_ce_sum += l_g_ce_d
            loss_gan_gen_ce_sum += l_g_ce_g
            if l_g_ce_g < 0.95 or l_g_ce_d > 0.55:
                sess.run(solver.gan_d_optimizer, feed_dict={solver.x_image: batch_x, solver.gan_d_lr: 0.00002})

            if l_g_ce_d < 0.95 or l_g_ce_g > 0.55:
                sess.run(solver.gan_g_optimizer, feed_dict={solver.x_image: batch_x, solver.gan_g_lr: 0.0002})

            # Reconstruction update

            l_r, _ = sess.run([solver.rec_loss, solver.rec_optimizer],
                              feed_dict={solver.x_image: batch_x, solver.y_labels: batch_y, solver.rec_lr: 0.0002})
            l_r /= model.batch_size
            loss_rec_sum += l_r


            if steps % 10 == 0:
                print("S %d, R %.4f, D %.2f, E %.2f, Gc_D: %.2f, Gc_G: %.2f" %
                      (steps, l_r, l_d, l_e, l_g_ce_d, l_g_ce_g, ), end='\r')
            steps += 1

        print('Epoch took %d seconds.                                        ' % (time.time()-start_time))

        print("Reconstruction Lost %f" % (loss_rec_sum/steps))
        print("Discrimination Lost %f" % (loss_disc_sum/steps))
        print("Encoder Lost %f \n" % (loss_enc_sum/steps))
        #print("GAN Discrimination Lost Sq %f" % (loss_gan_disc_sq_sum/steps))
        #print("GAN Generation Loss Sq %f \n" % (loss_gan_gen_sq_sum/steps))
        print("GAN Discrimination Lost Ce %f" % (loss_gan_disc_ce_sum/steps))
        print("GAN Generation Loss Ce %f \n" % (loss_gan_gen_ce_sum/steps))

        saver.save(sess, 'models/model_%s.ckpt' % name)


if __name__ == '__main__':
    scenario = 4
    mnist_z_dim = 5
    celeb_z_dim = 50
    cell_z_dim = 25
    celebbig_z_dim = 128

    gan = False
    if gan:
        train_func = train_gan
        solver_class = AaeGanSolver
        name = 'Gan'
    else:
        train_func = train
        solver_class = AaeSolver
        name = ''

    # MNIST++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Mnist dense with y labels
    if scenario == 1:
        y_dim = 10
        model = ModelDenseMnist(batch_size=128, z_dim=mnist_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = MNIST()
        print('Training Mnist dense with y labels')
        train_func(solver, data, name='%sMnist_Dense_y' % name, restore=False)

    # Mnist dense without y labels
    elif scenario == 2:
        y_dim = None
        model = ModelDenseMnist(batch_size=128, z_dim=mnist_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = MNIST()
        print('Training %s Mnist dense without y labels' % name)
        train_func(solver, data, name='%s_Mnist_Dense_noy_balanced' % name, restore=True)

    # CELEB++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Celeb convolution with y labels
    elif scenario == 3:
        y_dim = 40
        model = ModelConv32(batch_size=128, z_dim=celeb_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = CelebA()
        print('Training Celeb conv with y labels')
        train_func(solver, data, name='Celeb_Conv_y', restore=True)

    # Celeb convolution without y labels
    elif scenario == 4:
        y_dim = None
        model = ModelConv32(batch_size=128, z_dim=celeb_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = CelebA()
        print('Training Celeb conv without y labels')
        train_func(solver, data, name='Celeb_Conv_noy', restore=True)

    # Celeb subpix with y labels
    elif scenario == 5:
        y_dim = 40
        model = ModelSubpix32(batch_size=128, z_dim=celeb_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = CelebA()
        print('Training Celeb Subpix with y labels')
        train_func(solver, data, name='Celeb_Subpix_y', restore=False)

    # Celeb subpix without y labels
    elif scenario == 6:
        y_dim = None
        model = ModelSubpix32(batch_size=128, z_dim=celeb_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = CelebA()
        print('Training Celeb Subpix without y labels')
        train_func(solver, data, name='Celeb_Subpix_noy', restore=True)

    # Celeb subpix resnet with y labels
    elif scenario == 7:
        y_dim = 40
        model = ModelSubpixRes32(batch_size=128, z_dim=celeb_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = CelebA()
        print('Training Celeb Subpix Resnet with y labels')
        train_func(solver, data, name='Celeb_SubpixRes_y', restore=False)

    # Celeb subpix resnet without y labels
    elif scenario == 8:
        y_dim = None
        model = ModelSubpixRes32(batch_size=128, z_dim=celeb_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = CelebA()
        print('Training Celeb Subpix Resnet without y labels')
        train_func(solver, data, name='Celeb_SubpixRes_noy', restore=False)

    # CELEB_BIG++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # CelebBig Subpix with y labels
    elif scenario == 9:
        y_dim = 40
        model = ModelSubpix128(batch_size=32, z_dim=celebbig_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = CelebBig()
        print('Training Subpix128 without y labels')
        train_func(solver, data, name='CelebBig_Subpix_y', restore=False)

    # CelebBig Subpix without y labels
    elif scenario == 10:
        y_dim = None
        model = ModelSubpix128(batch_size=32, z_dim=celebbig_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = CelebBig()
        print('Training Subpix128 without y labels')
        train_func(solver, data, name='CelebBig_Subpix_noy', restore=False)

    # CELL +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    elif scenario == 12:
        y_dim = None
        model = ModelDenseCell(batch_size=128, z_dim=cell_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = Cell()
        print('Training Cell Dense without y labels')
        train_func(solver, data, name='Cell_Dense_noy', restore=False)
