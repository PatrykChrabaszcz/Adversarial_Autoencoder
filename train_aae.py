import tensorflow as tf

from src.model_dense_mnist import ModelDenseMnist
from src.model_conv_mnist import ModelConvMnist

from src.model_conv_32 import ModelConv32
from src.model_subpix_32 import ModelSubpix32

from src.model_dense_cell import ModelDenseCell

from src.model_subpix_128 import ModelSubpix128

from src.datasets import MNIST, CelebA, CelebBig, Cell

from src.aae_solver import AaeSolver
from src.aae_wgan_solver import AaeWGanSolver
from src.aae_gan_solver import AaeGanSolver

from src.utils import count_params
import time


# warm parameter is ignored in this function
# When restore is true training starts from last saved
# model. It was mostly used to adjust learning rates by hand
# during training
def train(solver, data, name, restore=False, warm=False):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()


    # To restore previous model
    if restore:
        print("Restoring")
        saver.restore(sess, 'models/model_%s.ckpt' % name)

    # Training part
    n_epochs = 25
    for epoch in range(n_epochs):
        start_time = time.time()

        print("Starting epoch %d" % epoch)
        loss_rec_sum = 0
        loss_disc_sum = 0
        loss_enc_sum = 0
        steps = 0

        l_e = 0.69
        l_d = 0.69
        for batch_x, batch_y in data.iterate_minibatches(model.batch_size, shuffle=True):

            ops = [solver.rec_loss, solver.rec_optimizer, solver.disc_loss, solver.enc_loss]

            # Discriminator/Encoder update (Trick to keep it in balance between them)
            # log(0.5) = 0.69 (Random guessing)
            if l_e < 0.95 or l_d > 0.45:
                ops.append(solver.disc_optimizer)
            if l_d < 0.95 or l_e > 0.45:
                ops.append(solver.enc_optimizer)

            # You can try to use for first epochs
            # rec_lr: 0.01
            # enc_lr: 0.00005
            # enc_lr: 0.00005
            # When reconstruction loss is low you can
            # change it to:
            # rec_lr: 0.00005
            # enc_lr: 0.00005
            # enc_lr: 0.00005
            # It should slowly converge z distribution
            # to prior distribution (Adjust parameters if necessary)

            res = sess.run(ops, feed_dict={solver.x_image: batch_x,
                                           solver.y_labels: batch_y,
                                           solver.rec_lr: 0.00005,
                                           solver.enc_lr: 0.00005,
                                           solver.disc_lr: 0.00005})
            l_r = res[0]
            l_d = res[2]
            l_e = res[3]

            loss_rec_sum += l_r
            loss_enc_sum += l_e
            loss_disc_sum += l_d

            if steps % 10 == 0:
                print("step %d, Current loss: Rec %.4f, Disc %.4f, Enc %.4f" %
                      (steps, l_r, l_d, l_e), end='\r')
            steps += 1

        s = ' ' * 20
        print('Epoch took %d seconds. %s' % (time.time()-start_time, s))

        print("Reconstruction Lost %f" % (loss_rec_sum/steps))
        print("Discrimination Lost %f" % (loss_disc_sum/steps))
        print("Encoder Lost %f \n" % (loss_enc_sum/steps))

        saver.save(sess, 'models/model_%s.ckpt' % name)
        print('Model saved as models/model_%s.ckpt' % name)


# If warm true it will try to load model pretrained with
# train(...) function. This should help

def train_gan(solver, data, name, restore=False, warm=False):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    if restore:
        if warm:
            print("Restore model without gan")
            t_vars = tf.trainable_variables()
            rec_vars = [var for var in t_vars if
                        'RMS' not in var.name and
                        'gan' not in var.name]
            saver = tf.train.Saver(rec_vars)
            saver.restore(sess, 'models/model_%s.ckpt' % name)
            # Reinit saver so it saves all variables
            saver = tf.train.Saver()
        else:
            print("Restore model with gan")
            saver.restore(sess, 'models/model_Gan_%s.ckpt' % name)

    # Training part
    n_epochs = 2000

    for epoch in range(n_epochs):
        start_time = time.time()

        print("Starting epoch %d" % epoch)
        loss_rec_sum = 0
        loss_disc_sum = 0
        loss_enc_sum = 0
        loss_gan_disc_sum = 0
        loss_gan_gen_sum = 0
        steps = 0

        l_e = 0.69
        l_d = 0.69
        l_g_d = 0.69
        l_g_g = 0.69

        # Pretrain gan discriminator
        if epoch == 0 and warm is True:
            for batch_x, batch_y in data.iterate_minibatches(model.batch_size, shuffle=True):
                if l_g_d > 0.30:
                    l_g_d, _ = sess.run([solver.gan_d_loss, solver.gan_d_optimizer],
                                        feed_dict={solver.x_image: batch_x,
                                        solver.y_labels: batch_y,
                                        solver.gan_d_lr: 0.0002})
                else:
                    break

        for batch_x, batch_y in data.iterate_minibatches(model.batch_size, shuffle=True):
            ops = [solver.rec_loss, solver.rec_optimizer, solver.disc_loss, solver.enc_loss,
                   solver.gan_d_loss, solver.gan_g_loss]

            # Discriminator/Encoder update (Trick to keep it in balance between them)
            # log(0.5) = 0.69 (Random guessing)
            if l_e < 0.95 or l_d > 0.45:
               ops.append(solver.disc_optimizer)
            if l_d < 0.95 or l_e > 0.45:
               ops.append(solver.enc_optimizer)

            # Gan Discriminate/Generate
            if l_g_g < 0.95 or l_g_d > 0.45:
                ops.append(solver.gan_d_optimizer)

            if l_g_d < 0.95 or l_g_g > 0.45:
                ops.append(solver.gan_g_optimizer)

            res = sess.run(ops, feed_dict={solver.x_image: batch_x,
                                           solver.y_labels: batch_y,
                                           solver.rec_lr: 0.0002,
                                           solver.disc_lr: 0.00005,
                                           solver.enc_lr: 0.00005,
                                           solver.gan_d_lr: 0.00001,
                                           solver.gan_g_lr: 0.00005})

            l_r = res[0]
            l_d = res[2]
            l_e = res[3]
            l_g_d = res[4]
            l_g_g = res[5]

            loss_rec_sum += l_r
            loss_enc_sum += l_e
            loss_disc_sum += l_d

            loss_gan_disc_sum += l_g_d
            loss_gan_gen_sum += l_g_g

            if steps % 10 == 0:

                print("S %d, R %.4f, D %.2f, E %.2f, Gc_D: %.2f, Gc_G: %.2f" %
                      (steps, l_r, l_d, l_e, l_g_d, l_g_g, ), end='\r')
            steps += 1

        print('Epoch took %d seconds.                                        ' % (time.time()-start_time))

        print("Reconstruction Lost %f" % (loss_rec_sum/steps))
        print("Discrimination Lost %f" % (loss_disc_sum/steps))
        print("Encoder Lost %f \n" % (loss_enc_sum/steps))
        print("GAN Discrimination Lost Ce %f" % (loss_gan_disc_sum/steps))
        print("GAN Generation Loss Ce %f \n" % (loss_gan_gen_sum/steps))

        saver.save(sess, 'models/model_%s.ckpt' % name)
        print('Model saved as models/model_%s.ckpt' % name)


# NOT TESTED !
def train_wgan(solver, data, name, restore=False):
    # Session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Saver
    t_vars = tf.trainable_variables()
    rec_vars = [var for var in t_vars if 'enc' in var.name or 'disc' in var.name or 'dec' in var.name]
    saver = tf.train.Saver(rec_vars)

    # Training part
    n_epochs = 2000

    # To restore previous
    if restore:
        print("Restoring")
        # saver.restore(sess, 'models/model_%s.ckpt' % name)
        saver.restore(sess, 'models/model_Mnist_Conv_y.ckpt')

    saver = tf.train.Saver()

    for epoch in range(n_epochs):
        start_time = time.time()

        print("Starting epoch %d" % epoch)
        loss_rec_sum = 0
        loss_disc_sum = 0
        loss_enc_sum = 0
        loss_gan_disc_sum = 0
        loss_gan_gen_sum = 0
        steps = 0

        l_e = 0.69
        l_d = 0.69

        for batch_x, batch_y in data.iterate_minibatches(model.batch_size, shuffle=True):

            ops = [solver.rec_loss, solver.rec_optimizer, solver.disc_loss, solver.enc_loss]

            # Discriminator/Encoder update (Trick to keep it in balance between them)
            # log(0.5) = 0.69 (Random guessing)
            if l_e < 0.95 or l_d > 0.45:
               ops.append(solver.disc_optimizer)
            if l_d < 0.95 or l_e > 0.45:
               ops.append(solver.enc_optimizer)

            # Gan Discriminate/Generate

            if epoch == 0:
                it = 25
            else:
                it = 2

            for _ in range(it):
                l_g_d, _ = sess.run([solver.gan_d_loss, solver.gan_d_optimizer],
                                    feed_dict={solver.x_image: batch_x,
                                               solver.y_labels: batch_y,
                                               solver.gan_d_lr: 0.0002})
                sess.run(solver.clip_gan_d)

            l_g_g, _ = sess.run([solver.gan_g_loss, solver.gan_g_optimizer],
                                feed_dict={solver.x_image: batch_x,
                                           solver.y_labels: batch_y,
                                           solver.gan_g_lr: 0.0002})

            res = sess.run(ops, feed_dict={solver.x_image: batch_x,
                                           solver.y_labels: batch_y,
                                           solver.rec_lr: 0.00002,
                                           solver.disc_lr: 0.00002,
                                           solver.enc_lr: 0.00002})

            l_r = res[0]
            l_d = res[2]
            l_e = res[3]

            loss_rec_sum += l_r
            loss_enc_sum += l_e
            loss_disc_sum += l_d

            loss_gan_disc_sum += l_g_d
            loss_gan_gen_sum += l_g_g

            if steps % 10 == 0:

                print("S %d, R %.4f, D %.2f, E %.2f, Gc_D: %.2f, Gc_G: %.2f" %
                      (steps, l_r, l_d, l_e, l_g_d, l_g_g, ), end='\r')
            steps += 1

        print('Epoch took %d seconds.                                        ' % (time.time()-start_time))

        print("Reconstruction Lost %f" % (loss_rec_sum/steps))
        print("Discrimination Lost %f" % (loss_disc_sum/steps))
        print("Encoder Lost %f \n" % (loss_enc_sum/steps))
        print("GAN Discrimination Lost Ce %f" % (loss_gan_disc_sum/steps))
        print("GAN Generation Loss Ce %f \n" % (loss_gan_gen_sum/steps))

        saver.save(sess, 'models/model_%s.ckpt' % name)
        print('Model saved as models/model_%s.ckpt' % name)


if __name__ == '__main__':
    scenario = 2
    mnist_z_dim = 5
    celeb_z_dim = 50
    cell_z_dim = 25
    celebbig_z_dim = 128

    gan = ''
    if gan == 'Gan':
        train_func = train_gan
        solver_class = AaeGanSolver
    elif gan == 'WGan':
        train_func = train_wgan
        solver_class = AaeWGanSolver
    else:
        train_func = train
        solver_class = AaeSolver

    # MNIST++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Mnist dense with y labels
    if scenario == 1:
        y_dim = 10
        model = ModelDenseMnist(batch_size=128, z_dim=mnist_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = MNIST()
        print('Training Mnist dense with y labels')
        train_func(solver, data, name='Mnist_Dense_y', restore=False)

    # Mnist dense without y labels
    elif scenario == 2:
        y_dim = None
        model = ModelDenseMnist(batch_size=128, z_dim=mnist_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = MNIST()
        print('Training Mnist dense without y labels')
        train_func(solver, data, name='Mnist_Dense_noy', restore=True)

    # Mnist conv with y labels
    if scenario == 3:
        y_dim = 10
        model = ModelConvMnist(batch_size=128, z_dim=mnist_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = MNIST()
        print('Training Mnist conv with y labels')
        train_func(solver, data, name='Mnist_Conv_y', restore=False, warm=False)

    # Mnist conv without y labels
    elif scenario == 4:
        y_dim = None
        model = ModelConvMnist(batch_size=128, z_dim=mnist_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = MNIST()
        print('Training Mnist conv without y labels')
        train_func(solver, data, name='Mnist_Conv_noy', restore=True, warm=True)

    # CELEB++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Celeb convolution with y labels
    elif scenario == 5:
        y_dim = 40
        model = ModelConv32(batch_size=128, z_dim=celeb_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = CelebA()
        print('Training Celeb conv with y labels')
        train_func(solver, data, name='Celeb_Conv_y', restore=True)

    # Celeb convolution without y labels
    elif scenario == 6:
        y_dim = None
        model = ModelConv32(batch_size=128, z_dim=celeb_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = CelebA()
        print('Training Celeb conv without y labels')
        train_func(solver, data, name='Celeb_Conv_noy', restore=True)

    # Celeb subpix with y labels
    elif scenario == 7:
        y_dim = 40
        model = ModelSubpix32(batch_size=128, z_dim=celeb_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = CelebA()
        print('Training Celeb Subpix with y labels')
        train_func(solver, data, name='Celeb_Subpix_y', restore=False)

    # Celeb subpix without y labels
    elif scenario == 8:
        y_dim = None
        model = ModelSubpix32(batch_size=128, z_dim=celeb_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = CelebA()
        print('Training Celeb %sSubpix without y labels')
        train_func(solver, data, name='Celeb_%sSubpix', restore=True)

    # CELEB_BIG++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # CelebBig Subpix with y labels
    elif scenario == 7:
        y_dim = 40
        model = ModelSubpix128(batch_size=32, z_dim=celebbig_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = CelebBig()
        print('Training Subpix128 without y labels')
        train_func(solver, data, name='CelebBig_Subpix_y', restore=False)

    # CelebBig Subpix without y labels
    elif scenario == 8:
        y_dim = None
        model = ModelSubpix128(batch_size=64, z_dim=celebbig_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = CelebBig()
        print('Training %sSubpix128 without y labels')
        train_func(solver, data, name='%sCelebBig_Subpix_noy', restore=False)

    # CELL +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    elif scenario == 12:
        y_dim = None
        model = ModelDenseCell(batch_size=128, z_dim=cell_z_dim, y_dim=y_dim)
        solver = solver_class(model=model)
        print("Number of parameters in model %d" % count_params())
        data = Cell()
        print('Training Cell Dense without y labels')
        train_func(solver, data, name='Cell_Dense_noy', restore=False)
