import time
import tensorflow as tf

from src.model_conv_mnist import ModelConvMnist
from src.model_dense_mnist import ModelDenseMnist

from src.model_conv_32 import ModelConv32
from src.model_res_32 import ModelRes32
from src.model_subpix_32 import ModelSubpix32

from src.model_res_128 import ModelRes128
from src.model_sp_128 import ModelSubpixel128

from src.datasets import MNIST, CelebA, CelebBig
from src.aae_solver import AaeSolver


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


if __name__ == '__main__':
    scenario = 10
    mnist_z_dim = 5
    celeb_z_dim = 50
    celebbig_z_dim = 128

    # MNIST++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Mnist dense with y labels
    if scenario == 1:
        y_dim = 10
        model = ModelDenseMnist(batch_size=128, z_dim=mnist_z_dim, y_dim=y_dim)
        solver = AaeSolver(model=model)
        data = MNIST()
        train(solver, data, name='Mnist_Dense_y', restore=False)

    # Mnist dense without y labels
    elif scenario == 2:
        y_dim = None
        model = ModelDenseMnist(batch_size=128, z_dim=mnist_z_dim, y_dim=y_dim)
        solver = AaeSolver(model=model)
        data = MNIST()
        train(solver, data, name='Mnist_Dense_noy', restore=False)

    # Mnist convolution with y labels
    elif scenario == 3:
        y_dim = 10
        model = ModelConvMnist(batch_size=128, z_dim=mnist_z_dim, y_dim=y_dim)
        solver = AaeSolver(model=model)
        data = MNIST()
        train(solver, data, name='Mnist_Conv_y', restore=False)

    # Mnist convolution without y labels
    elif scenario == 4:
        y_dim = None
        model = ModelConvMnist(batch_size=128, z_dim=mnist_z_dim, y_dim=y_dim)
        solver = AaeSolver(model=model)
        data = MNIST()
        train(solver, data, name='Mnist_Conv_noy', restore=False)

    # CELEB++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Celeb convolution with y labels
    elif scenario == 5:
        y_dim = 40
        model = ModelConv32(batch_size=128, z_dim=celeb_z_dim, y_dim=y_dim)
        data = CelebA()
        solver = AaeSolver(model=model)
        train(solver, data, name='Celeb_Conv_y')

    # Celeb convolution without y labels
    elif scenario == 6:
        y_dim = None
        model = ModelConv32(batch_size=128, z_dim=celeb_z_dim, y_dim=y_dim)
        solver = AaeSolver(model=model)
        data = CelebA()
        train(solver, data, name='Celeb_Conv_noy')

    # Celeb resnet with y labels
    elif scenario == 7:
        y_dim = 40
        model = ModelRes32(batch_size=128, z_dim=celeb_z_dim, y_dim=y_dim)
        solver = AaeSolver(model=model)
        data = CelebA()
        train(solver, data, name='Celeb_Res_y', restore=False)

    # Celeb resnet without y labels
    elif scenario == 8:
        y_dim = None
        model = ModelRes32(batch_size=128, z_dim=celeb_z_dim, y_dim=y_dim)
        solver = AaeSolver(model=model)
        data = CelebA()
        train(solver, data, name='Celeb_Res_noy', restore=False)

    # Celeb subpix with y labels
    elif scenario == 9:
        y_dim = 40
        model = ModelSubpix32(batch_size=128, z_dim=celeb_z_dim, y_dim=y_dim)
        solver = AaeSolver(model=model)
        data = CelebA()
        train(solver, data, name='Celeb_Subpix_y', restore=False)

    # Celeb subpix without y labels
    elif scenario == 10:
        y_dim = None
        model = ModelSubpix32(batch_size=128, z_dim=celeb_z_dim, y_dim=y_dim)
        solver = AaeSolver(model=model)
        data = CelebA()
        train(solver, data, name='Celeb_Subpix_noy', restore=False)

    # CELEB_BIG++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # # CelebBig resnet with y labels
    # elif scenario == 9:
    #     y_dim = 40
    #     model = ModelRes128(batch_size=128, z_dim=celeb_z_dim, y_dim=y_dim)
    #     solver = AaeSolver(model=model)
    #     data = CelebBig(mean=True)
    #     train(solver, data, name='Celeb_Res_y', restore=True)

    # CelebBig resnet without y labels
    elif scenario == 12:
        y_dim = None
        model = ModelRes128(batch_size=128, z_dim=celebbig_z_dim, y_dim=y_dim)
        solver = AaeSolver(model=model)
        data = CelebBig()
        train(solver, data, name='CelebBig_Res_noy', restore=False)
