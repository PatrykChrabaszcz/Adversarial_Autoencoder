import time
import tensorflow as tf

from src.model_mnist_conv import ModelConvMnist
from src.model_mnist_dense import ModelDenseMnist
from src.model_mnist_hq import ModelHqMnist
from src.model_celeb_conv import ModelConvCeleb
from src.model_celeb_res import ModelResCeleb
from src.model_celeb_subpixel import ModelSubpixelCeleb

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
    scenario = 7

    # Mnist dense with y labels
    if scenario == 1:
        y_dim = 10
        model = ModelDenseMnist(batch_size=128, z_dim=5, y_dim=y_dim)
        data = MNIST(mean=False)
        train(model, data, name='Mnist_Dense_y', restore=False)

    # Mnist dense without y labels
    elif scenario == 2:
        y_dim = None
        model = ModelDenseMnist(batch_size=128, z_dim=5, y_dim=y_dim)
        data = MNIST(mean=False)
        train(model, data, name='Mnist_Dense_noy', restore=False)

    # Mnist convolution with y labels
    elif scenario == 3:
        y_dim = 10
        model = ModelConvMnist(batch_size=128, z_dim=5, y_dim=y_dim)
        data = MNIST(mean=False)
        train(model, data, name='Mnist_Conv_y', restore=False)

    # Mnist convolution without y labels
    elif scenario == 4:
        y_dim = None
        model = ModelConvMnist(batch_size=128, z_dim=5, y_dim=y_dim)
        data = MNIST(mean=False)
        train(model, data, name='Mnist_Conv_noy', restore=False)

    # Celeb convolution with y labels
    elif scenario == 5:
        y_dim = 40
        model = ModelConvCeleb(batch_size=128, z_dim=50, y_dim=y_dim)
        data = CelebA(mean=True)
        train(model, data, name='Celeb_Conv_y')

    # Celeb convolution without y labels
    elif scenario == 6:
        y_dim = None
        model = ModelConvCeleb(batch_size=128, z_dim=50, y_dim=y_dim)
        data = CelebA(mean=True)
        train(model, data, name='Celeb_Conv_noy')

    # Celeb resnet with y labels
    elif scenario == 7:
        y_dim = 40
        model = ModelResCeleb(batch_size=128, z_dim=50, y_dim=y_dim)
        data = CelebA(mean=True)
        train(model, data, name='Celeb_Res_y', restore = True)

    # Celeb resnet without y labels
    elif scenario == 8:
        y_dim = None
        model = ModelResCeleb(batch_size=128, z_dim=50, y_dim=y_dim)
        data = CelebA(mean=True)
        train(model, data, name='Celeb_Res_noy')

    elif scenario == 9:
        y_dim = 40
        model = ModelResCeleb(batch_size=128, z_dim=50, y_dim=y_dim)
        data = CelebA(mean=True)
        train(model, data, name='Celeb_Subpixel_y')

    elif scenario == 10:
        y_dim = None
        model = ModelResCeleb(batch_size=128, z_dim=50, y_dim=y_dim)
        data = CelebA(mean=True)
        train(model, data, name='Celeb_Subpixel_noy')

    # Mnist hq model with y (Not tested!)
    if scenario == 100:
        model = ModelHqMnist(batch_size=1, z_dim=5, y_dim=y_dim)
        data = MNIST(mean=False)
        train(model, data, name='Mnist_Hq')
